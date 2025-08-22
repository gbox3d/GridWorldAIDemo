#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal threaded STT while playing continuously (client-only)
- Keeps the main loop simple: play chunk → fire-and-forget STT thread
- Client-side 16 kHz / mono / 16-bit conversion
- No server change. Uses stt_client.STTClient
- Prints per-chunk RTT when each STT thread finishes
- Joins threads at the end and prints latency summary

Usage
  pip install yt_dlp pydub python-dotenv
  # ffmpeg 설치 및 PATH에 ffplay 포함

  # .env
  # ASR_HOST=127.0.0.1
  # ASR_PORT=22270
  # ASR_CHECKCODE=20250218

  python youtube_stt_min_threaded.py "<YouTube URL>" --chunk-ms 3000 [--no-play]
"""
import os, io, sys, time, argparse, shutil, subprocess, statistics, threading, traceback
from typing import Optional, List, Tuple

from pydub import AudioSegment
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
import yt_dlp

from stt_client import STTClient  # 교수님이 쓰시는 기존 클라이언트  fileciteturn3file12

# ------------------------ utils ------------------------
def load_env() -> Tuple[str, int, int]:
    if load_dotenv is not None:
        load_dotenv()
    host = os.getenv("ASR_HOST", "127.0.0.1")
    port = int(os.getenv("ASR_PORT", "22270"))
    checkcode = int(os.getenv("ASR_CHECKCODE", "20250218"))
    return host, port, checkcode


def download_youtube_wav(url: str, out_name="temp_full_audio.wav") -> str:
    base = os.path.splitext(out_name)[0]
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "outtmpl": base,
        "quiet": False,
    }
    print(f"[INFO] 전체 오디오 다운로드 시작: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    if not os.path.exists(out_name):
        raise RuntimeError("WAV 파일 생성 실패")
    print(f"[SUCCESS] 전체 오디오 다운로드 완료: {out_name}")
    return out_name


def play_via_ffplay(seg: AudioSegment, sr=16000):
    """Fire-and-forget-ish playback via a separate ffplay process (blocking until the chunk ends)."""
    if shutil.which("ffplay") is None:
        print("[WARN] ffplay를 찾을 수 없습니다. (ffmpeg 설치/경로 확인)")
        return
    seg = seg.set_frame_rate(sr).set_channels(1).set_sample_width(2)
    bio = io.BytesIO(); seg.export(bio, format="wav"); data = bio.getvalue()
    try:
        p = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        p.stdin.write(data); p.stdin.close()
        p.wait(timeout=(len(seg)/1000.0)+1.5)
    except subprocess.TimeoutExpired:
        p.kill(); print("[WARN] ffplay 재생 타임아웃으로 종료")


# ------------------------ minimal threaded STT ------------------------
def stt_worker(client: STTClient, idx: int, start_ms: int, end_ms: int, wav_bytes: bytes,
               latencies: List[float], lat_lock: threading.Lock):
    t0 = time.perf_counter()
    done = threading.Event(); holder = {"text": "", "err": None}
    def cb(text, err):
        holder["text"] = text or ""; holder["err"] = err; done.set()
    try:
        client.recognize_audio(wav_bytes, cb, format_code=1)
        if not done.wait(timeout=120.0):
            raise TimeoutError("STT 요청 타임아웃")
        if holder["err"] is not None:
            raise holder["err"]
        rtt = (time.perf_counter() - t0) * 1000.0
        with lat_lock:
            latencies.append(rtt)
        print(f"[LAT]  chunk #{idx}: {rtt:.1f} ms  ({start_ms/1000:.2f}~{end_ms/1000:.2f}s)")
        print("----------- [ STT 결과 ] -----------\n" + holder["text"] + "\n------------------------------------")
    except Exception as e:
        rtt = (time.perf_counter() - t0) * 1000.0
        with lat_lock:
            latencies.append(rtt)
        print(f"[ERROR] chunk #{idx} ({start_ms/1000:.2f}~{end_ms/1000:.2f}s): {e}")


def main():
    ap = argparse.ArgumentParser(description="YouTube → 16k minimal threaded STT while playing")
    ap.add_argument("url", type=str)
    ap.add_argument("--chunk-ms", type=int, default=3000)
    ap.add_argument("--no-play", action="store_true")
    args = ap.parse_args()

    host, port, checkcode = load_env()
    print(f"[INFO] STT 타겟: {host}:{port}, checkcode={checkcode}")
    client = STTClient(host=host, port=port, checkcode=checkcode)

    try:
        ok = client.ping(); print(f"[INFO] 서버 핑: {'OK' if ok else 'NG'}")
    except Exception as e:
        print(f"[WARN] ping 실패(무시): {e}")

    wav_path = None
    threads: List[threading.Thread] = []
    latencies: List[float] = []; lat_lock = threading.Lock()

    try:
        wav_path = download_youtube_wav(args.url)
        sound = AudioSegment.from_wav(wav_path)
        total_ms = len(sound)

        for idx, start in enumerate(range(0, total_ms, args.chunk_ms), start=1):
            part = sound[start : start + args.chunk_ms]
            print(f"[DEBUG] chunk #{idx} {start/1000:.2f}s ~ {(start+len(part))/1000:.2f}s")

            # 1) (optional) play immediately (blocking per chunk, but STT runs in separate thread)
            if not args.no_play:
                # STT thread is independent, so playback won't wait for server
                pass

            # 2) prepare 16k/mono WAV bytes for STT thread
            part16 = part.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            bio = io.BytesIO(); part16.export(bio, format="wav"); wav_bytes = bio.getvalue()

            # 3) launch STT thread (fire-and-forget; we join after loop)
            t = threading.Thread(target=stt_worker,
                                 args=(client, idx, start, start+len(part), wav_bytes, latencies, lat_lock),
                                 daemon=True)
            t.start(); threads.append(t)

            # 4) Play the chunk while STT thread is running
            if not args.no_play:
                play_via_ffplay(part)

        # join all STT threads
        for t in threads:
            t.join()

        # summary
        if latencies:
            avg = statistics.mean(latencies); p50 = statistics.median(latencies)
            try:
                p95 = statistics.quantiles(latencies, n=100)[94]
            except Exception:
                p95 = max(latencies)
            print("\n===== Latency Summary (ms) =====")
            print(f"count={len(latencies)} | avg={avg:.1f} | p50={p50:.1f} | p95={p95:.1f} | max={max(latencies):.1f}")
            print("================================")

    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단")
    except Exception:
        print("[FATAL] 예외 발생:")
        traceback.print_exc()
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path); print(f"[INFO] 임시 파일 삭제: {wav_path}")
            except Exception:
                pass

if __name__ == "__main__":
    main()
