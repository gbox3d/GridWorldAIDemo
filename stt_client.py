"""
STT 클라이언트 헬퍼 모듈

이 모듈은 STT(Speech-to-Text) 서버와 통신하는 클라이언트 헬퍼를 제공합니다.
스레드 기반 구현과 asyncio 기반 구현을 모두 제공하여 다양한 환경에서 사용할 수 있습니다.
"""

import socket
import struct
import threading
from typing import Callable, Optional, Any


# 스레드 기반 STT 클라이언트
class STTClient:
    """
    스레드 기반 STT 클라이언트
    
    각 STT 요청을 별도의 스레드에서 처리하고, 콜백 함수를 통해 결과를 반환합니다.
    
    장점:
    - 기존 동기 코드와 쉽게 통합 가능
    - 별도의 실행 환경 설정 없이 사용 가능
    
    단점:
    - 많은 수의 동시 요청 처리 시 스레드 오버헤드 발생 가능
    """
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, checkcode: Optional[int] = None):
        """STT 클라이언트 초기화
        
        Args:
            host: STT 서버 호스트 (None일 경우 환경변수에서 로드)
            port: STT 서버 포트 (None일 경우 환경변수에서 로드)
            checkcode: STT 서버 체크코드 (None일 경우 환경변수에서 로드)
        """
        # 환경변수 로드
        # load_dotenv()
        
        # 서버 설정
        self.server_host = host or "localhost"
        self.server_port = port or 4270
        self.checkcode = checkcode or 20250218
        
        print(f"[INFO] STTClient 초기화 완료: {self.server_host}:{self.server_port}, checkcode: {self.checkcode}")
        
        self._threads = []
    
    def ping(self) -> bool:
        """STT 서버에 ping 요청을 보내 연결 상태 확인
        
        Returns:
            bool: 서버가 정상적으로 응답하면 True, 그렇지 않으면 False
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((self.server_host, self.server_port))
                # 요청 패킷 생성: checkcode(4 bytes) + code(4 bytes)
                # checkcode = 20250122
                code = 99  # Ping 요청
                packet = struct.pack('!ii', self.checkcode, code)
                client_socket.sendall(packet)

                # 응답 패킷 수신
                response = client_socket.recv(9)  # checkcode(4) + code(4) + status_code(1)
                if len(response) != 9:
                    print("[ERROR] 응답 패킷 길이가 잘못되었습니다.")
                    return False
                else:
                    res_checkcode, res_code, status_code = struct.unpack('!iiB', response)
                    print(f"[INFO] 응답 - checkcode: {res_checkcode}, code: {res_code}, status_code: {status_code}")
                    if res_checkcode != self.checkcode or res_code != code:
                        print("[ERROR] 응답 패킷이 예상과 다릅니다.")
                        return False
                    elif status_code == 0:
                        # print("[INFO] STT 서버 연결 성공")
                        return True
                    
        except Exception as e:
            print(f"[ERROR] Ping 요청 실패: {e}")
            return False
    
    def recognize_file(self, file_path: str, callback: Callable[[str, Optional[Exception]], Any]) -> None:
        """파일에서 오디오를 읽어 STT 요청을 비동기로 수행
        
        Args:
            file_path: 오디오 파일 경로
            callback: 결과를 반환할 콜백 함수. 첫 번째 인자는 인식된 텍스트, 두 번째 인자는 예외(발생시)
        """
        thread = threading.Thread(target=self._process_file, args=(file_path, callback))
        thread.daemon = True  # 메인 스레드가 종료되면 이 스레드도 종료
        self._threads.append(thread)
        thread.start()
    
    def recognize_audio(self, audio_data: bytes, callback: Callable[[str, Optional[Exception]], Any], format_code: int = 1) -> None:
        """오디오 데이터로부터 STT 요청을 비동기로 수행
        
        Args:
            audio_data: 오디오 데이터 바이트
            format_code: 오디오 포맷 코드 (1: wav, 2: mp3, 3: webm 등)
            callback: 결과를 반환할 콜백 함수. 첫 번째 인자는 인식된 텍스트, 두 번째 인자는 예외(발생시)
        """
        thread = threading.Thread(target=self._process_audio, args=(audio_data, format_code, callback))
        thread.daemon = True
        self._threads.append(thread)
        thread.start()
    
    def _process_file(self, file_path: str, callback: Callable[[str, Optional[Exception]], Any]) -> None:
        """파일에서 오디오를 읽어 STT 요청 처리 (내부 메서드)"""
        try:
            with open(file_path, "rb") as f:
                audio_data = f.read()
            self._process_audio(audio_data, 1, callback)  # 기본 포맷 코드 1 (wav)
        except Exception as e:
            callback(None, e)
    
    def _process_audio(self, audio_data: bytes, format_code: int, callback: Callable[[str, Optional[Exception]], Any]) -> None:
        """오디오 데이터로부터 STT 요청 처리 (내부 메서드)"""
        try:
            # STT 요청 패킷 구성
            request_code = 0x01  # STT 요청
            header = struct.pack('!ii', self.checkcode, request_code)
            format_code_bytes = struct.pack('!B', format_code)
            audio_length = len(audio_data)
            audio_length_bytes = struct.pack('!i', audio_length)
            packet = header + format_code_bytes + audio_length_bytes + audio_data
            
            # 서버로 패킷 전송 및 응답 수신
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((self.server_host, self.server_port))
                
                # STT 요청 패킷 전송
                client_socket.sendall(packet)
                
                # 응답 헤더 수신: checkcode(4) + request_code(4) + status(1)
                response_header = client_socket.recv(9)
                if len(response_header) != 9:
                    raise ValueError("응답 헤더 길이가 잘못되었습니다.")
                
                res_checkcode, res_request_code, status_code = struct.unpack('!iiB', response_header)
                if status_code != 0:
                    raise ValueError(f"서버에서 오류 발생. status: {status_code}")
                
                # 성공일 경우, 다음 4바이트로 텍스트 길이 정보를 받음
                response_length_bytes = client_socket.recv(4)
                if len(response_length_bytes) != 4:
                    raise ValueError("응답 텍스트 길이 정보 수신 실패.")
                text_length = struct.unpack('!i', response_length_bytes)[0]
                
                # 텍스트 데이터 수신
                recognized_text_bytes = b""
                remaining = text_length
                while remaining > 0:
                    chunk = client_socket.recv(remaining)
                    if not chunk:
                        break
                    recognized_text_bytes += chunk
                    remaining -= len(chunk)
                
                if len(recognized_text_bytes) != text_length:
                    raise ValueError("텍스트 데이터 수신 길이 오류.")
                
                recognized_text = recognized_text_bytes.decode('utf-8')
                callback(recognized_text, None)
        except Exception as e:
            callback(None, e)
    
    def wait(self) -> None:
        """모든 비동기 요청이 완료될 때까지 대기"""
        for thread in self._threads:
            thread.join()
        self._threads = []

