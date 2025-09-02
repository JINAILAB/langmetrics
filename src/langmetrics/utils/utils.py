from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any
import json
import re
import subprocess
import sys
import time
import requests
import weakref
import threading
import os
import psutil
import signal
import ahocorasick
from typing import List, Set
import tomlkit
import toml

def string_search(dictionary: List[str], text: str) -> Set[str]:
    """
    사전(dictionary)에 있는 여러 단어(패턴)들을 text 안에서 검색해 발견된 단어들의 집합을 반환한다.
    """
    # Aho-Corasick 자동화 구조 생성
    automaton = ahocorasick.Automaton()

    # 사전의 단어들을 트라이에 삽입
    for idx, word in enumerate(dictionary):
        automaton.add_word(word, word)

    # 트라이에 대해 실패 링크 생성
    automaton.make_automaton()

    # 텍스트에서 단어 검색
    found_patterns = set()
    for end_idx, found_word in automaton.iter(text):
        found_patterns.add(found_word)
        
    return found_patterns


def trimAndLoadJson(
    input_string: str,
    metric: Optional[Any] = None,
) -> Any:
    """
    JSON 형식의 문자열을 파싱하여 Python 객체로 변환하는 함수입니다.
    
    처리 순서:
    1) 문자열에서 '{'의 시작 위치와 '}'의 마지막 위치를 찾아 부분 문자열(JSON 추정)을 추출한다.
    2) 만약 '}'가 없다면 하나를 인위적으로 붙여 준다.
    3) 추출한 문자열에서 객체나 배열 끝의 불필요한 쉼표(,)를 제거한다.
    4) 제어 문자(특히 0x00~0x1F, 0x7F)를 제거하여 JSON 디코딩 에러를 최소화한다.
    5) 최종 정제된 문자열을 json.loads로 파싱하여 결과를 반환한다.
    
    매개변수:
        input_string (str): JSON이 포함된 문자열
        metric (Optional[Any]): 메트릭 객체 (옵션)
    
    반환값:
        Any: 파싱된 Python 객체
    
    예외:
        ValueError: JSON 형식이 유효하지 않을 경우
        Exception: 예상치 못한 에러 발생 시
    """
    if not isinstance(input_string, str):
        raise ValueError("Input must be a string.")

    input_string = input_string.strip()
    if not input_string:
        raise ValueError("Input string is empty.")

    # 1) '{'와 '}' 위치 찾기
    start = input_string.find("{")
    end_pos = input_string.rfind("}")

    if start == -1:
        raise ValueError("No opening brace '{' found in input string.")

    # 2) '}'가 없는 경우 뒤에 하나 추가
    if end_pos == -1:
        input_string += "}"
        end_pos = len(input_string) - 1

    # 부분 문자열 추출
    json_str = input_string[start : end_pos + 1]

    # 3) 불필요한 쉼표 제거
    #   예: {"key": "value",} --> {"key": "value"}
    #       [1,2,] --> [1,2]
    json_str = re.sub(r",(\s*[\]}])", r"\1", json_str)

    # 4) 제어 문자(ASCII 0~31, 127) 제거 (JSON에서 유효하지 않음)
    #    - 예: \x00, \x1F, \x7F 등
    #    - 필요시 \n, \r, \t 등을 보존하거나 이스케이프 처리할 수도 있음.
    json_str = re.sub(r"[\x00-\x1F\x7F]", "", json_str)

    # 5) 로드 시도
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        error_msg = (
            "Evaluation LLM outputted an invalid JSON. "
            "Please use a better evaluation model."
        )
        if metric is not None:
            setattr(metric, "error", error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred while parsing JSON: {str(e)}") from e
    
    
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
    
def save_json(data, file_path, indent=4, ensure_ascii=False):
    """
    데이터를 JSON 파일로 저장하는 함수
    
    Parameters:
        data: 저장할 데이터 (dict)
        file_path: 저장할 파일 경로 (str)
        indent: JSON 들여쓰기 칸 수 (int)
        ensure_ascii: ASCII 문자만 사용할지 여부 (bool)
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        print(f"Successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

def save_toml(file_path: str, data: dict):
    """가독성 높은 TOML 포맷으로 저장"""
    toml_doc = tomlkit.document()
    def process_data(data):
        if isinstance(data, dict):
            result = tomlkit.table()
            for key, value in data.items():
                result[key] = process_data(value)
            return result
        elif isinstance(data, list):
            result = tomlkit.array()
            for item in data:
                result.append(process_data(item))
            return result
        elif isinstance(data, str) and '\n' in data:
            # 줄바꿈이 있는 문자열을 멀티라인 문자열로 변환
            return tomlkit.string(data, multiline=True)
        else:
            return data

    # JSON 데이터를 TOML 구조로 변환
    for key, value in data.items():
        toml_doc[key] = process_data(value)

    # TOML 파일 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(tomlkit.dumps(toml_doc))
        
    print(f"TOML 저장 완료: {file_path}")
    

def load_toml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return toml.load(f)
        
        
def execute_shell_command(command: str, logging=False) -> subprocess.Popen:
    """
    Execute a shell command and return the process handle.

    이 함수는 백슬래시로 이어진 줄바꿈(라인 컨티뉴에이션) 및 남은 백슬래시를 제거하고,
    불필요한 공백을 정리하여 명령어를 실행합니다.

    Args:
        command: 실행할 셸 명령어 문자열 (백슬래시로 줄바꿈이 가능)

    Returns:
        subprocess.Popen: 실행된 명령어의 프로세스 핸들
    """
    # 백슬래시+줄바꿈을 공백으로 치환한 후, 남은 백슬래시도 모두 공백으로 치환
    command = re.sub(r'\\\s*\n', ' ', command)
    command = re.sub(r'\\', ' ', command)
    # 연속된 공백을 하나의 공백으로 줄이고 양쪽 공백 제거
    command = re.sub(r'\s+', ' ', command).strip()
    
    if logging:
        return subprocess.Popen(command, shell=True, text=True, stdout=subprocess.stdout, stderr=subprocess.stdout)
    else:
        return subprocess.Popen(command, shell=True, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,)

def wait_for_server(base_url: str, timeout: int = None) -> None:
    """서버의 /v1/models 엔드포인트를 주기적으로 확인하여 준비될 때까지 기다립니다.

    인자:
        base_url: 서버의 기본 URL
        timeout: 기다릴 최대 시간(초 단위). None이면 무한히 기다림.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                time.sleep(5)
                print(
                    """\n
                    참고: 보통 서버는 별도의 터미널에서 실행됩니다.
                    현재 CI 병렬 환경에서 실행 중이므로 실제 성능과는 차이가 있을 수 있습니다.
                    """
                )
                break

            if timeout and time.time() - start_time > timeout:
                raise TimeoutError("지정된 시간 내에 서버가 준비되지 않았습니다.")
        except requests.exceptions.RequestException:
            time.sleep(1)
            
            
def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """프로세스와 모든 자식 프로세스를 종료합니다."""
    # 로그 스팸 방지를 위해 SIGCHLD 핸들러를 제거합니다.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    # parent_pid가 None이면 현재 프로세스를 기준으로 설정합니다.
    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return  # 프로세스가 없으면 종료합니다.

    # 모든 자식 프로세스를 찾아 종료합니다.
    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue  # 지정된 PID는 건너뜁니다.
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass  # 이미 종료된 프로세스는 무시합니다.

    # 부모 프로세스도 종료할지 여부를 확인합니다.
    if include_parent:
        if parent_pid == os.getpid():
            sys.exit(0)  # 현재 프로세스라면 바로 종료합니다.

        try:
            itself.kill()
            # SIGKILL로 종료되지 않는 경우 추가로 SIGQUIT 신호를 보냅니다.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass  # 이미 종료된 경우 무시합니다.
        
            
def terminate_process(process):
    """
    Terminate the process and automatically release the reserved port.
    """
    process_socket_map = weakref.WeakKeyDictionary()
    kill_process_tree(process.pid)

    lock_socket = process_socket_map.pop(process, None)
    if lock_socket is not None:
        release_port(lock_socket)
        
def release_port(lock_socket):
    """
    Release the reserved port by closing the lock socket.
    """
    try:
        lock_socket.close()
    except Exception as e:
        print(f"Error closing socket: {e}")