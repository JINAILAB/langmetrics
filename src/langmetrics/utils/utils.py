from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any
import json
import re
import subprocess
import sys


class ResultFileHandler:
    """평가 결과 파일 처리를 위한 유틸리티 클래스"""
    
    def __init__(self, base_dir: str = "data/results"):
        self.result_dir = Path(base_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.result_file = self.result_dir / "evaluation_results.csv"
        self._initialize_result_file()
    
    def _initialize_result_file(self) -> None:
        """결과 파일 초기화"""
        if not self.result_file.exists():
            headers = [
                "evaluation_date",
                "dataset_name",
                "task_type",
                "evaluation_method",
                "evaluation_model",
                "score_type",
                "total_samples",
                "score"
            ]
            pd.DataFrame(columns=headers).to_csv(self.result_file, index=False)
    
    def save_results(self, results: Dict) -> None:
        """새로운 평가 결과 저장"""
        results["evaluation_date"] = datetime.now().strftime("%Y-%m-%d")
        pd.DataFrame([results]).to_csv(
            self.result_file, 
            mode='a', 
            header=False, 
            index=False
        )



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
    
    
def load_prompts_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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
        
        
def execute_shell_command(command: str) -> subprocess.Popen:
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

    return subprocess.Popen(command, shell=True, text=True, stderr=subprocess.STDOUT)