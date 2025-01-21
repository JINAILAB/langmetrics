

# tests/test_file_handler.py
import pytest
from pathlib import Path
from lang_evaluator.utils.file_handler import ResultFileHandler

def test_file_handler_initialization(tmp_path):
    """파일 핸들러 초기화 테스트"""
    handler = ResultFileHandler(base_dir=str(tmp_path))
    assert handler.result_file.exists()
    assert handler.result_file.is_file()

def test_save_results(tmp_path):
    """결과 저장 테스트"""
    handler = ResultFileHandler(base_dir=str(tmp_path))