import pytest
from llm_evaluator.config.evaluation_config import EvaluationConfig

def test_config_creation(sample_config):
    """설정 객체 생성 테스트"""
    assert sample_config.dataset_name == "test-dataset"
    assert sample_config.task_type == "multichoice"
    assert sample_config.model_name == "gpt-3.5-turbo"

def test_config_to_dict(sample_config):
    """설정 객체 직렬화 테스트"""
    config_dict = sample_config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict["dataset_name"] == "test-dataset"

def test_config_from_dict():
    """설정 객체 역직렬화 테스트"""
    config_dict = {
        "dataset_name": "test-dataset",
        "task_type": "multichoice",
        "evaluation_method": "llm",
        "evaluation_model": "gpt-3.5-turbo",
        "score_type": "accuracy"
    }
    config = EvaluationConfig.from_dict(config_dict)
    assert config.dataset_name == "test-dataset"
