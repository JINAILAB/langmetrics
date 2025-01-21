import pytest
from langmetrics.config.evaluation_config import EvaluationConfig

@pytest.fixture
def sample_config():
    """테스트용 설정 fixture"""
    return EvaluationConfig(
        dataset_name="test-dataset",
        task_type="multichoice",
        evaluation_method="llm",
        evaluation_model="gpt-3.5-turbo",
        score_type="accuracy"
    )

@pytest.fixture
def sample_evaluation_data():
    """테스트용 평가 데이터 fixture"""
    return {
        "question": "What is the capital of France?",
        "choices": ["London", "Berlin", "Paris", "Madrid"],
        "answer": "3"
    }
