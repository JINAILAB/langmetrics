import pytest
import asyncio
from lang_evaluator.evaluator import LLMEvaluator
from lang_evaluator.core.result import EvaluationResult

@pytest.mark.asyncio
async def test_single_evaluation(sample_config, sample_evaluation_data):
    """단일 평가 테스트"""
    evaluator = LLMEvaluator(sample_config)
    result = await evaluator.evaluate_single(sample_evaluation_data)
    
    assert isinstance(result, EvaluationResult)
    assert result.question == sample_evaluation_data["question"]
    assert result.ground_truth == sample_evaluation_data["answer"]
    assert isinstance(result.is_correct, bool)

# @pytest.mark.asyncio
# async def test_batch_evaluation(sample_config):
#     """배치 평가 테스트"""
#     evaluator = LLMEvaluator(sample_config)
#     batch_data = [sample_evaluation_data] * 3  # 3개의 동일한 예제로 배치 생성
    
#     results = await evaluator.evaluate_batch(batch_data)
#     assert len(results) == 3
#     assert all(isinstance(r, EvaluationResult) for r in results)