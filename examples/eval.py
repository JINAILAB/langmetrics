from langmetrics.llmdataset import LLMDataset
from langmetrics.llmtestcase import LLMTestCase
from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv
from langmetrics.llmfactory import LLMFactory
from langmetrics.config import ModelConfig, LocalModelConfig
from langmetrics.metrics import MCQMetric

load_dotenv(override=True)

async def evaluate_model(lora_path=None, model_name="Qwen/Qwen2.5-32B-Instruct", epoch_label=None):
    """모델을 평가하는 함수
    
    Args:
        lora_path (str, optional): LoRA 모델 경로. None이면 기본 모델 사용
        model_name (str, optional): 모델 이름
        epoch_label (str, optional): 결과 파일 이름에 포함할 에폭 라벨
    
    Returns:
        float: 평가 점수
    """
    # 테스트 데이터셋 로드
    test_dataset = LLMDataset.from_huggingface_hub('sickgpt/015_KorMedMCQA_test', split='train')
    
    # 모델 설정 생성
    if lora_path:
        custom_config = LocalModelConfig(
            model_name=model_name,
            lora_model_path=lora_path,
            lora_name="qwen",
        )
    else:
        custom_config = LocalModelConfig(
            model_name=model_name,
        )
    
    # 모델 생성
    custom_llm = LLMFactory.create_llm(custom_config, temperature=0)
    
    # 메트릭 설정
    metric = MCQMetric(
        output_model=custom_llm,
        template_language='ko',  # 'ko' 또는 'en'
        output_template_type='reasoning'  # 'reasoning' 또는 'only_answer'
    )
    
    # 평가 실행
    results = await metric.ameasure(test_dataset)
    
    # 결과 저장
    filename = f'qwen_32b{f"_{epoch_label}" if epoch_label else ""}.csv'
    results.df.to_csv(filename, index=False)
    
    # 점수 계산
    scores = sum([i.score for i in results]) / len(results)
    print(f'{epoch_label if epoch_label else "기본 모델"}: {scores}')
    
    # 모델 종료
    custom_llm.shutdown()
    
    return scores

async def main():
    # Epoch 0 모델 평가
    await evaluate_model(
        lora_path="/workspace/project/2025/model/llm/sickllm/qwen_32b_lora_ver1/epoch_0",
        epoch_label="epoch0"
    )
    
    # Epoch 1 모델 평가
    await evaluate_model(
        lora_path="/workspace/project/2025/model/llm/sickllm/qwen_32b_lora_ver1/epoch_1",
        epoch_label="epoch1"
    )
    
    # 기본 모델 평가
    await evaluate_model()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())