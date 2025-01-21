# LLM 평가 시스템 (LLM Evaluation System)

LLM 기반의 자동 평가 시스템입니다. 다양한 유형의 문제(객관식, 주관식, 대화형)에 대한 LLM의 응답을 자동으로 평가하고 결과를 저장합니다.

## 주요 기능

- 다양한 평가 유형 지원
  - 객관식 문제 평가
  - 주관식 문제 평가
  - 대화형 응답 평가
- 비동기 처리를 통한 효율적인 배치 평가
- 평가 결과의 자동 저장 및 관리
- 유연한 설정 시스템

## 시스템 요구사항

- Python 3.8+
- 필수 패키지:
  - langchain
  - openai
  - pandas
  - datasets
  - asyncio

## 설치 방법

1. 저장소 클론

```bash
git clone https://github.com/your-username/llm-evaluation-system.git
cd llm-evaluation-system
```

2. 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

3. 의존성 설치

```bash
pip install -r requirements.txt
```

## 환경 설정

1. OpenAI API 키 설정

```bash
export OPENAI_API_KEY='your-api-key'
```

## 사용 방법

### 기본 사용 예제

```python
import asyncio
from llm_evaluator.config.evaluation_config import EvaluationConfig
from llm_evaluator.core.evaluator import LLMEvaluator

async def main():
    config = EvaluationConfig(
        dataset_name="your-dataset-name",
        task_type="multichoice",
        evaluation_method="llm",
        evaluation_model="gpt-3.5-turbo",
        score_type="accuracy"
    )

    evaluator = LLMEvaluator(config)
    results_df = await evaluator.run_evaluation()

if __name__ == "__main__":
    asyncio.run(main())
```

### 설정 옵션

| 옵션              | 설명                                       | 기본값        |
| ----------------- | ------------------------------------------ | ------------- |
| dataset_name      | 평가할 데이터셋 이름                       | 필수          |
| task_type         | 평가 유형 (multichoice/multiturn/generate) | 필수          |
| evaluation_method | 평가 방법 (llm/machine/heuristic)          | 필수          |
| evaluation_model  | 평가에 사용할 모델                         | 필수          |
| score_type        | 점수 유형 (accuracy/humiliation/coherence) | 필수          |
| model_name        | 사용할 LLM 모델                            | gpt-3.5-turbo |
| temperature       | 모델 temperature                           | 0.0           |
| batch_size        | 배치 크기                                  | 10            |

## 프로젝트 구조

```
llm-evaluation-project/
├── Readme.md
├── data
│   └── results
├── examples
│   ├── 0_llm_factory.ipynb
│   ├── 1_evaluate_one_case.ipynb
│   ├── 2_Datasets.ipynb
│   ├── run_evaluation.py
│   └── test.py
├── langmetrics
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-311.pyc
│   ├── config
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── db_config.cpython-311.pyc
│   │   │   ├── evaluation_config.cpython-311.pyc
│   │   │   └── model_config.cpython-311.pyc
│   │   ├── db_config.py
│   │   ├── evaluation_config.py
│   │   └── model_config.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── base.cpython-311.pyc
│   │   │   └── result.cpython-311.pyc
│   │   ├── base.py
│   │   └── result.py
│   ├── db
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── base_manager.cpython-311.pyc
│   │   │   ├── duckdb_manager.cpython-311.pyc
│   │   │   ├── manager.cpython-311.pyc
│   │   │   └── server_manager.cpython-311.pyc
│   │   ├── base_manager.py
│   │   ├── duckdb_manager.py
│   │   ├── manager.py
│   │   ├── schema.sql
│   │   └── server_manager.py
│   ├── evaluator
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── binary_evaluator.cpython-311.pyc
│   │   │   └── llm_evaluator.cpython-311.pyc
│   │   ├── base_evaluator.py
│   │   ├── binarychoice_evaluator.py
│   │   ├── llm_evaluator.py
│   │   ├── multichoice_evaluator.py
│   │   ├── multiturn_evaluator.py
│   │   └── open_ended_evaluator.py
│   ├── llmdataset
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── dataset.cpython-311.pyc
│   │   │   ├── datasets.cpython-311.pyc
│   │   │   └── langdataset.cpython-311.pyc
│   │   └── dataset.py
│   ├── llmfactory
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── base_factory.cpython-311.pyc
│   │   │   ├── base_llmfactory.cpython-311.pyc
│   │   │   └── llmfactory.cpython-311.pyc
│   │   ├── base_factory.py
│   │   └── llmfactory.py
│   ├── llmtestcase
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── llmtestcase.cpython-311.pyc
│   │   │   └── testcase.cpython-311.pyc
│   │   └── llmtestcase.py
│   ├── metrics
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── base_metric.cpython-311.pyc
│   │   │   ├── base_result.cpython-311.pyc
│   │   │   ├── base_template.cpython-311.pyc
│   │   │   ├── prompts.cpython-311.pyc
│   │   │   └── utils.cpython-311.pyc
│   │   ├── base_metric.py
│   │   ├── base_result.py
│   │   ├── base_template.py
│   │   ├── mcq_choice
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── mcq_choice.cpython-311.pyc
│   │   │   │   ├── mcq_template.cpython-311.pyc
│   │   │   │   ├── prompt_dict.cpython-311.pyc
│   │   │   │   └── template.cpython-311.pyc
│   │   │   ├── mcq_choice.py
│   │   │   ├── mcq_template.py
│   │   │   └── prompt_dict.py
│   │   ├── medical
│   │   │   ├── __init__.py
│   │   │   └── robustness
│   │   │       ├── __init__.py
│   │   │       ├── __pycache__
│   │   │       │   ├── __init__.cpython-311.pyc
│   │   │       │   ├── mcq_choice.cpython-311.pyc
│   │   │       │   ├── mcq_template.cpython-311.pyc
│   │   │       │   ├── prompt_dict.cpython-311.pyc
│   │   │       │   └── template.cpython-311.pyc
│   │   │       ├── mcq_choice.py
│   │   │       ├── mcq_template.py
│   │   │       └── prompt_dict.py
│   │   ├── prompts.py
│   │   ├── stastical
│   │   └── utils.py
│   ├── prompt_storage
│   │   ├── __init__.py
│   │   ├── mcq_choice_prompt.json
│   │   └── medical_evaluate_prompt.json
│   ├── scorer
│   │   ├── __init__.py
│   │   ├── exact_match_scorer.py
│   │   └── scorer.py
│   ├── tmp.ipynb
│   └── utils
│       ├── __init__.py
│       └── utils.py
├── pyproject.toml
├── requirements.txt
```

## 테스트 실행

```bash
# 전체 테스트 실행
pytest

# 특정 모듈 테스트
pytest tests/test_evaluator.py

# 커버리지 리포트 생성
pytest --cov=llm_evaluator tests/
```

## 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 연락처

- 이메일: your-email@example.com
- 이슈 트래커: https://github.com/your-username/llm-evaluation-system/issues

## 크레딧

이 프로젝트는 다음 오픈소스 프로젝트들을 사용합니다:

- LangChain
- OpenAI API
- Hugging Face Datasets
- pandas
