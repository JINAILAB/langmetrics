# LangEvaluate

LangEvaluate는 LLM(Large Language Model)의 성능을 평가하기 위한 Python 라이브러리입니다. 다양한 평가 메트릭과 데이터셋 관리 기능을 제공하여 LLM의 성능을 체계적으로 분석할 수 있습니다.

## 주요 기능

- **다양한 LLM 지원**
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude)
  - Naver (Clova)
  - DeepSeek
  - 로컬 GPU 모델

- **다양한 평가 유형**
  - 객관식 문제 (MCQ)
  - 이진 선택 문제
  - 주관식 문제
  - 다중 턴 대화

- **데이터셋 관리**
  - Hugging Face 데이터셋 통합
  - 커스텀 데이터셋 지원
  - 데이터셋 변환 및 전처리

- **평가 메트릭**
  - 정확도 (Accuracy)
  - BLEU, ROUGE 스코어
  - LLM 기반 평가
  - 사용자 정의 메트릭

## 설치 방법

sglang이 라이브러리를 설치하려면 requirements.txt를 설치해야합니다.
만약에 linux 체제가 아니라면 pip install sglang을 해주세요.

```bash
pip install -r requirements
pip install -e .
```

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## todo

- evaluate으로 여러개의 metric 한번에 돌릴 수 있게하기
- benchmark dataset 추가 + 코드 짜기
