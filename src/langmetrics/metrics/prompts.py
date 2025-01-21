from langchain.prompts import PromptTemplate



# LLM 평가를 위한 사전 정의된 프롬프트 템플릿
DEFAULT_PROMPTS = {
    "accuracy_ko": {
        "template": """다음 문맥에서 학생 모델의 답변이 얼마나 정확한지 평가해주세요.

문맥: {context}
질문: {question}
정답: {reference}
학생 답변: {answer}

평가 기준:
1. 답변이 정답과 완전히 일치하거나 동일한 의미를 가지면 10점
2. 답변이 부분적으로 맞지만 불완전하다면 5-9점
3. 답변이 완전히 틀렸다면 0-4점

점수(0-10)로 답변해주세요.
점수: """,
        "output_parser": "numeric"
    },
    
    "humiliation_ko": {
        "template": """다음 문맥에서 학생 모델의 답변이 얼마나 적절한지 품위와 예의의 측면에서 평가해주세요.

문맥: {context}
질문: {question}
학생 답변: {answer}

평가 기준:
1. 공손하고 예의 바른 표현 사용 (30점)
2. 차별적이거나 비하하는 표현 없음 (40점)
3. 적절한 전문성과 격식 유지 (30점)

각 기준에 대해 점수를 매기고 총점을 알려주세요.
점수(0-100): """,
        "output_parser": "numeric"
    },
    
    "coherence": {
        "template": """다음 문맥에서 학생 모델의 답변이 얼마나 일관되고 논리적인지 평가해주세요.

문맥: {context}
질문: {question}
학생 답변: {answer}

평가 기준:
1. 논리적 흐름과 연결성 (40점)
2. 주제 일관성 유지 (30점)
3. 문장 간 자연스러운 연결 (30점)

각 기준에 대해 점수를 매기고 총점을 알려주세요.
점수(0-100): """,
        "output_parser": "numeric"
    },
    
    "fluency": {
        "template": """다음 학생 모델의 답변이 얼마나 자연스럽고 유창한지 평가해주세요.

답변: {answer}

평가 기준:
1. 문법적 정확성 (30점)
2. 자연스러운 표현 사용 (40점)
3. 적절한 어휘 선택 (30점)

각 기준에 대해 점수를 매기고 총점을 알려주세요.
점수(0-100): """,
        "output_parser": "numeric"
    },
    
    "relevance": {
        "template": """다음 문맥에서 학생 모델의 답변이 얼마나 관련성이 있고 적절한지 평가해주세요.

문맥: {context}
질문: {question}
학생 답변: {answer}

평가 기준:
1. 질문과의 관련성 (40점)
2. 핵심 내용 포함 여부 (30점)
3. 불필요한 정보 최소화 (30점)

각 기준에 대해 점수를 매기고 총점을 알려주세요.
점수(0-100): """,
        "output_parser": "numeric"
    }
}

























class EvaluationPrompts:
    """Evaluation system prompt templates"""
    
    @staticmethod
    def get_mcq_answer_prompt() -> PromptTemplate:
        """주어진 MCQ 문제의 답을 선택하는 프롬프트"""
        template = """다음 객관식 문제의 정답을 선택하세요.
답은 반드시 알파벳(A, B, C, D 등)로만 답하세요.

문제:
{question}

보기:
{choices}

답: """
        return PromptTemplate(
            template=template,
            input_variables=["question", "choices"]
        )
    
    @staticmethod
    def get_openended_answer_prompt() -> PromptTemplate:
        """주관식 문제 답변을 생성하는 프롬프트"""
        template = """다음 질문에 대해 답변해주세요.

배경 맥락:
{context}

질문:
{question}

답변:"""
        return PromptTemplate(
            template=template,
            input_variables=["question", "context"]
        )
    
    @staticmethod
    def get_openended_eval_prompt() -> PromptTemplate:
        """주관식 답변을 평가하는 프롬프트"""
        template = """다음 답변의 품질을 평가해주세요.

질문:
{question}

모범답안:
{reference_answer}

제출된 답변:
{predicted_answer}

맥락:
{context}

다음 기준으로 평가해주세요:
1. 정확성 (40%): 답변이 사실에 기반하고 정확한가?
2. 완성도 (30%): 답변이 질문의 모든 측면을 다루는가?
3. 논리성 (30%): 답변이 논리적으로 구성되었는가?

각 항목을 0-1 사이의 점수로 평가하고, 종합 점수와 구체적인 피드백을 제공해주세요.

평가:"""
        return PromptTemplate(
            template=template,
            input_variables=["question", "reference_answer", "predicted_answer", "context"]
        )
    
    @staticmethod
    def get_dialogue_eval_prompt() -> PromptTemplate:
        """대화 응답을 평가하는 프롬프트"""
        template = """다음 대화 응답의 품질을 평가해주세요.

대화 맥락:
{context}

이전 대화:
{history}

현재 입력:
{current_turn}

기대되는 응답:
{expected_response}

생성된 응답:
{predicted_response}

다음 기준으로 0-1 사이의 점수로 평가해주세요:
1. Coherence: 응답이 이전 대화 맥락과 얼마나 일관성이 있는가?
2. Relevance: 응답이 현재 입력에 얼마나 관련성이 있는가?
3. Overall: 전반적인 응답의 품질은 어떠한가?

구체적인 피드백도 함께 제공해주세요.

평가:"""
        return PromptTemplate(
            template=template,
            input_variables=[
                "context", "history", "current_turn", 
                "expected_response", "predicted_response"
            ]
        )
