from typing import Dict, Any, List
from ..db.base import BaseEvaluator
from ..core.result import EvaluationResult
from ..prompts import EvaluationPrompts
from langchain_core.output_parsers import JsonOutputParser
import string
from pydantic import BaseModel, Field

class MCQJsonAnswer(BaseModel):
    """객관식 문제 답변을 위한 JSON 형식 정의
    
    Attributes:
        reason (str): 답안 선택에 대한 상세한 설명
        answer (str): 선택한 답안 (A, B, C, D 중 하나)
    """
    reason : str = Field(description="""A detailed step-by-step explanation of why this answer was chosen. \
                     The reasoning should break down the thought process and cite specific \
                     information from the question that led to this conclusion.""")
    answer : str = Field(description="""The selected answer choice, represented as a single uppercase letter
                     (e.g., 'A', 'B', 'C', or 'D').""")


class MCQEvaluator(BaseEvaluator):
    """객관식 문제 평가기 클래스
    
    LLM을 사용하여 객관식 문제의 답안을 생성하고 평가합니다.
    답안 생성 시 선택의 이유도 함께 제공합니다.
    
    Attributes:
        answer_chain: LLM과 JSON 파서를 연결한 처리 파이프라인
        
    Example:
        >>> evaluator = MCQEvaluator(config, teacher_llm, student_llm)
        >>> result = await evaluator.evaluate_single({
        ...     "question": "대한민국의 수도는?",
        ...     "choices": ["서울", "부산", "대전", "광주"],
        ...     "answer": "A"
        ... })
        >>> print(f"정답 여부: {result.is_correct}")
        >>> print(f"선택 이유: {result.metadata['reason']}")
    """
    
    def _initialize_chain(self) -> None:
        """답안 생성을 위한 처리 체인을 초기화합니다.
        
        JSON 형식으로 답안을 파싱하기 위한 파서와 프롬프트를 설정하고,
        이를 LLM과 연결하여 처리 파이프라인을 구성합니다.
        """
        # JSON 형식으로 응답을 파싱하기 위한 파서 초기화
        json_parser = JsonOutputParser(pydantic_object=MCQJsonAnswer)
        # 객관식 문제용 프롬프트 템플릿 가져오기
        prompt = EvaluationPrompts.get_mcq_prompt()
        
        # LLM과 파서를 파이프라인으로 연결
        llm = self.teacher_llm
        self.answer_chain = prompt | llm | json_parser
        
    
    def get_prompt(self) -> str:
        """객관식 문제용 프롬프트 템플릿을 반환합니다."""
        return EvaluationPrompts.get_mcq_prompt()
    
    async def evaluate_single(self, example: Dict[str, Any]) -> EvaluationResult:
        """단일 객관식 문제를 평가합니다.
        
        Args:
            example (Dict[str, Any]): 평가할 문제 데이터
                - question: 문제 텍스트
                - choices: 선택지 리스트
                - answer: 정답 (A, B, C, D)
                
        Returns:
            EvaluationResult: 평가 결과 객체
                - is_correct: 정답 여부
                - predicted: 모델이 선택한 답안과 이유
                - metadata: 선택지 등 부가 정보
        """
        # 입력 데이터 추출
        question = example["question"]
        choices = example["choices"]
        ground_truth = example["answer"]
        
        # 선택지를 
        # "A. 선택지1
        # B. 선택지2"
        # 형식으로 포맷팅
        choices_str = "\n".join(
        f"{letter}. {choice}" 
        for letter, choice in zip(string.ascii_uppercase, choices)
    )
        
        # LLM을 통해 답안 생성
        predicted = await self.answer_chain.ainvoke({
            "question": question,
            "choices": choices_str
        })
        
        
        # 평가 결과 반환
        return EvaluationResult(
            question=question,
            ground_truth=ground_truth,
            predicted=predicted,
            is_correct=predicted['answer'] == ground_truth,
            evaluation_feedback="",  # 객관식은 별도의 피드백이 필요 없음
            metadata={
                "reason" : predicted['reason'],
                "choices": choices,
                "task_type": "mcq"
            }
        )
    
    async def evaluate_batch(self, examples: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """여러 객관식 문제를 배치로 평가합니다.
        
        Args:
            examples (List[Dict[str, Any]]): 평가할 문제 데이터 리스트
            
        Returns:
            List[EvaluationResult]: 각 문제별 평가 결과 리스트
            
        Example:
            >>> batch_results = await evaluator.evaluate_batch([
            ...     {"question": "문제1", "choices": ["보기1", ...], "answer": "A"},
            ...     {"question": "문제2", "choices": ["보기1", ...], "answer": "B"}
            ... ])
            >>> for result in batch_results:
            ...     print(f"정답: {result.ground_truth}, 예측: {result.predicted['answer']}")
        """
        # 배치 처리를 위한 입력 데이터 준비
        
        batch_inputs = []
        
        for example in examples:
            question = example["question"]
            choices = example["choices"]
            
            # 선택지 포맷팅
            choices_str = "\n".join(
                f"{letter}. {choice}" 
                for letter, choice in zip(string.ascii_uppercase, choices)
            )
            
            batch_inputs.append({
                "question": question,
                "choices": choices_str
            })
        
        # 배치로 답안 생성
        predicted_batch = await self.answer_chain.abatch(batch_inputs)
        
        
        # 결과 처리 및 반환
        results = []
        for example, predicted in zip(examples, predicted_batch):
            results.append(EvaluationResult(
                question=example["question"],
                ground_truth=example["answer"],
                predicted=predicted,
                is_correct=predicted['answer'] == example["answer"],
                evaluation_feedback=None,  # 객관식은 별도의 피드백이 필요 없음
                metadata={
                    "reason": predicted['reason'],
                    "choices": example["choices"],
                    "task_type": "mcq"
                }
            ))
        
        return results