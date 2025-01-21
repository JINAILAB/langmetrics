from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class EvaluationResult:
    """평가 결과를 저장하기 위한 데이터 클래스"""
    question: str
    predicted: str
    language : Literal['ko', 'en']
    score : float
    
    

@dataclass
class BCQResult(EvaluationResult):
    """BCQ 평가 결과를 저장하기 위한 데이터 클래스"""
    ground_truth : str
    token_usage : Optional[int] = None

    def to_dict(self) -> dict:
        """결과를 딕셔너리로 변환"""
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "score": self.score,
            "language" : self.language,
            "token_usage" : self.token_usage,
        }


@dataclass
class MCQResult(EvaluationResult):
    """MCQ 평가 결과를 저장하기 위한 데이터 클래스"""
    ground_truth : str
    choice : str
    reasoning : str
    token_usage : Optional[int] = None
    
    def __str__(self) -> str:
        result = '정답' if self.score == 1 else '오답'
        """결과를 문자열로 변환하여 출력"""
        return f"문제: {self.question}\n" \
                f"선택지: {self.choice}\n" \
                f"정답: {self.ground_truth}\n" \
                f"결과: {result}\n" \
                f"추론: {self.reasoning}\n" \
                f"토큰 사용량: {self.token_usage}"
                

    def to_dict(self) -> dict:
        """결과를 딕셔너리로 변환"""
        return {
            "question": self.question,
            "choice" : self.choice,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "score": self.score,
            "reasoning" : self.reasoning,
            "language" : self.language,
            "token_usage" : self.token_usage, 
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'MCQResult':
        """딕셔너리로부터 MCQResult 객체를 생성

        Args:
            data (dict): MCQResult 객체로 변환할 딕셔너리

        Returns:
            MCQResult: 생성된 MCQResult 객체
        """
        return cls(
            question=data["question"],
            choice=data["choice"],
            ground_truth=data["ground_truth"],
            predicted=data["predicted"],
            score=data["score"],
            reasoning=data["reasoning"],
            language=data["language"],
            token_usage=data.get("token_usage")  # token_usage는 Optional이므로 get 메서드 사용
        )
        
@dataclass
class OpenEndedResult(EvaluationResult):
    """MCQ 평가 결과를 저장하기 위한 데이터 클래스"""
    reason : str
    evaluate_prompt : str
    evaluate_prompt_type : str
    token_usage : Optional[int] = None
    
    def to_dict(self) -> dict:
        """결과를 딕셔너리로 변환"""
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "evaluate_prompt_type" : self.prompt_type,
            "evaluate_prompt" : self.evaluate_prompt,
            "reason": self.reason,
            "score" : self.score,
            "language" : self.language
        }
        
@dataclass
class MultiturnResult(EvaluationResult):
    """MCQ 평가 결과를 저장하기 위한 데이터 클래스"""
    reason : str
    evalutate_prompt : str
    evaluate_prompt_type : Literal['Multi-Turn', 'Recollection', 'Refinement', 'Follow-Up']
    token_usage : Optional[int] = None
    
    def to_dict(self) -> dict:
        """결과를 딕셔너리로 변환"""
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "evaluate_prompt_type" : self.prompt_type,
            "evaluate_prompt" : self.evaluate_prompt,
            "reason": self.reason,
            "score" : self.score,
            "language" : self.language
        }
        
        
@dataclass
class JudgeResult(EvaluationResult):
    """MCQ 평가 결과를 저장하기 위한 데이터 클래스"""
    reasoning : str
    token_usage : Optional[int] = None
    
    def __str__(self) -> str:
        """결과를 문자열로 변환하여 출력"""
        return f"문제: {self.question}\n" \
                f"점수: {self.score}\n" \
                f"추론: {self.reasoning}\n" \
                f"토큰 사용량: {self.token_usage}"
                

    def to_dict(self) -> dict:
        """결과를 딕셔너리로 변환"""
        return {
            "question": self.question,
            "predicted": self.predicted,
            "score": self.score,
            "reasoning" : self.reasoning,
            "language" : self.language,
            "token_usage" : self.token_usage, 
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'MCQResult':
        """딕셔너리로부터 MCQResult 객체를 생성

        Args:
            data (dict): MCQResult 객체로 변환할 딕셔너리

        Returns:
            MCQResult: 생성된 MCQResult 객체
        """
        return cls(
            question=data["question"],
            predicted=data["predicted"],
            score=data["score"],
            reasoning=data["reasoning"],
            language=data["language"],
            token_usage=data.get("token_usage")  # token_usage는 Optional이므로 get 메서드 사용
        )