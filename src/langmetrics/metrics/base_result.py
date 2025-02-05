from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class EvaluationResult:
    """평가 결과를 저장하기 위한 데이터 클래스"""
    question: str
    student_answer: str
    score : float
    metadata : dict
    reasoning : str
    


@dataclass
class MCQResult(EvaluationResult):
    """MCQ 평가 결과를 저장하기 위한 데이터 클래스"""
    ground_truth : str
    choice : str
    
    def __str__(self) -> str:
        result = '✅ 정답' if self.score == 1 else '❌ 오답'
        
        """결과를 문자열로 변환하여 출력"""
        return (
            f"📝 문제: {self.question}\n"
            f"\n"
            f"🤔 LLM 답: {self.student_answer}\n"
            f"📋 선택지: {self.choice}\n"
            f"💡 정답: {self.ground_truth}\n"
            f"\n"
            f"📊 채점 결과: {result}\n"
            f"\n"
            f"💭 추론 과정:\n{self.reasoning}\n"
            f"\n"
            f"ℹ️ 메타데이터: {self.metadata}\n"
            f"{'='*50}"
        )

    def to_dict(self) -> dict:
        """결과를 딕셔너리로 변환"""
        return {
            "question": self.question,
            "choice" : self.choice,
            "ground_truth": self.ground_truth,
            "student_answer": self.student_answer,
            "score": self.score,
            "reasoninging" : self.reasoninging,
            "metadata" : self.metadata, 
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
            student_answer=data["student_answer"],
            score=data["score"],
            reasoning=data["reason"],
            metadata=data["language"],
        )
        
        
@dataclass
class JudgeResult(EvaluationResult):
    """MCQ 평가 결과를 저장하기 위한 데이터 클래스"""
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
            "student_answer": self.student_answer,
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
            student_answer=data["student_answer"],
            score=data["score"],
            reasoning=data["reasoning"],
            language=data["language"],
            token_usage=data.get("token_usage")  # token_usage는 Optional이므로 get 메서드 사용
        )