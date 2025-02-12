from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class LLMResult:
    """평가 결과를 저장하기 위한 데이터 클래스"""
    input : str
    student_answer : str
    teacher_answer : str
    expected_output : str
    context : str
    retrieval_context : str
    reasoning : str
    choices : str
    score : float
    metadata : dict
    
    def to_dict(self) -> dict:
        """결과를 딕셔너리로 변환"""
        return {
            "input": self.input,
            "student_answer" : self.student_answer,
            "teacher_answer" : self.teacher_answer,
            "expected_output" : self.expected_output,
            "context" : self.context,
            "retrieval_context" : self.retrieval_context,
            "reasoning" : self.reasoning,
            "choices" : self.choices,
            "score": self.score,
            "metadata" : self.metadata, 
        }
    
        
    @classmethod
    def from_dict(cls, data: dict) -> 'LLMResult':
        """딕셔너리로부터 LLMResult 객체를 생성
        
        Args:
            data (dict): LLMResult 객체로 변환할 딕셔너리
            
        Returns:
            LLMResult: 생성된 LLMResult 객체
        """
        return cls(
            input=data["input"],
            student_answer=data["student_answer"],
            teacher_answer=data["teacher_answer"], 
            expected_output=data["expected_output"],
            context=data["context"],
            retrieval_context=data["retrieval_context"],
            reasoning=data["reasoning"],
            choices=data["choices"],
            score=data["score"],
            metadata=data["metadata"]
        )
