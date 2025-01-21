from dataclasses import dataclass
from typing import Literal

@dataclass
class EvaluationResult:
    """평가 결과를 저장하기 위한 데이터 클래스"""
    question: str
    predicted: str
    token_used : int
    language : Literal['ko', 'en']
    
    
@dataclass
class JudgeResult(EvaluationResult):
    """MCQ 평가 결과를 저장하기 위한 데이터 클래스"""
    ground_truth : str

    def to_dict(self) -> dict:
        """결과를 딕셔너리로 변환"""
        return {
            "question": self.question,
            "choice" : self.choice,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "is_correct": self.is_correct,
            "language" : self.language,
            "token_used" : self.token_used, 
        }



@dataclass
class BCQResult(EvaluationResult):
    """BCQ 평가 결과를 저장하기 위한 데이터 클래스"""
    ground_truth : str
    is_correct: bool

    def to_dict(self) -> dict:
        """결과를 딕셔너리로 변환"""
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "is_correct": self.is_correct,
            "language" : self.language,
            "token_used" : self.token_used,
        }
        
@dataclass
class ExactMatchResult(EvaluationResult):
    """BCQ 평가 결과를 저장하기 위한 데이터 클래스"""
    ground_truth : str
    is_correct: bool

    def to_dict(self) -> dict:
        """결과를 딕셔너리로 변환"""
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "is_correct": self.is_correct,
            "language" : self.language,
            "token_used" : self.token_used,
        }


@dataclass
class MCQResult(EvaluationResult):
    """MCQ 평가 결과를 저장하기 위한 데이터 클래스"""
    ground_truth : str
    choice : str
    is_correct: bool

    def to_dict(self) -> dict:
        """결과를 딕셔너리로 변환"""
        return {
            "question": self.question,
            "choice" : self.choice,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "is_correct": self.is_correct,
            "language" : self.language,
            "token_used" : self.token_used, 
        }
        
@dataclass
class OpenEndedResult(EvaluationResult):
    """MCQ 평가 결과를 저장하기 위한 데이터 클래스"""
    score : float
    reason : str
    evaluate_prompt : str
    evaluate_prompt_type : str
    
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
    score : float
    reason : str
    evalutate_prompt : str
    evaluate_prompt_type : Literal['Multi-Turn', 'Recollection', 'Refinement', 'Follow-Up']
    
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
        
