from dataclasses import dataclass, asdict
from typing import Literal, Optional

@dataclass
class LLMResult:
    """평가 결과를 저장하기 위한 데이터 클래스"""
    input : str 
    output : str = None
    scoring_model_output : str = None
    expected_output : str = None
    context : str = None
    retrieval_context : str = None
    choices : str = None
    score : float = None
    metadata : dict = None
    
    
    def to_dict(self) -> dict:
        """결과를 딕셔너리로 변환"""
        return asdict(self)
    
        
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
            output=data["output"],
            scoring_model_output=data["scoring_model_output"], 
            expected_output=data["expected_output"],
            context=data["context"],
            retrieval_context=data["retrieval_context"],
            choices=data["choices"],
            score=data["score"],
            metadata=data["metadata"]
        )

    def __repr__(self):
        # 각 속성을 딕셔너리로 변환하고 None을 제외
        fields = {k: v for k, v in self.__dict__.items() if v is not None}
        
        # repr 문자열을 생성
        repr_str = f"LLMResult(\n  " + ",\n  ".join(f"{k}={repr(v)}" for k, v in fields.items()) + "\n)"
        return repr_str
