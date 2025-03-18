from dataclasses import dataclass, asdict
from typing import List, Optional, Union

@dataclass
class LLMTestCase:
    input: str # llm에게 넣은 질문
    output: Optional[str] = None # llm이 실제로 답변한 답
    expected_output: str = None # 원래 갖고 있는 답안
    context: Optional[List[str]] = None # 답을 채점하기 위하여 알고 있어야하는 사실
    retrieval_context: Optional[List[str]] = None # 검색돼서 이 답안을 가지고 답변이 나옴
    choices: Optional[str] = None # MCQtestcase에 사용
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LLMTestCase':
        """딕셔너리로부터 LLMTestCase 객체를 생성
        
        Args:
            data (dict): LLMTestCase 객체로 변환할 딕셔너리
            
        Returns:
            LLMTestCase: 생성된 LLMTestCase 객체
        """
        return cls(
            input=data["input"],
            output=data["output"],
            expected_output=data["expected_output"],
            context=data["context"],
            retrieval_context=data["retrieval_context"],
            choices=data["choices"]
        )
        
    def __repr__(self):
        # 각 속성을 딕셔너리로 변환하고 None을 제외
        fields = {k: v for k, v in self.__dict__.items() if v is not None}
        
        # repr 문자열을 생성
        repr_str = f"LLMTestCase(\n  " + ",\n  ".join(f"{k}={repr(v)}" for k, v in fields.items()) + "\n)"
        return repr_str
