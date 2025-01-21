from dataclasses import dataclass
from typing import List, Optional, Union

@dataclass
class LLMTestCase:
    input: str # llm에게 넣은 질문
    expected_output: str # 원래 갖고 있는 답안
    output: Optional[str] = None # llm이 실제로 답변한 답
    context: Optional[List[str]] = None # 답을 채점하기 위하여 알고 있어야하는 사실
    retrieval_context: Optional[List[str]] = None # 검색돼서 이 답안을 가지고 답변이 나옴
    reasoning : Optional[str] = None# 답을 생성해기 위한 추론
    
    def __post_init__(self):
        # Ensure `context` is None or a list of strings
        if self.context is not None:
            if not isinstance(self.context, list) or not all(
                isinstance(item, str) for item in self.context
            ):
                raise TypeError("'context' must be None or a list of strings")

        # Ensure `retrieval_context` is None or a list of strings
        if self.retrieval_context is not None:
            if not isinstance(self.retrieval_context, list) or not all(
                isinstance(item, str) for item in self.retrieval_context
            ):
                raise TypeError(
                    "'retrieval_context' must be None or a list of strings"
                )
                
@dataclass
class MCQTestCase:
    input: str # llm에게 넣은 질문
    choices :List[str] # 선택지 리스트 형식
    expected_output: Union[int, str] # 원래 갖고 있는 답안
    output: Optional[str] = None # llm이 실제로 답변한 답
    reasoning : Optional[str] = None # 답을 생성해기 위한 추론

    def __post_init__(self):
        # Ensure `context` is None or a list of strings
        if not isinstance(self.choices, list) or not all(
            isinstance(item, str) for item in self.choices
        ):
            raise TypeError("'choices' must be a list of strings")
        
        
@dataclass
class JudgeTestCase:
    input: str # llm에게 넣은 질문
    output: Optional[str] = None # llm이 실제로 답변한 답
    reasoning : Optional[str] = None # 답을 생성해기 위한 추론
    

@dataclass
class BCQTestCase:
    input: str # llm에게 넣은 질문
    expected_output: str # 원래 갖고 있는 답안
    output: Optional[str] = None # llm이 실제로 답변한 답 ex) yes or no
    reasoning : Optional[str] = None  # 답을 생성해기 위한 추론

