from dataclasses import dataclass, asdict
from typing import List, Optional, Union

@dataclass
class LLMTestCase:
    input: str # llm에게 넣은 질문
    output: Optional[str] = None # llm이 실제로 답변한 답
    expected_output: str = None # 원래 갖고 있는 답안
    context: Optional[List[str]] = None # 답을 채점하기 위하여 알고 있어야하는 사실
    retrieval_context: Optional[List[str]] = None # 검색돼서 이 답안을 가지고 답변이 나옴
    reasoning : Optional[str] = None # 답을 생성해기 위한 추론
    choices: Optional[str] = None # MCQtestcase에 사용
    
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
    
    def to_dict(self):
        return asdict(self)