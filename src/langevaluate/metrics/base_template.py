from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate

class BaseTemplate(ABC):
    """
    프롬프트 템플릿의 기본 클래스입니다.
    모든 템플릿 클래스는 이 클래스를 상속받아야 합니다.
    
    Attributes:
        _default_instruction_prompts (Dict[str, str]): 기본 instruction 프롬프트를 저장하는 딕셔너리
        _default_prompts (Dict[str, Dict[str, str]]): 기본 프롬프트를 저장하는 딕셔너리
    """
    
    def __init__(self):
        """
        BaseTemplate 클래스의 생성자입니다.
        기본 프롬프트 딕셔너리를 초기화합니다.
        """
        self._default_instruction_prompts: Dict[str, str] = {}
        self._default_prompts: Dict[str, Dict[str, str]] = {}
    
    @abstractmethod
    def _initialize_messages(self) -> None:
        """
        instruction과 프롬프트를 설정하는 추상 메서드입니다.
        자식 클래스에서 반드시 구현해야 합니다.
        """
        pass
    
    def get_prompt_for_answer(self) -> ChatPromptTemplate:
        """
        답변을 위한 프롬프트를 반환하는 메서드입니다.
        
        Returns:
            ChatPromptTemplate: 형식화된 프롬프트 템플릿
        """
        pass
    
    def get_prompt_for_score(self) -> ChatPromptTemplate:
        """
        채점을 위한 프롬프트를 반환하는 메서드입니다.
        
        Returns:
            ChatPromptTemplate: 형식화된 프롬프트 템플릿
        """
        pass
    
    def format_prompt(self, **kwargs: Any) -> str:
        """
        주어진 매개변수를 사용하여 프롬프트를 형식화합니다.
        
        Args:
            **kwargs: 프롬프트 형식화에 사용될 키워드 인자들
            
        Returns:
            str: 형식화된 프롬프트 문자열
        """
        prompt = self.get_prompt_for_answer()
        return prompt.format(**kwargs)
    