from abc import ABC, abstractmethod
from langmetrics.config import ModelConfig
from langchain.chat_models.base import BaseChatModel

class BaseFactory(ABC):
    """LLM 생성을 위한 추상 팩토리 클래스
    
    각 AI 제공업체별 LLM 인스턴스 생성을 위한 인터페이스를 정의합니다.
    팩토리 메서드 패턴을 사용하여 구체적인 LLM 구현체 생성을 서브클래스에 위임합니다.
    """
    @abstractmethod
    def create_llm(self, config: ModelConfig, temperature: float) -> BaseChatModel:
        pass