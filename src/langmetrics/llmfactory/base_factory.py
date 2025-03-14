from abc import ABC, abstractmethod
from langmetrics.config import ModelConfig
from typing import Any
from langchain.chat_models.base import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
import math

class BaseFactory(ABC):
    """LLM 생성을 위한 추상 팩토리 클래스
    
    각 AI 제공업체별 LLM 인스턴스 생성을 위한 인터페이스를 정의합니다.
    팩토리 메서드 패턴을 사용하여 구체적인 LLM 구현체 생성을 서브클래스에 위임합니다.
    """
    def create_llm(self, config: ModelConfig, temperature: float, rpm: int = None, **kwargs) -> Any:
        """LLM 인스턴스를 생성합니다."""
        if rpm:
            rate_limiter = self._create_rate_limiter(rpm)
            kwargs['rate_limiter'] = rate_limiter
        return self._create_llm(config, temperature, **kwargs)
    
    @abstractmethod
    def _create_llm(self, config: ModelConfig, temperature: float, **kwargs) -> Any:
        """구체적인 LLM 인스턴스를 생성하는 메서드입니다. 하위 클래스에서 구현해야 합니다."""
        pass
    
    def _create_rate_limiter(self, rpm: int) -> InMemoryRateLimiter:
        """RPM 기반의 rate limiter를 생성합니다."""
        g = math.gcd(60, rpm)
        check_every_n_seconds = 60 // g
        requests_per_second = rpm // g
        max_bucket_size = rpm
        
        return InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=check_every_n_seconds,
            max_bucket_size=max_bucket_size,
        )