from langmetrics.config import ModelConfig, NaverModelConfig
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatClovaX
from .base_factory import BaseFactory
from typing import Union

class OpenAIFactory(BaseFactory):
    """OpenAI LLM 생성 팩토리"""
    def create_llm(self, config: ModelConfig, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            temperature=temperature,
            model=config.model_name,
            base_url=config.api_base,
            api_key=config.api_key,
            seed=config.seed,
            max_tokens=config.max_tokens
        )

class AnthropicFactory(BaseFactory):
    """Anthropic(Claude) LLM 생성 팩토리"""
    def create_llm(self, config: ModelConfig, temperature: float) -> ChatAnthropic:
        return ChatAnthropic(
            temperature=temperature,
            model=config.model_name,
            api_key=config.api_key,
            seed=config.seed,
            max_tokens_to_sample=config.max_tokens
        )

class NaverFactory(BaseFactory):
    """Naver LLM 생성 팩토리"""
    def create_llm(self, config: NaverModelConfig, temperature: float) -> ChatClovaX:
        # Naver API 클라이언트 구현
        # 실제 Naver Clova API 사용을 위한 구현 필요
        return ChatClovaX(
            temperature=temperature,
            model=config.model_name,
            apigw_api_key=config.apigw_api_key,
            api_key=config.api_key,
            seed=config.seed,
            max_tokens=config.max_tokens
        )



class LLMFactory:
    """모델별 설정을 관리하는 레지스트리
        
    다양한 LLM 모델들의 기본 설정을 중앙 집중식으로 관리하고,
    모델명 또는 제공업체명을 통해 적절한 설정과 팩토리를 반환합니다.
    
    Examples:
        >>> # 모델 설정 가져오기
        >>> config = ModelConfigRegistry.get_config("gpt-4o")
        >>> # 제공업체별 팩토리 가져오기
        >>> factory = ModelConfigRegistry._get_factory("openai")
        >>> # 기존 모델 LLM 인스턴스 생성
        >>> llm = factory.create_llm("gpt-4o", temperature=0.7)
        >>> # 새로운 model 추가하는 방법
        >>> from lang_evaluator.config import ModelConfigRegistry, ModelConfig
        >>> new_configs = ModelConfig(
        model_name="gpt-4-turbo-preview",
        api_base="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=128000
        seed=66,
        provider="openai")
        >>> llm = factory.create_llm(new_configs, temperature=0.7)
    """
    
    DEFAULT_CONFIGS = {
        "gpt-4o": ModelConfig(
            model_name="gpt-4o",
            api_base="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=8000,
            seed=66,
            provider="openai"
        ),
        
        "gpt-4o-mini": ModelConfig(
            model_name="gpt-4o-mini",
            api_base="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=8000,
            seed=66,
            provider="openai"
        ),
        
        "deepseek-v3": ModelConfig(
            model_name="deepseek-chat",
            api_base="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            max_tokens=8000,
            seed=66,
            provider="openai"
        ),
        
        "deepseek-reasoner": ModelConfig(
            model_name="deepseek-reasoner",
            api_base="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            max_tokens=8000,
            seed=66,
            provider="openai"
        ),
        
        "claude-3.5-sonnet": ModelConfig(
            model_name="claude-3-5-sonnet-latest",
            api_base="https://api.anthropic.com",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=8000,
            seed=66,
            provider="anthropic"
        ),
        
        "claude-3.5-haiku": ModelConfig(
            model_name="claude-3-5-sonnet-latest",
            api_base="https://api.anthropic.com",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=8000,
            seed=66,
            provider="anthropic"
        ),
    
        "naver": NaverModelConfig(
            model_name="HCX-003",
            apigw_api_key=os.getenv("NCP_APIGW_API_KEY"),
            api_key=os.getenv("NCP_CLOVASTUDIO_API_KEY"),
            max_tokens=4096,
            seed=66,
            provider="naver",
        )
        # 다른 모델들도 여기에 추가
    }
    
    _factories = {
        "openai": OpenAIFactory(),
        "anthropic": AnthropicFactory(),
        "naver": NaverFactory()
    }
    
    @classmethod
    def get_model_list(cls):
        """기본적으로 갖고 있는 model list 반환"""
        return list(cls.DEFAULT_CONFIGS.keys())
    
    @classmethod
    def get_config(cls, model_name: str) -> ModelConfig:
        """모델 이름에 해당하는 설정 반환
        
        Args:
            model_name (str): 모델 이름
            
        Returns:
            ModelConfig: 해당 모델의 설정
            
        Raises:
            ValueError: 지원하지 않는 모델명인 경우
        """
        if model_name not in cls.DEFAULT_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}")
        return cls.DEFAULT_CONFIGS[model_name]
    
    @classmethod
    def _get_factory(cls, provider: str) -> BaseFactory:
        """프로바이더에 해당하는 팩토리 반환
        
        Args:
            provider (str): AI 제공업체명
            
        Returns:
            LLMFactory: 해당 제공업체의 LLM 팩토리
            
        Raises:
            ValueError: 지원하지 않는 제공업체인 경우
        """
        if provider not in cls._factories:
            raise ValueError(f"Unsupported provider: {provider}")
        return cls._factories[provider]

    @classmethod
    def create_llm(cls, model_name_or_config: str | ModelConfig, temperature: float = 0.7) -> Union[ChatOpenAI, ChatAnthropic, ChatClovaX]:
        """LLM 인스턴스를 생성하는 통합 메서드
        
        등록된 모델명이나 커스텀 설정으로 LLM 인스턴스를 생성합니다.
        
        Args:
            model_name_or_config (Union[str, ModelConfig]): 
                등록된 모델명(str) 또는 커스텀 모델 설정(ModelConfig)
            temperature (float): 생성 temperature 값
            
        Returns:
            BaseChatModel: 생성된 LLM 인스턴스
            
        Examples:
            >>> # 등록된 모델 사용
            >>> llm = ModelConfigRegistry.create_llm("gpt-4o", temperature=0.7)
            >>> # 커스텀 설정 사용
            >>> llm = ModelConfigRegistry.create_llm(custom_config, temperature=0.7)
            
        Raises:
            ValueError: 지원하지 않는 모델이거나 프로바이더인 경우
        """
        if isinstance(model_name_or_config, str):
            # 문자열인 경우 DEFAULT_CONFIGS에 있는지 확인
            if model_name_or_config in cls.DEFAULT_CONFIGS:
                config = cls.get_config(model_name_or_config)
            else:
                # DEFAULT_CONFIGS에 없는 경우 ValueError 발생
                raise ValueError(f"Model '{model_name_or_config}' not found in DEFAULT_CONFIGS. Please provide a ModelConfig instance instead.")
        else:
            # ModelConfig 인스턴스인 경우 직접 사용
            config = model_name_or_config
            
        factory = cls._get_factory(config.provider)
        return factory.create_llm(config, temperature)

