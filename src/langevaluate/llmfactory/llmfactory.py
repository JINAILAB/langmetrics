from langevaluate.config import ModelConfig, NaverModelConfig, LocalModelConfig, GeminiModelConfig
from typing import Any, Optional
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models import ChatClovaX
from langchain_google_genai import ChatGoogleGenerativeAI
from .base_factory import BaseFactory
from typing import Union
from langevaluate.utils import(
    execute_shell_command,
    wait_for_server,
    terminate_process)

class LocalChatOpenAI(ChatOpenAI):
    server_process: Optional[Any] = None  # 서버 프로세스용 필드 추가
    """shutdown을 지원"""
    def __init__(self, server_process=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_process = server_process

    def shutdown(self):
        if self.server_process:
            terminate_process(self.server_process)


class GeminiFactory(BaseFactory):    
    def _create_llm(self, config: GeminiModelConfig, temperature: float, **kwargs) -> ChatGoogleGenerativeAI:
        
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model=config.model_name,
            api_key=config.api_key,
            seed=config.seed,
            max_tokens_to_sample=config.max_tokens,
            **kwargs
        )

class OpenAIFactory(BaseFactory):
    """OpenAI LLM 생성 팩토리"""
    def _create_llm(self, config: ModelConfig, temperature: float, **kwargs) -> ChatOpenAI:
        return ChatOpenAI(
            temperature=temperature,
            model=config.model_name,
            base_url=config.api_base,
            api_key=config.api_key,
            seed=config.seed,
            max_tokens=config.max_tokens,
            **kwargs
        )

class AnthropicFactory(BaseFactory):
    """Anthropic(Claude) LLM 생성 팩토리"""
    def _create_llm(self, config: ModelConfig, temperature: float, **kwargs) -> ChatAnthropic:

        return ChatAnthropic(
            temperature=temperature,
            model=config.model_name,
            api_key=config.api_key,
            max_tokens_to_sample=config.max_tokens,
            **kwargs
        )

class NaverFactory(BaseFactory):
    """Naver LLM 생성 팩토리"""
    def _create_llm(self, config: NaverModelConfig, temperature: float, **kwargs) -> ChatClovaX:
        # Naver API 클라이언트 구현
        # 실제 Naver Clova API 사용을 위한 구현 필요
        return ChatClovaX(
            temperature=temperature,
            model=config.model_name,
            apigw_api_key=config.apigw_api_key,
            api_key=config.api_key,
            seed=config.seed,
            max_tokens=config.max_tokens,
            **kwargs
        )

class DeepseekFactory(BaseFactory):
    def _create_llm(self, config: ModelConfig, temperature: float, **kwargs) -> ChatOpenAI:
        return ChatDeepSeek(
            temperature=temperature,
            model=config.model_name,
            base_url=config.api_base,
            api_key=config.api_key,
            seed=config.seed,
            max_tokens=config.max_tokens,
            **kwargs
        )

class LocalLLMFactory(BaseFactory):
    def _create_llm(self, config: LocalModelConfig, temperature: float, **kwargs) -> ChatOpenAI:
        print('waiting llm server boot')
        
        # Base command
        base_command = f"""CUDA_VISIBLE_DEVICES={config.gpus} vllm serve {config.model_name} \
--host 0.0.0.0 \
--port {config.port} \
--tensor-parallel-size {config.tp} \
--seed {config.seed} \
--gpu-memory-utilization {config.mem_fraction_static} \
--max-num-seqs {config.max_running_request}"""
        
        # Add LoRA support if configured
        if config.lora_model_path:
            # Enable LoRA
            base_command += " --enable-lora"
                
            # Add LoRA module (using the new JSON format if lora_name is provided)
            if config.lora_name:
                lora_name = config.lora_name
                lora_module = f" --lora-modules '{{\
\"name\": \"{lora_name}\", \
\"path\": \"{config.lora_model_path}\", \
\"base_model_name\": \"{config.model_name}\"\
}}'"
                base_command += lora_module
            else:
                lora_name = "lora-adapter"  # Default name if not specified
                lora_module = f" --lora-modules '{{\
\"name\": \"{lora_name}\", \
\"path\": \"{config.lora_model_path}\", \
\"base_model_name\": \"{config.model_name}\"\
}}'"
                base_command += lora_module
            
        
        # Execute the command
        server_process = execute_shell_command(base_command)
                    
        wait_for_server(f"http://localhost:{config.port}")
        
        # Determine which model to use (base model or LoRA model)
        model_name = lora_name if config.lora_model_path else config.model_name
        
        # LocalChatOpenAI는 shutdown_server를 지원함.
        return LocalChatOpenAI(
            temperature=temperature,
            model=model_name,
            base_url=f"http://localhost:{config.port}/v1",
            api_key="EMPTY",
            max_tokens=config.max_tokens,
            server_process=server_process,
            **kwargs,
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
            api_key=None,
            max_tokens=8000,
            rpm=500,
            seed=66,
            provider="openai"
        ),
        
        "gpt-4o-mini": ModelConfig(
            model_name="gpt-4o-mini",
            api_base="https://api.openai.com/v1",
            api_key=None,
            max_tokens=8000,
            rpm=500,
            seed=66,
            provider="openai"
        ),
        
        "deepseek-v3": ModelConfig(
            model_name="deepseek-chat",
            api_base="https://api.deepseek.com",
            api_key=None,
            max_tokens=4096,
            seed=66,
            provider="deepseek"
        ),
        
        "deepseek-reasoner": ModelConfig(
            model_name="deepseek-reasoner",
            api_base="https://api.deepseek.com",
            api_key=None,
            max_tokens=4096,
            seed=66,
            provider="deepseek"
        ),
        
        "claude-3.7-sonnet": ModelConfig(
            model_name="claude-3-7-sonnet-latest",
            api_base="https://api.anthropic.com",
            api_key=None,
            max_tokens=8000,
            provider="anthropic"
        ),
        
        "claude-3.5-sonnet": ModelConfig(
            model_name="claude-3-5-sonnet-latest",
            api_base="https://api.anthropic.com",
            api_key=None,
            max_tokens=8000,
            provider="anthropic"
        ),
        
        "claude-3.5-haiku": ModelConfig(
            model_name="claude-3-5-sonnet-latest",
            api_base="https://api.anthropic.com",
            api_key=None,
            max_tokens=8000,
            provider="anthropic"
        ),
    
        "naver": NaverModelConfig(
            model_name="HCX-003",
            apigw_api_key=None,
            api_key=None,
            max_tokens=4096,
            seed=66,
            provider="naver",
        ),
        
        "gemini-2.0-flash": GeminiModelConfig(
            model_name="gemini-2.0-flash",
            api_key=None,
            max_tokens=8000,
            seed=66,
            rpm=15,
            provider="gemini"
        ),
        # 다른 모델들도 여기에 추가
    }
    
    _factories = {
        "openai": OpenAIFactory(),
        "anthropic": AnthropicFactory(),
        "naver": NaverFactory(),
        "deepseek" : DeepseekFactory(),
        "gemini" : GeminiFactory(),
        "local" : LocalLLMFactory(),
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

        config = cls.DEFAULT_CONFIGS[model_name]

        if config.provider == "naver":
            if not config.api_key:
                config.api_key = os.getenv("NCP_CLOVASTUDIO_API_KEY")
            if not config.apigw_api_key:
                config.apigw_api_key = os.getenv("NCP_APIGW_API_KEY")
            if not config.api_key or not config.apigw_api_key:
                raise ValueError("Naver API 키가 제대로 설정되지 않았습니다.")
        else:
            if not config.api_key:
                provider = config.provider.upper()
                config.api_key = os.getenv(f"{provider}_API_KEY")
            if not config.api_key:
                raise ValueError(f"{model_name}의 API 키가 제대로 설정되지 않았습니다.")

        return config
    
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
    def create_llm(cls, model_name_or_config: str | ModelConfig | LocalModelConfig, temperature: float = 0.7, rpm: int = None) -> Union[ChatOpenAI, ChatAnthropic, ChatClovaX, ChatGoogleGenerativeAI, ChatDeepSeek, LocalChatOpenAI]:
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
            if model_name_or_config in cls.DEFAULT_CONFIGS:
                config = cls.get_config(model_name_or_config)
            else:
                raise ValueError(f"Model '{model_name_or_config}' not found in DEFAULT_CONFIGS. Please provide a ModelConfig instance instead.")
        elif isinstance(model_name_or_config, LocalModelConfig):
            config = model_name_or_config
            factory = cls._get_factory('local')
            return factory.create_llm(config, temperature, rpm=rpm)
        else:
            config = model_name_or_config
        
        factory = cls._get_factory(config.provider)
        return factory.create_llm(config, temperature, rpm=rpm)

