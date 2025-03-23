from typing import Literal, Dict
from langmetrics.metrics import BaseTemplate
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import toml


class ArenaTemplate(BaseTemplate):
    """
    두 언어 모델 응답 비교 평가를 위한 프롬프트 템플릿 클래스
    
    모드에 따라 서로 다른 템플릿을 제공합니다:
    1. model_vs_model: 두 모델의 응답을 비교
    2. context_vs_model: 참조 컨텍스트와 모델 응답을 비교
    
    템플릿은 영어 또는 한국어로 제공됩니다.
    """
    
    def __init__(
        self, 
        language: Literal['ko', 'en'] = 'ko',
        arena_mode: Literal['model_vs_model', 'context_vs_model'] = 'model_vs_model',
        prompt: str = None,
    ):
        """
        ArenaTemplate 클래스 초기화 메서드
        
        Args:
            language (Literal['ko', 'en'], optional): 
                템플릿 언어. 'ko'(한국어) 또는 'en'(영어) 선택 가능
            arena_mode (Literal['model_vs_model', 'context_vs_model'], optional): 
                비교 모드. 'model_vs_model'은 두 모델의 출력을 비교하고, 
                'context_vs_model'은 컨텍스트와 모델 출력을 비교합니다.
            prompt (str, optional): 
                사용자 정의 프롬프트 템플릿. 제공되지 않으면 기본 템플릿 사용
        """
        self.language = language
        self.arena_mode = arena_mode
        self.human_message: HumanMessagePromptTemplate
        self.prompt: ChatPromptTemplate
        
        self._default_prompts = self._load_prompt_template()
        self._initialize_messages(prompt)
    
    def _load_prompt_template(self) -> Dict[str, Dict[str, str]]:
        """
        TOML 파일에서 프롬프트 템플릿을 로드합니다.
        
        Returns:
            Dict[str, Dict[str, str]]: 로드된 프롬프트 템플릿 사전
        
        Raises:
            FileNotFoundError: 템플릿 파일을 찾을 수 없는 경우
            ValueError: 유효하지 않은 TOML 형식인 경우
        """
        toml_path = Path(__file__).parent.parent.parent / 'prompt_storage' / 'arena_prompt.toml'
        try:
            with open(toml_path, 'r', encoding='utf-8') as f:
                return toml.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt template file not found at {toml_path}")
        except toml.TomlDecodeError:
            raise ValueError(f"Invalid toml format in prompt template file at {toml_path}")
    
    def _initialize_messages(self, prompt: str = None) -> None:
        """
        프롬프트 메시지를 초기화합니다.
        
        Args:
            prompt (str, optional): 사용자 정의 프롬프트 템플릿
        
        Raises:
            ValueError: 유효하지 않은 모드나 언어인 경우
        """
        if prompt:
            self.human_message = HumanMessagePromptTemplate.from_template(prompt)
        else:
            try:
                # 모드에 맞는 템플릿 가져오기
                mode_templates = self._default_prompts['arena_mode'][self.arena_mode]
                template = mode_templates[self.language]
                self.human_message = HumanMessagePromptTemplate.from_template(template)
            except KeyError as e:
                raise ValueError(f"Invalid arena mode or language: {e}")
        
        self.prompt = ChatPromptTemplate.from_messages([self.human_message])
    
    def get_prompt_for_score(self) -> ChatPromptTemplate:
        """
        평가에 사용될 프롬프트 템플릿을 반환합니다.
        
        Returns:
            ChatPromptTemplate: 포맷팅된 프롬프트 템플릿
        """
        return self.prompt
    
    def format_prompt(self, question: str, answer_1: str, answer_2: str) -> str:
        """
        질문과 두 답변으로 프롬프트를 포맷팅합니다.
        
        Args:
            question (str): 사용자 질문
            answer_1 (str): 첫 번째 답변 (테스트케이스의 출력 또는 컨텍스트)
            answer_2 (str): 두 번째 답변 (모델 생성 출력)
            
        Returns:
            str: 포맷팅된 프롬프트
        """
        return self.prompt.format_messages(
            question=question,
            answer_1=answer_1,
            answer_2=answer_2
        )