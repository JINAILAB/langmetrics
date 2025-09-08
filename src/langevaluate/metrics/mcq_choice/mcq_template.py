from typing import Literal, Dict, List, Optional
from langevaluate.metrics import BaseTemplate
from pathlib import Path
import toml
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


class MCQTemplate(BaseTemplate):
    def __init__(self, language : Literal['ko', 'en'] = 'ko', 
                 template_type : Literal['reasoning', 'only_answer'] = 'reasoning',
                 prompt_for_answer : str = None,):
        self.system_message_for_answer : SystemMessagePromptTemplate
        self.human_message_for_answer : HumanMessagePromptTemplate
        self.prompt_for_answer : Optional[ChatPromptTemplate] = prompt_for_answer
        self.language = language
        self.template_type = template_type
        self._default_prompts = self._load_prompt_template()
        self._initialize_messages()
        
    def _load_prompt_template(self) -> Dict[str, Dict[str, str]]:
        """Load prompt template from toml file."""
        toml_path = Path(__file__).parent.parent.parent / 'prompt_storage' / 'mcq_choice_prompt.toml'
        try:
            with open(toml_path, 'r', encoding='utf-8') as f:
                return toml.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt template file not found at {toml_path}")
        except toml.TomlDecodeError:
            raise ValueError(f"Invalid toml format in prompt template file at {toml_path}")

    def _initialize_messages(self) -> None:
        self.system_message = SystemMessagePromptTemplate.from_template(
            self._default_prompts['system_messages'][self.language]
        )
        self.human_message = HumanMessagePromptTemplate.from_template(
            self._default_prompts[self.template_type][self.language]
        )
        self.prompt_for_answer = ChatPromptTemplate.from_messages(
            [self.system_message, self.human_message]
        )
        
    def get_prompt_for_answer(self) -> ChatPromptTemplate:
        """
        언어에 따른 적절한 프롬프트를 반환합니다.
        
        Args:
            actual_output (str): 분석할 텍스트
            
        Returns:
            str: 언어에 맞는 형식화된 프롬프트 문자열
        """
        
        return self.prompt_for_answer
    
    def format_prompt(self, question : str, choices : List[str]) -> ChatPromptTemplate:
        """

        Returns:
            str: _description_
        """
        
        formatted_choices = '\n'.join([f"{chr(65 + i)}: {value}" for i, value in enumerate(choices)])
        
        return self.prompt.format_message(question=question, choices=formatted_choices)
        