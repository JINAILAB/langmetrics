from langmetrics.config import EvaluationConfig
from langmetrics.core.result import EvaluationResult
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEvaluator(ABC):
    """Base class for all evaluators"""
    
    def __init__(self, config: EvaluationConfig, teacher_llm, student_llm):
        self.config = config
        self.teacher_llm = teacher_llm
        self.student_llm = student_llm

        if not self.evaluation_config.has_student_answer:
            self._initialize_student_chain()
            
    
    @abstractmethod
    def _initialize_teacher_chain(self) -> None:
        """Initialize the teacher LLMChain"""
        pass
    
    @abstractmethod
    def _initialize_student_chain(self) -> None:
        """Initialize the student LLMChain"""
        pass
    
    @abstractmethod
    async def evaluate(self, example: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single example"""
        pass
    
    @abstractmethod
    def get_prompt(self) -> str:
        """Get the prompt template for this evaluator"""
        pass