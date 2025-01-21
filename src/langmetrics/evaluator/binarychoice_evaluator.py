from typing import Dict, Any
from langchain.chains import LLMChain
from langmetrics.core import BaseEvaluator
from ..core.result import EvaluationResult
from ..metrics import EvaluationPrompts

class BCQEvaluator(BaseEvaluator):
    """Yes or No Choice Question Evaluator"""
    
    def _initialize_chain(self) -> None:
        """Initialize the answer chain"""
        self.chain = LLMChain(
            llm=self.config.llm,
            prompt=EvaluationPrompts.get_mcq_prompt(),
            callbacks=[self.callback] if self.callback else None
        )
    
    def get_prompt(self) -> str:
        return EvaluationPrompts.get_mcq_prompt()
    
    async def evaluate_single(self, example: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single MCQ example"""
        question = example["question"]
        choices = example["choices"]
        ground_truth = example["answer"]
        
        # Format choices
        choices_str = "\n".join(
            f"{idx}. {choice}" 
            for idx, choice in enumerate(choices, start=1)
        )
        
        # Get model's answer
        predicted = await self.chain.arun({
            "question": question,
            "choices": choices_str
        })
        
        # Clean prediction (extract just the number)
        predicted = predicted.strip()
        if predicted and predicted[0].isdigit():
            predicted = predicted[0]
        
        return EvaluationResult(
            question=question,
            ground_truth=ground_truth,
            predicted=predicted,
            is_correct=predicted == ground_truth,
            evaluation_feedback="",  # MCQ doesn't need detailed feedback
            metadata={
                "choices": choices,
                "task_type": "mcq"
            }
        )
    
    async def evaluate_batch(self, example: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single MCQ example"""
        question = example["question"]
        choices = example["choices"]
        ground_truth = example["answer"]
        
        # Format choices
        choices_str = "\n".join(
            f"{idx}. {choice}" 
            for idx, choice in enumerate(choices, start=1)
        )
        
        # Get model's answer
        predicted = await self.chain.arun({
            "question": question,
            "choices": choices_str
        })
        
        # Clean prediction (extract just the number)
        predicted = predicted.strip()
        if predicted and predicted[0].isdigit():
            predicted = predicted[0]
        
        return EvaluationResult(
            question=question,
            ground_truth=ground_truth,
            predicted=predicted,
            is_correct=predicted == ground_truth,
            evaluation_feedback="",  # MCQ doesn't need detailed feedback
            metadata={
                "choices": choices,
                "task_type": "mcq"
            }
        )