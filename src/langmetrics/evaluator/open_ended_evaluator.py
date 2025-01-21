from typing import Dict, Any
from langchain.chains import LLMChain
from ..db.base import BaseEvaluator
from ..core.result import EvaluationResult
from ..prompts import EvaluationPrompts

class OpenEndedEvaluator(BaseEvaluator):
    """Open-ended question evaluator"""
    
    def _initialize_chain(self) -> None:
        """Initialize the chains"""
        # 답변 생성을 위한 체인
        self.answer_chain = LLMChain(
            llm=self.config.llm,
            prompt=EvaluationPrompts.get_openended_answer_prompt(),
            callbacks=[self.callback] if self.callback else None
        )
        
        # 답변 평가를 위한 체인
        self.eval_chain = LLMChain(
            llm=self.config.llm,
            prompt=EvaluationPrompts.get_openended_eval_prompt(),
            callbacks=[self.callback] if self.callback else None
        )
    
    def get_prompt(self) -> str:
        return EvaluationPrompts.get_openended_answer_prompt()
    
    async def evaluate_single(self, example: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single open-ended question"""
        question = example["question"]
        reference_answer = example.get("reference_answer", "")
        context = example.get("context", "")
        
        # Get model's answer
        predicted = await self.answer_chain.arun({
            "question": question,
            "context": context
        })
        
        # Evaluate the answer
        evaluation = await self.eval_chain.arun({
            "question": question,
            "reference_answer": reference_answer,
            "predicted_answer": predicted,
            "context": context
        })
        
        # Parse the evaluation to get the score
        # 평가 결과에서 숫자 점수 추출 (0-1 사이)
        try:
            score = float(evaluation.split("Score:")[1].split("\n")[0].strip())
            is_correct = score >= 0.8  # 80% 이상을 정답으로 간주
        except:
            score = 0.0
            is_correct = False
        
        return EvaluationResult(
            question=question,
            ground_truth=reference_answer,
            predicted=predicted.strip(),
            is_correct=is_correct,
            evaluation_feedback=evaluation,
            metadata={
                "score": score,
                "context": context,
                "task_type": "open_ended"
            }
        )