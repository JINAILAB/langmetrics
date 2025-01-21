from typing import Dict, Any, List
from langchain.chains import LLMChain
from ..db.base import BaseEvaluator
from ..core.result import EvaluationResult
from ..prompts import EvaluationPrompts

class MultiturnEvaluator(BaseEvaluator):
    """Dialogue interaction evaluator"""
    
    def _initialize_chain(self) -> None:
        """Initialize the chains"""
        # 대화 응답 생성 체인
        self.response_chain = LLMChain(
            llm=self.config.llm,
            prompt=EvaluationPrompts.get_dialogue_response_prompt(),
            callbacks=[self.callback] if self.callback else None
        )
        
        # 대화 평가 체인
        self.eval_chain = LLMChain(
            llm=self.config.llm,
            prompt=EvaluationPrompts.get_dialogue_eval_prompt(),
            callbacks=[self.callback] if self.callback else None
        )
    
    def get_prompt(self) -> str:
        return EvaluationPrompts.get_dialogue_response_prompt()
    
    async def evaluate_single(self, example: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single dialogue interaction"""
        context = example.get("context", "")
        history = example.get("history", [])
        current_turn = example.get("current_turn", "")
        expected_response = example.get("expected_response", "")
        
        # Format dialogue history
        formatted_history = "\n".join([
            f"{'User' if i%2==0 else 'Assistant'}: {turn}"
            for i, turn in enumerate(history)
        ])
        
        # Get model's response
        predicted = await self.response_chain.arun({
            "context": context,
            "history": formatted_history,
            "current_turn": current_turn
        })
        
        # Evaluate the response
        evaluation = await self.eval_chain.arun({
            "context": context,
            "history": formatted_history,
            "current_turn": current_turn,
            "expected_response": expected_response,
            "predicted_response": predicted
        })
        
        # Parse evaluation results
        try:
            # 평가 결과에서 각 항목별 점수 추출
            coherence = float(evaluation.split("Coherence:")[1].split("\n")[0].strip())
            relevance = float(evaluation.split("Relevance:")[1].split("\n")[0].strip())
            quality = float(evaluation.split("Overall:")[1].split("\n")[0].strip())
            
            # 종합 점수 계산
            score = (coherence + relevance + quality) / 3
            is_correct = score >= 0.8
        except:
            score = 0.0
            is_correct = False
        
        return EvaluationResult(
            question=current_turn,  # 현재 대화 턴을 question으로 사용
            ground_truth=expected_response,
            predicted=predicted.strip(),
            is_correct=is_correct,
            evaluation_feedback=evaluation,
            metadata={
                "context": context,
                "history": history,
                "coherence": coherence,
                "relevance": relevance,
                "quality": quality,
                "score": score,
                "task_type": "dialogue"
            }
        )