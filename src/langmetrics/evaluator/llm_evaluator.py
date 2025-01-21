import asyncio
from typing import List, Dict, Type
import pandas as pd
from datasets import load_dataset


from ..config import EvaluationConfig
from ..config import ModelConfigRegistry
from ..db.base import BaseEvaluator
from ..core.result import EvaluationResult
from ..utils.utils import ResultFileHandler
from . import MCQEvaluator, MultiturnEvaluator, BCQEvaluator, OpenEndedEvaluator


class LLMEvaluator:
    """LLM 기반 평가 시스템

    이 클래스는 다양한 유형의 태스크에 대해 LLM의 성능을 평가하는 시스템을 구현합니다.
    교사 모델(teacher)과 학생 모델(student)을 설정하고, 지정된 데이터셋에 대해
    배치 단위로 평가를 수행합니다.
    
    Attributes:
        evaluation_config (EvaluationConfig): 평가 설정 정보를 담고 있는 객체
        teacher_llm: 평가 기준을 제공하는 교사 LLM 모델
        student_llm: 평가 대상이 되는 학생 LLM 모델
        callback : 평가 결과를 파일로 저장하는 핸들러
    
    Examples:
        >>> config = EvaluationConfig(
        ...     task_type="multichoice",
        ...     dataset_name="klue",
        ...     teacher_model_name="deepseek-v3",
        ...     student_model_name="deepseek-v3"
        ... )
        >>> evaluator = LLMEvaluator(config)
        >>> results_df = await evaluator.run_evaluation()
        >>> print(f"평가 점수: {results_df['is_correct'].mean():.2%}")
    """
    
    # 태스크 타입별 evaluator 매핑
    EVALUATOR_MAP: Dict[str, Type[BaseEvaluator]] = {
        "multichoice": MCQEvaluator, # 객관식 문제 평가
        "binarychoice" : BCQEvaluator, # 이지선다 문제 평가
        "open_ended": OpenEndedEvaluator, # 주관식 문제 평가
        "multiturn": MultiturnEvaluator, # 다중 턴 대화 평가
    }
    
    def __init__(self, evaluation_config: EvaluationConfig, teacher_llm : str = None, student_llm : str = None, callback : str = None):
        self.evaluation_config = evaluation_config
        self.teacher_llm = teacher_llm
        self.student_llm = student_llm
        self.callback = callback
        # evaluation method가 llm인 경우만 teacher llm 생성
        if self.evaluation_config.evaluation_method == 'llm':
            self._setup_teacher_llm()
        # student answer는 정답이 없다면 student answer 생성을 위한 student llm 지정
        if not self.evaluation_config.has_student_answer:
            self._setup_student_llm()
        self._setup_evaluator()
    

    def _setup_teacher_llm(self) -> None:
        """교사 LLM 모델을 초기화하고 설정합니다.
        
        evaluation_config에 지정된 teacher_model_name과 temperature를 사용하여
        교사 역할을 수행할 LLM 모델을 설정합니다.
        """
        if self.teacher_llm is None:
            self.teacher_llm = ModelConfigRegistry.create_llm(
                self.evaluation_config.teacher_model_name,
                temperature=self.evaluation_config.teacher_temperature
            )
            
    def _setup_student_llm(self) -> None:
        """학생 LLM 모델을 초기화하고 설정합니다.
        
        evaluation_config에 지정된 student_model_name과 temperature를 사용하여
        평가 대상이 되는 LLM 모델을 설정합니다. custom_api_base가 지정된 경우
        해당 API 엔드포인트를 사용하도록 설정합니다.
        """
        if self.student_llm is None:
            self.student_llm = ModelConfigRegistry.create_llm(
                self.evaluation_config.student_model_name,
                temperature=self.evaluation_config.student_temperature
            )
    
    
    def _setup_evaluator(self) -> None:
        """태스크 유형에 맞는 evaluator를 설정합니다.
        
        evaluation_config의 task_type에 따라 적절한 evaluator를 
        EVALUATOR_MAP에서 선택하여 초기화합니다.
        
        Raises:
            ValueError: 지원하지 않는 task_type이 지정된 경우
        """
        # task_type에 해당하는 evaluator 클래스 가져오기 (예: 'multichoice' -> MCQEvaluator)
        evaluator_cls = self.EVALUATOR_MAP.get(self.evaluation_config.task_type)
        
        # 지원하지 않는 task_type인 경우 에러 발생
        if evaluator_cls is None:
            raise ValueError(f"Unknown task type: {self.evaluation_config.task_type}, choose between 'multiturn, multichoice, binarychoice, open_ended'")
        
        
        # evaluator 인스턴스 생성 및 설정
        self.evaluator = evaluator_cls(
            config=self.evaluation_config,
            teacher_llm = self.tecaher_llm,
            student_llm = self.student_llm,
        )
    
    
    