from dataclasses import dataclass
from typing import Literal, Union, List
from langevaluate.metrics.base_metric import BaseMetric
from importlib import import_module


# 각 평가 방법별로 가능한 스코어 타입을 정의
LLM_SCORE_TYPES = Literal["accuracy", "bias", "coherence", "fluency", "relevance", "prompt_alignment"]
EXACT_MATCH_SCORE_TYPES = Literal["accuracy", "recall", 'precision', "f1_score", "semantic_similarity"]
STATISTICAL_SCORE_TYPES = Literal[
    "bleu",           # BLEU 스코어 (BiLingual Evaluation Understudy)
    "rouge_l",        # ROUGE-L (Longest Common Subsequence based)
    "rouge_1",        # ROUGE-1 (단일 단어 기반)
    "rouge_2",        # ROUGE-2 (bi-gram 기반)
    "rouge_s",
    "meteor",         # METEOR (인간 판단과 상관관계가 높은 평가 지표)
    "bert_score",     # BERTScore (문맥적 유사도)
    "ter",            # TER (Translation Edit Rate)
    "chrf",           # ChrF (character n-gram F-score)
    "perplexity",     # 언어 모델의 예측 확실성 측정
    "wer"             # WER (Word Error Rate)
]


# 평가 방법별 가능한 스코어 타입을 매핑
EVALUATION_METHOD_SCORES = {
    "llm": set(["accuracy", "bias", "coherence", "fluency", "relevance", "prompt_alignment"]),
    "exact_match": set(["accuracy", "recall", 'precision', "f1_score", "semantic_similarity"]),
    "statistical": set(["bleu", "rouge_l", "rouge_1", "rouge_2", "rouge_s", "meteor", "bert_score", "ter", "chrf", "perplexity", "wer"])
}

        
        
@dataclass
class MetricConfig:
    """평가 지표 설정을 위한 클래스
    
    이 클래스는 평가 방법(LLM, Exact Match, Statistical)에 따라
    사용할 수 있는 평가 지표를 설정하고 검증합니다.
    """
    metric_name: str  # 선택한 단일 평가 지표 (예: "bleu", "accuracy", "rouge_l")
    _metric_class: BaseMetric = None  # 동적으로 로드될 메트릭 클래스를 저장할 private 속성

    def __post_init__(self):
        """데이터클래스 초기화 후 메트릭 클래스를 동적으로 로드합니다."""
        self._load_metric_class()
        
    def _load_metric_class(self):
        """메트릭 이름에 따라 해당하는 메트릭 클래스를 동적으로 임포트합니다."""
        try:
            # 메트릭 이름을 클래스 이름 형식으로 변환 (예: accuracy -> AccuracyMetric)
            class_name = ''.join(word.capitalize() for word in self.metric_name.split('_')) + 'Metric'
            
            # 상대 경로로 메트릭 모듈 임포트
            metrics_module = import_module('.metrics', package=__package__)
            
            # 해당 클래스 가져오기
            self._metric_class = getattr(metrics_module, class_name)
            
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import metric class for {self.metric_name}: {str(e)}")
        
    def get_metric_class(self) -> BaseMetric:
        """로드된 메트릭 클래스를 반환합니다.
        
        Returns:
            Type: BaseMetric을 상속받은 메트릭 클래스
        
        Raises:
            ImportError: 메트릭 클래스 로딩에 실패한 경우
        """
        if self._metric_class is None:
            self._load_metric_class()
        return self._metric_class

    def validate(self, evaluation_method: str) -> bool:
        """선택된 평가 지표가 해당 평가 방법에서 사용 가능한지 검증합니다.
        
        Args:
            evaluation_method (str): 평가 방법 ("llm", "Exact Match", "Statistical")
            
        Returns:
            bool: 평가 지표가 유효한 경우 True, 그렇지 않은 경우 False
            
        Examples:
            >>> metric_config = metricConfig(metric_name="bleu")
            >>> metric_config.validate("Statistical")  # True
            >>> metric_config.validate("llm")  # False
            
            >>> metric_config = metricConfig(metric_name="accuracy")
            >>> metric_config.validate("llm")  # True
        """
        valid_metrics = EVALUATION_METHOD_SCORES[evaluation_method]  # 해당 평가 방법에서 사용 가능한 지표들의 집합
        return self.metric_name in valid_metrics  # 선택한 지표가 사용 가능한 지표들 중에 있는지 확인
    
    
    
    

@dataclass
class EvaluationConfig:
    """언어 모델 평가 시스템을 위한 설정 클래스

    이 클래스는 언어 모델의 평가를 위한 다양한 설정을 관리합니다.
    교사 모델과 학생 모델의 설정, 평가 방법, 데이터셋 등을 포함합니다.

    Attributes:
        dataset_name (str): 평가에 사용할 데이터셋의 이름
        task_type (Literal): 평가 태스크의 유형
            - "multiturn": 다중 턴 대화
            - "binarychoice": 이진 선택
            - "multichoice": 다중 선택
            - "open_ended": 자유 형식 응답
        metric_config (Union[MetricConfig, List[MetricConfig]]): 평가 지표 설정
        has_student_answer (bool) : student model의 answer 값을 갖고 있는지 없는지 
        teacher_model_name (str): 교사 모델의 이름 (기본값: "deepseek-v3")
        student_model_name (str): 학생 모델의 이름 (기본값: "deepseek-v3")
        teacher_temperature (float): 교사 모델의 temperature 값 (기본값: 0.0)
        student_temperature (float): 학생 모델의 temperature 값 (기본값: 0.0)
        project_name (str): 프로젝트 이름 (기본값: "multiple_choice_eval")
        batch_size (int): 배치 크기 (기본값: 10)

    Examples:
        >>> # BLEU 점수를 사용하는 번역 평가 설정
        >>> config = EvaluationConfig(
        ...     dataset_name="wmt14",
        ...     task_type="open_ended",
        ...     evaluation_method="Statistical",
        ...     metric_config=metricConfig(metric_name="bleu"),
        ...     teacher_model_name="gpt4",
        ...     student_model_name="llama2",
        ...     project_name="translation_eval"
        ... )
        
        >>> # LLM 기반의 다중 선택 문제 평가 설정
        >>> config = EvaluationConfig(
        ...     dataset_name="copa",
        ...     task_type="multichoice",
        ...     evaluation_method="llm",
        ...     metric_config=metricConfig(metric_name="accuracy"),
        ...     teacher_model_name="gpt4",
        ...     student_model_name="mistral7b"
        ... )
    """
    # 필드 정의
    project_name: str # project 이름(예 : "20250104_dataset_평가")
    dataset_name: str  # 평가에 사용할 데이터셋 이름 (예: "wmt14", "copa")
    has_student_answer : bool # 데이터셋에 student llm 정답 포함 여부
    task_type: Literal["multiturn", "binarychoice", "multichoice", "open_ended"]  # 평가 태스크 유형
    metric_config: Union[MetricConfig, List[MetricConfig]]  # 평가 지표 설정
    teacher_model_name: str = "deepseek-v3"  # 교사 모델 이름
    student_model_name: str = "deepseek-v3"  # 학생 모델 이름
    teacher_temperature: float = 0.0  # 교사 모델의 temperature (낮을수록 결정적)
    student_temperature: float = 0.0  # 학생 모델의 temperature (낮을수록 결정적)
    

    def __post_init__(self):
        """설정 값들의 유효성을 검증합니다.
        
        선택된 평가 지표가 평가 방법에 적합한지 확인하고,
        부적절한 경우 ValueError를 발생시킵니다.
        
        Raises:
            ValueError: 선택된 평가 지표가 해당 평가 방법에서 사용할 수 없는 경우
        """
        if not self.metric_config.validate(self.evaluation_method):
            raise ValueError(
                f"Invalid metric type '{self.metric_config.metric_name}' for {self.evaluation_method}. "
                f"Valid types are: {EVALUATION_METHOD_SCORES[self.evaluation_method]}"
            )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "EvaluationConfig":
        """딕셔너리로부터 EvaluationConfig 객체를 생성합니다.
        
        Args:
            config_dict (dict): 설정 정보를 담은 딕셔너리
            
        Returns:
            EvaluationConfig: 생성된 설정 객체
            
        Examples:
            >>> config_dict = {
            ...     "dataset_name": "wmt14",
            ...     "task_type": "open_ended",
            ...     "evaluation_method": "Statistical",
            ...     "metric_config": {"metric_name": "bleu"},
            ...     "teacher_model_name": "gpt4"
            ... }
            >>> config = EvaluationConfig.from_dict(config_dict)
        """
        metric_config_dict = config_dict.pop("metric_config", {})  # metric_config 정보 추출
        metric_config = MetricConfig(metric_name=metric_config_dict.get("metric_name", ""))  # MetricConfig 객체 생성
        return cls(metric_config=metric_config, **config_dict)  # EvaluationConfig 객체 생성 및 반환

    def to_dict(self) -> dict:
        """EvaluationConfig 객체를 딕셔너리로 변환합니다.
        
        Returns:
            dict: 설정 정보를 담은 딕셔너리
            
        Examples:
            >>> config = EvaluationConfig(
            ...     dataset_name="wmt14",
            ...     task_type="open_ended",
            ...     evaluation_method="Statistical",
            ...     metric_config=metricConfig(metric_name="bleu")
            ... )
            >>> config_dict = config.to_dict()
            >>> print(config_dict)
            {
                'dataset_name': 'wmt14',
                'task_type': 'open_ended',
                'evaluation_method': 'Statistical',
                'metric_config': {'metric_name': 'bleu'},
                'teacher_model_name': 'deepseek-v3',
                'student_model_name': 'deepseek-v3',
                ...
            }
        """
        # metric_config를 제외한 모든 필드를 딕셔너리로 변환
        
        result = {field.name : getattr(self, field.name) for field in self.__dataclass_fields__.values() if field.name != "metric_config"}
        
        
        # metric_config를 별도로 처리하여 추가
        result["metric_config"] = {
            "metric_name": self.metric_config.metric_name
        }
        return result