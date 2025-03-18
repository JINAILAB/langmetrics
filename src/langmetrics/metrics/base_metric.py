from abc import abstractmethod
from typing import Optional, Dict, Union, List
from langmetrics.llmtestcase import LLMTestCase
from langmetrics.llmdataset import LLMDataset


class BaseMetric:
    threshold: float = None
    score: Optional[float] = None
    score_breakdown: Dict = None
    reason: Optional[str] = None
    success: Optional[bool] = None
    evaluation_model: Optional[str] = None
    async_mode: bool = True
    verbose_mode: bool = True
    verbose_logs: Optional[str] = None

    @abstractmethod
    async def ameasure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        raise NotImplementedError(
            f"Async execution for {self.__class__.__name__} not supported yet. Please set 'async_mode' to 'False'."
        )
        
    @staticmethod
    def normalize_testcases(
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> List[LLMTestCase]:
        """
        JudgeMetric, MCQMetric 등에서 공통으로 사용하는 
        테스트케이스 정규화 함수

        Args:
            testcase (LLMTestCase | List[LLMTestCase] | LLMDataset): 
                정규화할 테스트 케이스

        Returns:
            List[LLMTestCase]: 정규화된 테스트케이스 리스트

        Raises:
            ValueError: 빈 리스트가 제공된 경우
            TypeError: 지원되지 않는 타입이 제공된 경우
        """
        if isinstance(testcase, LLMDataset):
            # LLMDataset은 이미 리스트처럼 동작하므로 그대로 반환
            return testcase
        elif isinstance(testcase, list):
            if not testcase:
                raise ValueError("Empty list provided for testcases.")
            if not all(isinstance(item, LLMTestCase) for item in testcase):
                raise TypeError("All items in the list must be LLMTestCase instances.")
            return testcase
        elif isinstance(testcase, LLMTestCase):
            # 단일 LLMTestCase를 리스트로 묶어서 반환
            return [testcase]
        else:
            # 지원되지 않는 타입
            raise TypeError(
                "Invalid input type for testcases. "
                "Expected LLMTestCase, List[LLMTestCase], or LLMDataset."
            )

    @abstractmethod
    def is_successful(self) -> bool:
        raise NotImplementedError
    
    def log_info(self, message: str) -> None:
        """
        verbose_mode가 True인 경우에만 로그를 출력해주는 유틸 메서드
        """
        if self.verbose_mode:
            print(message)
            
    def _get_token_usage(response):
        token_usage = response.response_metadata.get('token_usage', {})
        return {
            'prompt_tokens': token_usage.get('prompt_tokens', 0),
            'completion_tokens': token_usage.get('completion_tokens', 0),
            'total_tokens': token_usage.get('total_tokens', 0)
        }
    
    @property
    def __name__(self) -> str:
        """
        클래스 명을 반환. 필요 시 하위 클래스에서 오버라이드 가능.

        Returns:
            str: 클래스 명
        """
        return self.__class__.__name__