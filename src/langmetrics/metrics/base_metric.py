from abc import abstractmethod
from typing import Optional, Dict, Union, List, Any, TypeVar, Callable
from langmetrics.llmtestcase import LLMTestCase
from langmetrics.llmdataset import LLMDataset
from langmetrics.llmresult import LLMResult
from langmetrics.utils import trimAndLoadJson
import asyncio
from tqdm.asyncio import tqdm

T = TypeVar('T', bound=LLMTestCase)


class BaseMetric:
    def __init__(self, verbose_mode: bool = False, max_concurrency: Optional[int] = None):
        
        """
        Initialize the BaseMetric.
        
        Args:
            verbose_mode (bool): Whether to print detailed logs during processing
            max_concurrency (Optional[int]): Maximum number of concurrent operations
        """
        self.verbose_mode = verbose_mode
        self.semaphore = asyncio.Semaphore(max_concurrency if max_concurrency else 500)


    @abstractmethod
    async def ameasure(self, test_case: Union[T, List[T], LLMDataset], *args, **kwargs) -> Union[LLMResult, List[LLMResult]]:
        """
        Asynchronously measure the performance of a language model on given test cases.
        
        Args:
            test_case: Test case(s) to evaluate
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Evaluation result(s)
            
        Raises:
            NotImplementedError: When not implemented by a subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}을 작성하세요."
        )
    
    
    @abstractmethod
    def _build_prompt(self, case: LLMTestCase):
        """
        Build a prompt for the given test case.
        
        Args:
            case: Test case to build a prompt for
            
        Returns:
            A prompt in the format expected by the language model
            
        Raises:
            NotImplementedError: When not implemented by a subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}을 작성하세요."
        )
    
    @abstractmethod
    def _validate_testcase(self, case: LLMTestCase) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__}을 작성하세요."
        )
    
    
    def validate_testcases(self, cases: List[LLMTestCase]):
        for case in cases:
            self._validate_testcase(case)
    
    @staticmethod
    def _normalize_testcases(
        testcase: Union[T, List[T], LLMDataset]
    ) -> List[T]:
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
    
    
    def _log_process_info(self, case: Union[LLMTestCase, LLMResult]) -> None:
        """
        verbose_mode가 True인 경우에만 로그를 출력해주는 유틸 메서드
        """
        if self.verbose_mode:
            print(case)
            
    def _get_token_usage(self, response):
        token_usage = response.response_metadata.get('token_usage', {})
        return {
            'prompt_tokens': token_usage.get('prompt_tokens', 0),
            'completion_tokens': token_usage.get('completion_tokens', 0),
            'total_tokens': token_usage.get('total_tokens', 0)
        }
        
    def _parse_json_output(self, output, parsed_field='answer'):
        """
        주어진 JSON 형식의 문자열을 파싱하여 'answer' 값을 추출하는 함수.

        Parameters:
            output (str): JSON 형식의 문자열
            metadata (dict): 에러 발생 시 에러 메시지를 저장할 딕셔너리

        Returns:
            any: 'answer' 키에 해당하는 값 또는 None
        """
        try:
            parsed_output = trimAndLoadJson(output)
            return parsed_output.get(parsed_field, None)
        except Exception as e:
            if self.verbose_mode:
                print(f"Error parsing JSON output: {e}")
            return None

    
    async def gather_with_concurrency(self, n: Optional[int], *coros: asyncio.coroutine) -> list:
        """
        지정된 최대 동시 실행 개수(n) 내에서 여러 개의 코루틴을 동시에 실행하는 함수.

        Args:
            n (Optional[int]): 동시에 실행할 코루틴의 최대 개수. None이면 제한 없이 실행됨.
            *coros (asyncio.coroutine): 실행할 코루틴들의 리스트.

        Returns:
            list: 모든 코루틴의 실행 결과를 포함하는 리스트.
        """
        if n is None:
            return await tqdm.gather(*coros)

        semaphore = asyncio.Semaphore(n)

        async def gated_coro(sem: asyncio.Semaphore, coro: asyncio.coroutine) -> Any:
            """
            세마포어를 사용하여 동시 실행 개수를 제한하는 내부 코루틴.
            """
            async with sem:
                return await coro

        return await tqdm.gather(*(gated_coro(semaphore, c) for c in coros))

    async def _limited_task(self, coro: asyncio.coroutine):
        """
        동시 실행 개수를 제한하여 개별 코루틴을 실행하는 함수.

        Args:
            coro (asyncio.coroutine): 실행할 코루틴.

        Returns:
            Any: 실행된 코루틴의 반환값.
        """
        async with self.semaphore:
            return await coro
        
    @property
    def __name__(self) -> str:
        """
        클래스 명을 반환. 필요 시 하위 클래스에서 오버라이드 가능.

        Returns:
            str: 클래스 명
        """
        return self.__class__.__name__