import json
import asyncio
from typing import Union, Literal, Optional, List
from langmetrics.metrics.base_metric import BaseMetric
from langmetrics.llmtestcase import LLMTestCase
from langmetrics.metrics.mcq_choice.mcq_template import MCQTemplate
from langmetrics.llmdataset import LLMDataset, ResultDataset
from langchain_core.messages import AIMessage
from langmetrics.metrics import BaseTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatClovaX
from langmetrics.utils import trimAndLoadJson
from langmetrics.result import LLMResult


class MCQMetric(BaseMetric):
    """
    객관식 문제 평가를 위한 메트릭 클래스
    
    이 클래스는 언어 모델(LLM)의 객관식 문제 풀이 능력을 평가하기 위한 메트릭을 제공합니다.
    문제와 선택지를 받아 모델이 생성한 답변의 정확도를 측정합니다.
    
    주요 기능:
    1. 객관식 문제 답변 생성 (answer_model이 제공된 경우)
    2. 생성된 답변의 정확도 평가
    3. 답변에 대한 추론 과정 추출
    4. 동기 및 비동기 처리 지원
    
    Attributes:
        answer_model: 답변 생성에 사용될 LLM 모델
        answer_model_name: 사용된 모델의 이름
        verbose_mode: 상세 로그 출력 여부
        template_language: 프롬프트 템플릿 언어 ('ko' 또는 'en')
        generate_template_type: 답변 생성 템플릿 유형 ('reasoning': 추론 과정 포함, 'only_answer': 답만 제공)
        template: 답변 생성에 사용될 프롬프트 템플릿
        template_for_answer: 실제 답변 생성에 사용될 포맷된 템플릿
    
    Examples:
        >>> from langchain_openai import ChatOpenAI
        >>> from langmetrics.llmtestcase import LLMTestCase
        >>> from langmetrics.metrics.mcq_choice.mcq_metric import MCQMetric
        >>> 
        >>> # 모델 초기화
        >>> model = ChatOpenAI(model="gpt-3.5-turbo")
        >>> 
        >>> # 메트릭 초기화
        >>> metric = MCQMetric(
        ...     answer_model=model,
        ...     verbose_mode=True,
        ...     template_language='ko'
        ... )
        >>> 
        >>> # 테스트 케이스 생성
        >>> testcase = LLMTestCase(
        ...     input="대한민국의 수도는?",
        ...     choices=["서울", "부산", "인천", "대구"],
        ...     expected_output="A"  # A: 서울
        ... )
        >>> 
        >>> # 측정 실행
        >>> result = metric.measure(testcase)
        >>> print(f"Score: {result.score}")  # 1 (정답인 경우)
    """

    def __init__(
        self,
        answer_model: Union[ChatOpenAI, ChatAnthropic, ChatClovaX] = None,
        verbose_mode: bool = False,
        template_language: Literal['ko', 'en'] = 'ko',
        generate_template_type: Literal['reasoning', 'only_answer'] = 'reasoning',
        template: Optional[Union[MCQTemplate, BaseTemplate]] = None,
    ):
        """
        객관식 문제 평가를 위한 메트릭 클래스의 초기화 함수
        
        Args:
            answer_model (Union[ChatOpenAI, ChatAnthropic, ChatClovaX], optional): 
                LLM 답변 생성 모델. 제공되지 않으면 테스트 케이스에 이미 output이 있어야 함
            verbose_mode (bool, optional): 
                상세 로그 출력 여부. True일 경우 처리 과정과 결과를 상세히 출력함
            template_language (Literal['ko', 'en'], optional): 
                템플릿 언어. 'ko'(한국어) 또는 'en'(영어) 선택 가능
            generate_template_type (Literal['reasoning', 'only_answer'], optional): 
                템플릿 유형. 'reasoning'(추론 과정 포함) 또는 'only_answer'(답만 제공) 선택 가능
            template (Optional[Union[MCQTemplate, BaseTemplate]], optional): 
                답변 생성에 사용될 프롬프트 템플릿. 없으면 기본 MCQTemplate 사용
        
        Examples:
            >>> metric = MCQMetric(
            ...     answer_model=ChatOpenAI(model="gpt-3.5-turbo"),
            ...     verbose_mode=True,
            ...     template_language='ko',
            ...     generate_template_type='reasoning'
            ... )
        """
        self.answer_model = answer_model
        self.answer_model_name = answer_model.model_name if answer_model else None
        self.verbose_mode = verbose_mode
        self.template_language = template_language
        self.generate_template_type = generate_template_type

        # 템플릿 설정
        if template is None:
            self.template = MCQTemplate(self.template_language, self.generate_template_type)
        else:
            self.template = template

        self.template_for_answer = self.template.get_prompt_for_answer()

    def measure(
        self, 
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> Union[LLMResult, List[LLMResult]]:
        """
        모델 답변의 정확도를 동기적으로 평가합니다.
        
        주어진 테스트케이스에 대해 모델의 답변을 생성하거나 기존 답변을 평가하여 
        정확도를 측정합니다. 모든 처리는 동기적으로 수행됩니다.
        
        Args:
            testcase (Union[LLMTestCase, List[LLMTestCase], LLMDataset]): 
                평가할 테스트케이스. 단일 케이스, 케이스 리스트, 또는 LLMDataset 형태로 제공 가능
        
        Returns:
            Union[LLMResult, List[LLMResult]]: 
                측정 결과. 입력이 단일 케이스면 LLMResult, 
                리스트나 데이터셋이면 ResultDataset(결과 리스트) 반환
        
        Raises:
            ValueError: 테스트케이스 유효성 검사 실패 시 발생
            TypeError: 지원되지 않는 입력 타입인 경우 발생
        
        Examples:
            >>> # 단일 케이스 평가
            >>> result = metric.measure(testcase)
            >>> print(f"Score: {result.score}")
            >>> 
            >>> # 여러 케이스 평가
            >>> results = metric.measure([testcase1, testcase2, testcase3])
            >>> avg_score = sum(r.score for r in results) / len(results)
            >>> print(f"Average score: {avg_score}")
        """
        testcases = self._normalize_testcases(testcase)
        results = ([self._process_single_case(case) for case in testcases])
        return results[0] if isinstance(testcase, LLMTestCase) else ResultDataset(results)

    async def ameasure(
        self, 
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset],
        batch_size: int = 256  # 한 번에 처리할 최대 테스트케이스 수
    ) -> Union[LLMResult, List[LLMResult]]:
        """
        모델 답변의 정확도를 비동기적으로 평가합니다.
        
        주어진 테스트케이스에 대해 모델의 답변을 생성하거나 기존 답변을 평가하여 
        정확도를 측정합니다. 모든 처리는 비동기적으로 수행되어 대량의 테스트케이스를 
        효율적으로 처리할 수 있습니다. langchain의 abatch 메서드를 사용하여 
        한 번에 모든 케이스를 처리합니다.
        
        Args:
            testcase (Union[LLMTestCase, List[LLMTestCase], LLMDataset]): 
                평가할 테스트케이스. 단일 케이스, 케이스 리스트, 또는 LLMDataset 형태로 제공 가능
        
        Returns:
            Union[LLMResult, List[LLMResult]]: 
                측정 결과. 입력이 단일 케이스면 LLMResult, 
                리스트나 데이터셋이면 ResultDataset(결과 리스트) 반환
        
        Raises:
            ValueError: 테스트케이스 유효성 검사 실패 시 발생
            TypeError: 지원되지 않는 입력 타입인 경우 발생
        
        Examples:
            >>> # 비동기 실행을 위한 코드
            >>> import asyncio
            >>> 
            >>> async def run_evaluation():
            ...     # 단일 케이스 평가
            ...     result = await metric.ameasure(testcase)
            ...     print(f"Score: {result.score}")
            ...     
            ...     # 여러 케이스 평가
            ...     results = await metric.ameasure([testcase1, testcase2, testcase3])
            ...     avg_score = sum(r.score for r in results) / len(results)
            ...     print(f"Average score: {avg_score}")
            >>> 
            >>> # 비동기 함수 실행
            >>> asyncio.run(run_evaluation())
        """
        # testcase 체크
        testcases = self._normalize_testcases(testcase)
        all_results = []

        # batch_size 만큼씩 끊어서 순차적으로 gather
        for i in range(0, len(testcases), batch_size):
            chunk = testcases[i:i + batch_size]
            tasks = [asyncio.create_task(self._a_process_single_case(tc)) for tc in chunk]
            results_chunk = await asyncio.gather(*tasks)
            all_results.extend(results_chunk)

        # 입력이 단일 LLMTestCase였다면 하나만 반환
        return all_results[0] if isinstance(testcase, LLMTestCase) else ResultDataset(all_results)

    def _normalize_testcases(
        self, 
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> List[LLMTestCase]:
        """
        입력된 테스트케이스를 리스트 형태로 정규화합니다.
        
        다양한 형태로 입력된 테스트케이스를 처리하기 쉬운 리스트 형태로 변환합니다.
        
        Args:
            testcase (Union[LLMTestCase, List[LLMTestCase], LLMDataset]): 
                정규화할 테스트케이스
        
        Returns:
            List[LLMTestCase]: 정규화된 테스트케이스 리스트
        
        Raises:
            ValueError: 빈 리스트가 제공된 경우
            TypeError: 지원되지 않는 입력 타입인 경우
        """
        if isinstance(testcase, LLMDataset):
            return testcase
        elif isinstance(testcase, list):
            if not testcase:
                raise ValueError("Empty list provided")
            if not all(isinstance(item, LLMTestCase) for item in testcase):
                raise TypeError("All items in the list must be LLMTestCase instances")
            return testcase
        elif isinstance(testcase, LLMTestCase):
            return [testcase]
        else:
            raise TypeError("Invalid input type. Expected LLMTestCase, List[LLMTestCase], or LLMDataset")

    def _process_generated_answer(self, response: AIMessage, case: LLMTestCase) -> LLMResult:
        """
        LLM 응답을 처리하여 결과를 생성합니다.
        
        LLM의 응답을 파싱하고, 메타데이터를 업데이트하며, 최종 결과를 생성합니다.
        응답은 JSON 형식으로 파싱되며, 정답과 추론 과정이 추출됩니다.
        
        Args:
            response (AIMessage): LLM에서 생성된 응답
            case (LLMTestCase): 처리 중인 테스트케이스
        
        Returns:
            LLMResult: 처리된 결과 객체
        
        Notes:
            - 응답은 JSON 형식({"answer": "A", "reasoning": "..."})을 기대합니다.
            - JSON 파싱 실패 시 빈 값으로 대체하고 오류 메시지를 메타데이터에 기록합니다.
        """
        case.output = response.content
        metadata = {'student_template_language': self.template_language}
        metadata['student_model_name'] = response.response_metadata.get('model_name', '')
        token_usage = response.response_metadata.get('token_usage', {})
        metadata['student_token_usage'] = {
            'completion_tokens': token_usage.get('completion_tokens'),
            'prompt_tokens': token_usage.get('prompt_tokens'),
            'total_tokens': token_usage.get('total_tokens')
        }
        try:
            parsed_output = trimAndLoadJson(case.output)
            parsed_output = {
                'answer': parsed_output.get('answer', ''),
                'reasoning': parsed_output.get('reasoning', '')
            }
        except json.JSONDecodeError:
            if self.verbose_mode:
                print(f"Warning: JSON parsing failed. Raw output: {case.output}")
            parsed_output = {'answer': '', 'reasoning': ''}
            metadata['error'] = f"Warning: JSON parsing failed. Raw output: {case.output}"

        case.reasoning = parsed_output.get('reasoning', '')
        
        # verbose mode시 case 출력
        self._log_process_info(case)
            
        return LLMResult(
                input=getattr(case, 'input', ''),
                student_answer=getattr(case, 'output', ''),
                teacher_answer=None,
                expected_output=getattr(case, 'expected_output', ''),
                context=None,
                retrieval_context=None,
                reasoning=getattr(case, 'reasoning', ''),
                choices=getattr(case, 'choices', ''),
                score=int(parsed_output['answer'] == case.expected_output),
                metadata=metadata
            )

    def _process_single_case(self, case: LLMTestCase) -> LLMResult:
        """
        단일 테스트케이스를 동기적으로 처리합니다.
        
        테스트케이스의 유효성을 검사하고, 필요한 경우 답변을 생성한 후 
        결과를 처리하여 반환합니다. 처리 과정에서 오류가 발생하면 
        오류 정보를 포함한 결과를 반환합니다.
        
        Args:
            case (LLMTestCase): 처리할 테스트케이스
        
        Returns:
            LLMResult: 처리된 결과 객체
        
        Notes:
            - case.output이 없고 answer_model이 있으면 답변을 생성합니다.
            - case.output이 이미 있으면 그대로 사용합니다.
            - 오류 발생 시 score=0과 오류 메시지를 포함한 결과를 반환합니다.
        """
        try:
            self._validate_testcase(case)
            if not case.output:
                if self.answer_model is None:
                    raise ValueError("output이 없고 answer_model도 설정되지 않았습니다. output을 직접 제공하거나 answer_model을 설정해주세요.")
                response = self._generate_answer_one_case(case)
            else:  # output이 이미 있는 경우
                # AIMessage로 변환하여 처리. 추후 수정 필요
                response = AIMessage(
                    content=case.output,
                    response_metadata={
                        'model_name': getattr(case, 'model_name', ''),
                        'token_usage': getattr(case, 'token_usage', {})
                    }
                )
            result = self._process_generated_answer(response, case)
                
            return result
        except Exception as e:
            if self.verbose_mode:
                print(f"Error processing test case: {str(e)}")
                
            return LLMResult(
                input=getattr(case, 'input', ''),
                student_answer=getattr(case, 'output', ''),
                teacher_answer=None,
                expected_output=getattr(case, 'expected_output', ''),
                context=None,
                retrieval_context=None,
                reasoning=getattr(case, 'reasoning', ''),
                choices=getattr(case, 'choices', ''),
                score=0,
                metadata={'error' : str(e)})

    async def _a_process_single_case(self, case: LLMTestCase) -> LLMResult:
        """
        단일 테스트케이스를 비동기적으로 처리합니다.
        
        테스트케이스의 유효성을 검사하고, 필요한 경우 답변을 비동기적으로 생성한 후 
        결과를 처리하여 반환합니다. 처리 과정에서 오류가 발생하면 
        오류 정보를 포함한 결과를 반환합니다.
        
        Args:
            case (LLMTestCase): 처리할 테스트케이스
        
        Returns:
            LLMResult: 처리된 결과 객체
        
        Notes:
            - case.output이 없고 answer_model이 있으면 답변을 비동기적으로 생성합니다.
            - case.output이 이미 있으면 그대로 사용합니다.
            - 오류 발생 시 score=0과 오류 메시지를 포함한 결과를 반환합니다.
        """
        try:
            self._validate_testcase(case)
            if not case.output:
                if self.answer_model is None:
                    raise ValueError("output이 없고 answer_model도 설정되지 않았습니다. output을 직접 제공하거나 answer_model을 설정해주세요.")
                response = await self._a_generate_answer_one_case(case)
            else:  # output이 이미 있는 경우
                # AIMessage로 변환하여 처리
                response = AIMessage(
                    content=case.output,
                    response_metadata={
                        'model_name': getattr(case, 'model_name', ''),
                        'token_usage': getattr(case, 'token_usage', {})
                    }
                )
            result = self._process_generated_answer(response, case)

            return result
        except Exception as e:
            if self.verbose_mode:
                print(f"Error processing test case: {str(e)}")
            return LLMResult(
                input=getattr(case, 'input', ''),
                student_answer=getattr(case, 'output', ''),
                teacher_answer=None,
                expected_output=getattr(case, 'expected_output', ''),
                context=None,
                retrieval_context=None,
                reasoning=getattr(case, 'reasoning', ''),
                choices=getattr(case, 'choices', ''),
                score=0,
                metadata={'error' : str(e)})

    def _generate_answer_one_case(self, case: LLMTestCase) -> AIMessage:
        """
        LLM을 사용하여 동기적으로 답변을 생성합니다.
        
        테스트케이스의 문제와 선택지로 프롬프트를 구성하고,
        answer_model을 사용하여 답변을 생성합니다.
        
        Args:
            case (LLMTestCase): 답변을 생성할 테스트케이스
        
        Returns:
            AIMessage: 생성된 답변 메시지
        
        Raises:
            RuntimeError: 답변 생성 과정에서 오류 발생 시
        
        Examples:
            >>> response = metric._generate_answer_one_case(testcase)
            >>> print(response.content)  # JSON 형식의 응답 내용
        """
        try:
            prompt = self._build_prompt(case)
            response = self.answer_model.invoke(prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")

    async def _a_generate_answer_one_case(self, case: LLMTestCase) -> AIMessage:
        """
        LLM을 사용하여 비동기적으로 답변을 생성합니다.
        
        테스트케이스의 문제와 선택지로 프롬프트를 구성하고,
        answer_model을 사용하여 비동기적으로 답변을 생성합니다.
        
        Args:
            case (LLMTestCase): 답변을 생성할 테스트케이스
        
        Returns:
            AIMessage: 생성된 답변 메시지
        
        Raises:
            RuntimeError: 답변 생성 과정에서 오류 발생 시
        
        Examples:
            >>> response = await metric._a_generate_answer_one_case(testcase)
            >>> print(response.content)  # JSON 형식의 응답 내용
        """
        try:
            prompt = self._build_prompt(case)
            response = await self.answer_model.ainvoke(prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")

    def _build_prompt(self, case: LLMTestCase) -> str:
        """
        테스트케이스의 내용으로 프롬프트를 생성합니다.
        문제와 선택지를 포맷팅하여 LLM에게 전달할 프롬프트를 구성합니다.
        
        Args:
            case (LLMTestCase): 프롬프트를 생성할 테스트케이스
        
        Returns:
            str: 포맷팅된 프롬프트
        
        Notes:
            - 선택지는 'A: 항목1\nB: 항목2\n...' 형식으로 포맷팅됩니다.
            - 템플릿에 question과 choices 변수를 전달합니다.
        """
        choices_str = '\n'.join(f"{chr(65 + i)}: {value}" for i, value in enumerate(case.choices))
        return self.template_for_answer.format_messages(question=case.input, choices=choices_str)

    def _validate_testcase(self, case: LLMTestCase) -> None:
        """
        테스트케이스의 유효성을 검사합니다.
        
        필수 속성과 값이 모두 존재하는지 확인합니다.
        
        Args:
            case (LLMTestCase): 검사할 테스트케이스
        
        Raises:
            ValueError: 필수 속성이 없거나 값이 비어있는 경우
        
        Notes:
            필수 속성:
            - input: 문제 내용
            - choices: 선택지 목록
            - expected_output: 기대되는 정답 (예: "A", "B", 등)
        """
        if not hasattr(case, 'input') or not hasattr(case, 'choices') or not hasattr(case, 'expected_output'):
            raise ValueError("테스트케이스는 'input'과 'choices', 'expected_output' 속성을 가져야 합니다.")
        if not case.choices:
            raise ValueError("choices가 비어있습니다.")
        if not case.input:
            raise ValueError("input이 비어있습니다.")
        if not case.expected_output:
            raise ValueError("expected_output이 비어있습니다.")
        
    def _log_process_info(self, case: LLMTestCase):
        """
        테스트케이스 처리 정보를 로그로 출력합니다.
        
        verbose_mode가 True인 경우에만 작동합니다.
        
        Args:
            case (LLMTestCase): 로그를 출력할 테스트케이스
        
        Notes:
            출력 정보:
            - 입력 문제
            - 생성된 답변
            - 기대되는 정답
            - 정답 여부
            - 추론 과정
        """
        if self.verbose_mode:
            print(f"Input: {case.input}")
            print(f"Generated answer: {case.output}")
            print(f"Expected answer: {case.expected_output}")
            print(f"Is correct: {case.output == case.expected_output}")
            print(f"Reasoning: {case.reasoning}")
            

    @property
    def __name__(self):
        """
        클래스 이름을 반환하는 프로퍼티
        
        Returns:
            str: 클래스 이름 ("MCQMetric")
        """
        return "MCQMetric"