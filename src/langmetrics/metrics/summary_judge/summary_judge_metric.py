from langmetrics.metrics.base_metric import BaseMetric
from typing import Union, Literal, List, Optional, Dict, Any
from langmetrics.llmtestcase import LLMTestCase
from langmetrics.metrics.summary_judge.summary_judge_template import SummaryJudgeTemplate
from langmetrics.llmdataset import LLMDataset, ResultDataset
from langchain_core.messages import AIMessage
from langmetrics.metrics import BaseTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatClovaX
from langmetrics.utils import trimAndLoadJson, load_toml
import json
import asyncio
from langmetrics.llmresult import LLMResult
from pathlib import Path


class SummaryJudgeMetric(BaseMetric):
    """    
    판별 모델을 이용한 언어 모델 응답 평가 지표

    이 클래스는 판별 역할을 하는 또 다른 언어 모델을 활용하여 언어 모델의 응답을 평가합니다.  
    응답이 제공되지 않은 경우 선택적으로 응답을 생성할 수 있으며,  
    시간적 적절성, 정확성 등 다양한 평가 기준을 바탕으로 응답을 평가할 수 있습니다.  

    주요 기능:
    1. 지정된 언어 모델을 사용하여 응답을 생성 (선택 사항)
    2. 생성된 응답을 판별 모델을 이용해 평가
    3. 판별 모델의 평가에서 점수 및 평가 근거 추출
    4. 동기 및 비동기 처리 방식 지원

    속성:
    - output_model: 응답을 생성하는 데 사용되는 모델 (선택 사항)
    - output_model_name: 응답 생성 모델의 이름
    - score_model: 응답을 평가하는 모델
    - score_model_name: 평가에 사용되는 모델의 이름
    - verbose_mode: 상세 로그 출력 여부
    - template_language: 프롬프트 템플릿 언어 ('ko' 또는 'en')
    - generate_template_type: 응답 생성 시 사용하는 템플릿 유형
    - category: 평가 기준 (예: 'temporal_relevance')
    - template: 평가를 위한 프롬프트 템플릿
    
    Examples:
        >>> from langchain_openai import ChatOpenAI
        >>> from langmetrics.llmtestcase import LLMTestCase
        >>> from langmetrics.metrics.summary_judge.summary_judge_metric import SummaryJudgeMetric
        >>> 
        >>> # 평가 모델 초기화
        >>> score_model = ChatOpenAI(model="gpt-4o")
        >>> 
        >>> # 메트릭 초기화
        >>> metric = SummaryJudgeMetric(
        ...     score_model=score_model,
        ...     category='temporal_relevance',
        ...     verbose_mode=True,
        ...     template_language='ko'
        ... )
        >>> 
        >>> # 테스트 케이스 생성
        >>> testcase = LLMTestCase(
        ...     input="대한민국의 최근 기후변화 대응 정책은 어떻게 되나요?",
        ...     output="대한민국은 2050년 탄소중립을 목표로 다양한 정책을 시행 중입니다..."
        ... )
        >>> 
        >>> # 측정 실행
        >>> result = metric.measure(testcase)
        >>> print(f"Score: {result.score}, Reasoning: {result.reasoning}")
    """

    def __init__(
        self,
        score_model: Union[ChatOpenAI, ChatAnthropic, ChatClovaX],
        category: str = 'temporal_relevance',
        output_model: Optional[Union[ChatOpenAI, ChatAnthropic, ChatClovaX]] = None,
        verbose_mode: bool = False,
        template_language: Literal['ko', 'en'] = 'ko',
        generate_template_type: Literal['reasoning', 'only_answer'] = 'reasoning',
        judge_template: Optional[Union[str, SummaryJudgeTemplate, BaseTemplate]] = None,
        max_concurrency: Optional[int] = 500,
    ):
        """
        JudgeMetric 클래스 초기화 메서드
        
        Args:
            score_model (Union[ChatOpenAI, ChatAnthropic, ChatClovaX]): 
                평가에 사용되는 언어 모델
            category (str, optional): 
                평가 기준 (기본값: 'temporal_relevance')
            output_model (Optional[Union[ChatOpenAI, ChatAnthropic, ChatClovaX]], optional): 
                응답을 생성하는 데 사용되는 언어 모델. 없으면 테스트 케이스에 이미 output이 있어야 함
            verbose_mode (bool, optional): 
                상세 로그 출력 여부. True일 경우 처리 과정과 결과를 상세히 출력함
            template_language (Literal['ko', 'en'], optional): 
                템플릿 언어. 'ko'(한국어) 또는 'en'(영어) 선택 가능
            generate_template_type (Literal['reasoning', 'only_answer'], optional): 
                템플릿 유형. 'reasoning'(추론 과정 포함) 또는 'only_answer'(답만 제공) 선택 가능
            judge_template (Optional[Union[str, SummaryJudgeTemplate, BaseTemplate]], optional): 
                사용자 정의 평가 템플릿. 없으면 기본 SummaryJudgeTemplate 사용
            max_concurrency (Optional[int], optional): 
                비동기 처리 시 최대 동시 실행 작업 수
        
        Examples:
            >>> score_model = ChatOpenAI(model="gpt-4o")
            >>> judge_metric = SummaryJudgeMetric(
            ...     score_model=score_model,
            ...     category='temporal_relevance',
            ...     template_language='ko'
            ... )
        """
        super().__init__(verbose_mode=verbose_mode, max_concurrency=max_concurrency)
        self.output_model = output_model
        self.output_model_name = output_model.model_name if output_model else ''
        self.score_model = score_model
        self.score_model_name = score_model.model_name
        self.template_language = template_language
        self.generate_template_type = generate_template_type
        self.category = category

        # 템플릿 클래스 초기화
        if judge_template is None:
            self.template = SummaryJudgeTemplate(
                self.template_language, 
                self.generate_template_type, 
                self.category
            )
        else:
            self.template = judge_template
        # judge를 위한 템플릿 문자열 가져오기
        self.template_for_judge = self.template.get_prompt_for_score()

    async def ameasure(
        self,
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset],
        max_concurrent: Optional[int] = None
    ) -> Union[LLMResult, List[LLMResult]]:
        """
        비동기 방식으로 모델 답변의 정확도를 평가합니다.
        
        주어진 테스트케이스에 대해 모델의 답변을 생성하거나 기존 답변을 평가하여 
        정확도를 측정합니다. 모든 처리는 비동기적으로 수행되어 대량의 테스트케이스를 
        효율적으로 처리할 수 있습니다.
        
        Args:
            testcase (Union[LLMTestCase, List[LLMTestCase], LLMDataset]): 
                평가할 테스트케이스. 단일 케이스, 케이스 리스트, 또는 LLMDataset 형태로 제공 가능
            max_concurrent (Optional[int], optional):
                asyncio.gather 한번에 돌아가는 개수
        
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
            ...     print(f"Score: {result.score}, Reasoning: {result.reasoning}")
            ...     
            ...     # 여러 케이스 평가
            ...     results = await metric.ameasure([testcase1, testcase2, testcase3])
            ...     avg_score = sum(r.score for r in results) / len(results)
            ...     print(f"Average score: {avg_score}")
            >>> 
            >>> # 비동기 함수 실행
            >>> asyncio.run(run_evaluation())
        """
        # 테스트 케이스를 표준화된 형식으로 변환
        testcases = self._normalize_testcases(testcase)
        # 테스트 케이스 유효성 검사
        self.validate_testcases(testcases)
        
        # 각 테스트 케이스에 대한 비동기 처리 작업 생성
        tasks = [self._a_process_single_case(case) for case in testcases]
        results = await self.gather_with_concurrency(
            max_concurrent or self.semaphore._value, 
            *tasks
        )
        # 단일 케이스인 경우 첫 번째 결과만 반환, 그렇지 않으면 전체 결과셋 반환
        return results[0] if isinstance(testcase, LLMTestCase) else ResultDataset(results)

    async def _a_process_single_case(self, case: LLMTestCase) -> LLMResult:
        """
        단일 테스트케이스를 비동기적으로 처리합니다.
        
        답변을 비동기적으로 생성하거나 기존 답변을 평가한 후 
        결과를 처리하여 반환합니다. 처리 과정에서 오류가 발생하면 
        오류 정보를 포함한 결과를 반환합니다.
        
        Args:
            case (LLMTestCase): 처리할 테스트케이스
        
        Returns:
            LLMResult: 처리된 결과 객체
        
        Notes:
            - case.output이 없고 output_model이 있으면 답변을 비동기적으로 생성합니다.
            - case.output이 이미 있으면 그대로 사용합니다.
            - 오류 발생 시 score=None과 오류 메시지를 포함한 결과를 반환합니다.
        """
        try:
            # 테스트 케이스 유효성 검사
            self._validate_testcase(case)
            
            # output이 없는 경우 output_model을 사용하여 비동기적으로 답변 생성
            if not case.output:
                if self.output_model is None:
                    raise ValueError(
                        "No output provided and no output model set. "
                        "Either provide an output or set an output model."
                    )
                response = await self._a_generate_answer_one_case(case)
            else:
                # output이 이미 있는 경우 AIMessage 객체로 변환
                response = AIMessage(
                    content=case.output,
                    response_metadata={
                        'model_name': getattr(case, 'model_name', self.output_model_name),
                        'token_usage': getattr(case, 'token_usage', {})
                    }
                )
                
            # 답변 평가
            evaluate_response = await self._a_evaluate_answer(case, response)
            
            
            # 생성된 답변 처리
            result = self._process_response_message(response, evaluate_response, case)
            
            
            return result

        except Exception as e:
            if self.verbose_mode:
                print(f"Error processing case: {e}")
            
            # 오류 발생 시 기본 결과 반환
            return LLMResult(
                input=case.input,
                output=getattr(case, 'output', ''),
                expected_output=None,
                score=None,
                metadata={'error': str(e)},
                additional_info={}
            )

    def _process_response_message(self, response: AIMessage, evaluate_response: AIMessage, case: LLMTestCase) -> LLMResult:
        """
        LLM 응답을 처리하여 결과를 생성합니다.
        
        LLM의 응답과 그 평가를 처리하여 최종 LLMResult 객체를 생성합니다.
        평가 응답에서 JSON을 추출하고, 메타데이터를 업데이트합니다.
        
        Args:
            response (AIMessage): 응답 생성 모델의 응답
            evaluate_response (AIMessage): 평가 모델의 응답
            case (LLMTestCase): 처리 중인 테스트케이스
        
        Returns:
            LLMResult: 처리된 결과 객체
        
        Notes:
            - 평가 응답은 JSON 형식({"score": 5, "reasoning": "..."})을 기대합니다.
            - JSON 파싱 실패 시 빈 값으로 대체하고 오류 메시지를 메타데이터에 기록합니다.
        """
        # 답변을 가져온 후 metadata 저장
        case.output = response.content
        
        # 메타데이터 초기화 및 채우기
        metadata = {
            'output_model_name': self.output_model_name,
            'score_model_name': self.score_model_name,
            'template_language': self.template_language,
            'category': self.category
        }
        
        # 토큰 사용량 정보 수집
        output_token_usage = self._get_token_usage(response)
        score_token_usage = self._get_token_usage(evaluate_response)
        
        if output_token_usage:
            metadata['output_token_usage'] = output_token_usage
        
        if score_token_usage:
            metadata['score_token_usage'] = score_token_usage
        
        # 평가 응답에서 JSON 추출 시도
        try:
            parsed_output = trimAndLoadJson(evaluate_response.content)
            score = parsed_output.get('score', '')
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 경고 메시지 출력(verbose_mode가 활성화된 경우)
            if self.verbose_mode:
                print(f"Warning: JSON parsing failed. Raw output: {evaluate_response.content}")
            score = None
            metadata['error'] = f"JSON parsing failed: {evaluate_response.content}"
        
        # verbose mode가 활성화된 경우 로그 출력
        self._log_process_info(case)
            
        # 추가 정보 저장
        additional_info = evaluate_response.additional_kwargs
        
        # 최종 LLMResult 객체 생성 및 반환
        return LLMResult(
            input=case.input,
            output=case.output,
            expected_output=None,
            score=score,
            metadata=metadata,
            additional_info=additional_info
        )

    async def _a_generate_answer_one_case(self, case: LLMTestCase) -> AIMessage:
        """
        LLM을 사용하여 비동기적으로 답변을 생성합니다.
        
        테스트케이스의 입력을 사용하여 output_model을 통해
        비동기적으로 답변을 생성합니다.
        
        Args:
            case (LLMTestCase): 답변을 생성할 테스트케이스
        
        Returns:
            AIMessage: 생성된 답변 메시지
        
        Raises:
            RuntimeError: 답변 생성 과정에서 오류 발생 시
        
        Examples:
            >>> response = await metric._a_generate_answer_one_case(testcase)
            >>> print(response.content)  # 생성된 답변 내용
        """

        response = await self.output_model.ainvoke(case.input)
        
        return response

    async def _a_evaluate_answer(self, case: LLMTestCase, response: AIMessage) -> AIMessage:
        """
        LLM을 사용하여 비동기적으로 답변을 평가합니다.
        
        테스트케이스의 입력과 생성된 답변을 사용하여 score_model을 통해
        비동기적으로 답변을 평가합니다.
        
        Args:
            case (LLMTestCase): 평가할 테스트케이스
            response (AIMessage): 평가할 답변
        
        Returns:
            AIMessage: 평가 결과 메시지
        
        Raises:
            RuntimeError: 평가 과정에서 오류 발생 시
        """
        try:
            # 평가를 위한 프롬프트 생성
            evaluation_prompt = self.template_for_judge.format_messages(
                clinical_history=case.input,
                summary=response.content
            )
            
            # score_model을 사용하여 비동기적으로 평가 수행
            evaluation_response = await self.score_model.ainvoke(
                evaluation_prompt, 
                parse_json=True
            )
            
            evaluation_response.additional_kwargs['input_with_prompt'] = evaluation_prompt[0].content # prompt 입력값 가져오기
            
            
            return evaluation_response
        except Exception as e:
            # 오류 발생 시 예외 발생
            raise RuntimeError(f"답변 평가 중 오류 발생: {str(e)}")

    def _validate_testcase(self, case: LLMTestCase) -> None:
        """
        테스트케이스의 유효성을 검사합니다.
        SummaryJudgeMetric은 input이 필수로 존재해야 합니다.
        
        Args:
            case (LLMTestCase): 검사할 테스트케이스
        
        Raises:
            ValueError: 필수 속성이 없거나 값이 비어있는 경우
        
        Notes:
            필수 속성:
            - input: 입력 질문 또는 지시
        """
        if not hasattr(case, 'input'):
            raise ValueError("테스트케이스는 'input' 속성을 가져야 합니다.")
        if not case.input:
            raise ValueError("input이 비어있습니다.")
        else:
            if self.verbose_mode:
                print('테스트케이스 유효성 검사를 통과했습니다.')

    @classmethod
    def get_score_category(cls):
        """
        사용 가능한 평가 카테고리 목록을 반환합니다.
        
        Returns:
            List[str]: 사용 가능한 평가 카테고리 목록
        """
        toml_path = Path(__file__).parent.parent.parent / 'prompt_storage' / 'clinical_summary_prompt.toml'
        data = load_toml(toml_path)
        return list(data['category'].keys())