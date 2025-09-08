from langevaluate.metrics.base_metric import BaseMetric
from typing import Union, Literal, List, Optional, Dict, Any
from langevaluate.llmtestcase import LLMTestCase
from langevaluate.metrics.arena.arena_template import ArenaTemplate
from langevaluate.llmdataset import LLMDataset, ResultDataset
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatClovaX
from langevaluate.utils import trimAndLoadJson
import json
import asyncio
from langevaluate.llmresult import LLMResult
from pathlib import Path


class ArenaMetric(BaseMetric):
    """    
    두 언어 모델 응답의 품질을 비교 평가하는 지표
    
    이 클래스는 판별 역할을 하는 언어 모델을 활용하여 두 개의 언어 모델 응답을 비교 평가합니다.
    첫 번째 모드에서는 두 개의 모델 출력을 비교합니다.
    두 번째 모드에서는 주어진 참조 컨텍스트를 기준으로 출력의 품질을 평가합니다.
    
    주요 기능:
    1. 두 가지 비교 모드 지원 (모델 vs 모델, 컨텍스트 vs 출력)
    2. 판별 모델을 이용한 공정한 평가
    3. 평가 결과에서 승자와 평가 근거 추출
    4. 동기 및 비동기 처리 방식 지원
    
    속성:
    - output_model: 출력을 생성하는 모델 (model_vs_model, context_vs_model 모드에서 사용)
    - context_model: 비교 출력을 생성하는 모델 (model_vs_model 모드에서만 사용)
    - output_model_name: 출력 생성 모델의 이름
    - context_model_name: 비교 출력 생성 모델의 이름
    - score_model: 응답을 평가하는 모델
    - score_model_name: 평가에 사용되는 모델의 이름
    - verbose_mode: 상세 로그 출력 여부
    - template_language: 프롬프트 템플릿 언어 ('ko' 또는 'en')
    - arena_mode: 비교 모드 ('model_vs_model' 또는 'context_vs_model')
    - template: 평가를 위한 프롬프트 템플릿
    
    Examples:
        >>> from langchain_openai import ChatOpenAI
        >>> from langevaluate.llmtestcase import LLMTestCase
        >>> from langevaluate.metrics.arena.arena_metric import ArenaMetric
        >>> 
        >>> # 평가 모델 초기화
        >>> score_model = ChatOpenAI(model="gpt-4o")
        >>> output_model = ChatOpenAI(model="gpt-4")
        >>> context_model = ChatOpenAI(model="gpt-3.5-turbo")
        >>> 
        >>> # 메트릭 초기화 (모델 vs 모델 모드)
        >>> metric = ArenaMetric(
        ...     score_model=score_model,
        ...     output_model=output_model,
        ...     context_model=context_model,
        ...     arena_mode='model_vs_model',
        ...     verbose_mode=True,
        ...     template_language='ko'
        ... )
        >>> 
        >>> # 테스트 케이스 생성
        >>> testcase = LLMTestCase(
        ...     input="인공지능의 미래 전망에 대해 설명해주세요."
        ... )
        >>> 
        >>> # 측정 실행
        >>> result = metric.measure(testcase)
        >>> print(f"Score: {result.score}, Reasoning: {result.additional_info['reasoning']}")
    """

    def __init__(
        self,
        score_model: Union[ChatOpenAI, ChatAnthropic, ChatClovaX],
        arena_mode: Literal['model_vs_model', 'context_vs_model'] = 'model_vs_model',
        output_model: Optional[Union[ChatOpenAI, ChatAnthropic, ChatClovaX]] = None,
        context_model: Optional[Union[ChatOpenAI, ChatAnthropic, ChatClovaX]] = None,
        verbose_mode: bool = False,
        template_language: Literal['ko', 'en'] = 'ko',
        arena_template: Optional[Union[str, ArenaTemplate]] = None,
        max_concurrency: Optional[int] = 500,
    ):
        """
        ArenaMetric 클래스 초기화 메서드
        
        Args:
            score_model (Union[ChatOpenAI, ChatAnthropic, ChatClovaX]): 
                평가에 사용되는 언어 모델
            arena_mode (Literal['model_vs_model', 'context_vs_model'], optional): 
                비교 모드. 'model_vs_model'은 두 모델의 출력을 비교하고, 
                'context_vs_model'은 컨텍스트와 모델 출력을 비교합니다.
            output_model (Optional[Union[ChatOpenAI, ChatAnthropic, ChatClovaX]], optional): 
                출력을 생성하는 모델 (두 모드 모두에서 사용 가능)
            context_model (Optional[Union[ChatOpenAI, ChatAnthropic, ChatClovaX]], optional): 
                'model_vs_model' 모드에서 두 번째 출력을 생성하는 모델
            verbose_mode (bool, optional): 
                상세 로그 출력 여부. True일 경우 처리 과정과 결과를 상세히 출력함
            template_language (Literal['ko', 'en'], optional): 
                템플릿 언어. 'ko'(한국어) 또는 'en'(영어) 선택 가능
            arena_template (Optional[Union[str, ArenaTemplate]], optional): 
                사용자 정의 평가 템플릿. 없으면 기본 ArenaTemplate 사용
            max_concurrency (Optional[int], optional): 
                비동기 처리 시 최대 동시 실행 작업 수
        
        Examples:
            >>> score_model = ChatOpenAI(model="gpt-4o")
            >>> output_model = ChatOpenAI(model="gpt-4")
            >>> context_model = ChatOpenAI(model="gpt-3.5-turbo")
            >>> arena_metric = ArenaMetric(
            ...     score_model=score_model,
            ...     output_model=output_model,
            ...     context_model=context_model,
            ...     arena_mode='model_vs_model',
            ...     template_language='ko'
            ... )
        """
        super().__init__(verbose_mode=verbose_mode, max_concurrency=max_concurrency)
        self.output_model = output_model
        self.context_model = context_model
        self.output_model_name = output_model.model_name if output_model else ''
        self.context_model_name = context_model.model_name if context_model else ''
        self.score_model = score_model
        self.score_model_name = score_model.model_name
        self.template_language = template_language
        self.arena_mode = arena_mode

        # 모드에 따른 모델 검증
        if arena_mode == 'model_vs_model' and (output_model is None or context_model is None):
            raise ValueError("'model_vs_model' 모드에서는 output_model과 context_model이 모두 필요합니다.")

        # 템플릿 클래스 초기화
        if arena_template is None:
            self.template = ArenaTemplate(
                self.template_language,
                self.arena_mode
            )
        else:
            self.template = arena_template
        
        # 평가를 위한 템플릿 가져오기
        self.template_for_arena = self.template.get_prompt_for_score()

    async def ameasure(
        self,
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset],
        max_concurrent: Optional[int] = None
    ) -> Union[LLMResult, List[LLMResult]]:
        """
        비동기 방식으로 두 모델 답변을 비교 평가합니다.
        
        주어진 테스트케이스에 대해 두 모델의 답변을 비교하거나,
        참조 컨텍스트와 모델 답변을 비교하여 평가합니다.
        모든 처리는 비동기적으로 수행되어 대량의 테스트케이스를 
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
            ...     print(f"Score: {result.score}, Reasoning: {result.additional_info['reasoning']}")
            ...     
            ...     # 여러 케이스 평가
            ...     results = await metric.ameasure([testcase1, testcase2, testcase3])
            ...     avg_score = sum(float(r.score) for r in results) / len(results)
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
        
        arena_mode에 따라 두 모델의 답변을 비교하거나
        컨텍스트와 모델 답변을 비교하여 평가합니다.
        
        Args:
            case (LLMTestCase): 처리할 테스트케이스
        
        Returns:
            LLMResult: 처리된 결과 객체
        
        Notes:
            - model_vs_model 모드에서는 output_model과 context_model의 생성 답변을 비교합니다.
            - context_vs_model 모드에서는 case.context와 output_model 생성 답변(또는 case.output)을 비교합니다.
            - 오류 발생 시 score=None과 오류 메시지를 포함한 결과를 반환합니다.
        """
        try:
            # 테스트 케이스 유효성 검사
            self._validate_testcase(case)
            
            # 모드에 따라 다른 처리 방식 적용
            if self.arena_mode == 'model_vs_model':
                if not case.output:
                # 두 모델의 답변 생성
                    case.output = await self.output_model.ainvoke(case.input)
                if not case.context:
                    case.context = await self.context_model.ainvoke(case.input)
                
                # 비교 평가 수행
                evaluate_response = await self._a_evaluate_answers(
                    case, 
                )
                
                
            elif self.arena_mode == 'context_vs_model':
                # 컨텍스트가 없는 경우 오류 발생
                if not case.context:
                    raise ValueError(
                        "No context provided in testcase for context_vs_model mode."
                    )
                
                # 출력이 이미 있는지 확인
                if not case.output:
                    # output_model이 없는 경우 오류 발생
                    if self.output_model is None:
                        raise ValueError(
                            "No output_model provided and no output in testcase for context_vs_model mode."
                        )
                    # output_model을 사용하여 출력 생성
                    output_model_response = await self.output_model.ainvoke(case.input)
                    case.output = output_model_response.content
                
                # 컨텍스트와 출력 비교 평가 수행
                evaluate_response = await self._a_evaluate_answers(
                    case, 
                    case.context,
                    case.output
                )
            
            # 평가 결과 처리
            result = self._process_response_message(evaluate_response, case)
            
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

    def _process_response_message(self, evaluate_response: AIMessage, case: LLMTestCase) -> LLMResult:
        """
        LLM 평가 응답을 처리하여 결과를 생성합니다.
        
        평가 응답에서 JSON을 추출하고, 최종 결과를 LLMResult 객체로 반환합니다.
        
        Args:
            evaluate_response (AIMessage): 평가 모델의 응답
            case (LLMTestCase): 처리 중인 테스트케이스
        
        Returns:
            LLMResult: 처리된 결과 객체
        
        Notes:
            - 평가 응답은 JSON 형식({"score": -1/0/1, "reasoning": "..."})을 기대합니다.
            - JSON 파싱 실패 시 빈 값으로 대체하고 오류 메시지를 메타데이터에 기록합니다.
        """
        # 메타데이터 초기화 및 채우기
        metadata = {
            'output_model_name': self.output_model_name,
            'context_model_name': self.context_model_name,
            'score_model_name': self.score_model_name,
            'template_language': self.template_language,
            'arena_mode': self.arena_mode
        }
        
        # 토큰 사용량 정보 수집
        score_token_usage = self._get_token_usage(evaluate_response)
        
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
        if self.verbose_mode:
            print(f"입력: {case.input}")
            print(f"점수: {score}")
            
        # 추가 정보 저장
        additional_info = {
            **evaluate_response.additional_kwargs
        }
        
        # 최종 LLMResult 객체 생성 및 반환
        return LLMResult(
            input=case.input,
            output=case.output,
            expected_output=None,
            score=score,
            metadata=metadata,
            additional_info=additional_info
        )

    async def _a_evaluate_answers(self, case: LLMTestCase) -> AIMessage:
        """
        LLM을 사용하여 비동기적으로 두 답변을 평가합니다.
        
        테스트케이스의 입력과 두 답변을 사용하여 score_model을 통해
        비동기적으로 평가를 수행합니다.
        
        Args:
            case (LLMTestCase): 평가할 테스트케이스
            answer_a (str): 첫 번째 답변 (output_model의 출력 또는 컨텍스트)
            answer_b (str): 두 번째 답변 (context_model의 출력 또는 output_model의 출력)
        
        Returns:
            AIMessage: 평가 결과 메시지
        
        Raises:
            RuntimeError: 평가 과정에서 오류 발생 시
        """
        try:
            # 평가를 위한 프롬프트 생성
            evaluation_prompt = self.template_for_arena.format_messages(
                question=case.input,
                answer_1=case.output,
                answer_2=case.context
            )
            
            # score_model을 사용하여 비동기적으로 평가 수행
            evaluation_response = await self.score_model.ainvoke(
                evaluation_prompt, 
                parse_json=True
            )
            
            # 평가에 사용된 입력과 출력 저장
            evaluation_response.additional_kwargs['input_with_prompt'] = evaluation_prompt[0].content
            evaluation_response.additional_kwargs['evaluation_input'] = case.input
            evaluation_response.additional_kwargs['evaluation_output'] = evaluation_response.content
            
            return evaluation_response
        except Exception as e:
            # 오류 발생 시 예외 발생
            raise RuntimeError(f"답변 평가 중 오류 발생: {str(e)}")

    def _validate_testcase(self, case: LLMTestCase) -> None:
        """
        테스트케이스의 유효성을 검사합니다.
        
        입력 모드에 따라 필요한 속성의 존재 여부를 확인합니다.
        
        Args:
            case (LLMTestCase): 검사할 테스트케이스
        
        Raises:
            ValueError: 필수 속성이 없거나 값이 비어있는 경우
        
        Notes:
            필수 속성:
            - input: 모든 모드에서 필수
            - context: context_vs_model 모드에서만 필수
        """
        if not hasattr(case, 'input') or not case.input:
            raise ValueError("테스트케이스는 비어있지 않은 'input' 속성을 가져야 합니다.")
        
        # context_vs_model 모드에서는 context가 필수
        if self.arena_mode == 'context_vs_model':
            if not hasattr(case, 'context') or not case.context:
                raise ValueError("context_vs_model 모드에서는 'context' 속성이 필요합니다.")
        
        if self.verbose_mode:
            print('테스트케이스 유효성 검사를 통과했습니다.')

