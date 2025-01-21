from langmetrics.metrics.base_metric import BaseMetric
from typing import Union, Literal, Optional, List
from langmetrics.llmtestcase import MCQTestCase
from langmetrics.metrics.mcq_choice.mcq_template  import MCQTemplate
from langmetrics.llmdataset import MCQDataset
from langchain_core.messages import AIMessage
from langmetrics.metrics import BaseTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatClovaX
from langmetrics.utils import trimAndLoadJson
import json
import asyncio
from langmetrics.metrics.base_result import MCQResult

class JudgeMetric(BaseMetric):
    def __init__(
        self,
        answer_model : Union[ChatOpenAI, ChatAnthropic, ChatClovaX],
        verbose_mode: bool = False,
        template_language : Literal['ko', 'en'] = 'ko',
        generate_template_type : Literal['reasoning', 'only_answer'] = 'reasoning', 
        template : Optional[Union[MCQTemplate, BaseTemplate]]  = None,
    ):
        """
        객관식 문제(Multiple Choice Question) 평가를 위한 메트릭 클래스입니다.
        
        이 클래스는 LLM 모델의 객관식 문제 답변을 평가하고, 답변의 정확도를 측정합니다.
        동기와 비동기 방식 모두를 지원하며, 추론 과정 평가도 가능합니다.

        Attributes:
            answer_model: LLM 답변 생성 모델
            verbose_mode (bool): 상세 로그 출력 여부
            template_language (str): 템플릿 언어 ('ko' 또는 'en')
            generate_template_type (str): 템플릿 유형 ('reasoning' 또는 'only_answer')
            template (MCQTemplate): 답변 생성에 사용될 프롬프트 템플릿

        Examples:
            >>> # ChatGPT를 사용한 기본 설정
            >>> metric = MCQMetric(
            ...     answer_model=ChatOpenAI(model_name="gpt-3.5-turbo"),
            ...     template_language='ko',
            ...     generate_template_type='reasoning'
            ... )
            
            >>> # 단일 테스트케이스 평가
            >>> testcase = MCQTestCase(
            ...     input="대한민국의 수도는?",
            ...     choices=["서울", "부산", "대구", "인천"],
            ...     expected_output="A"
            ... )
            >>> result = metric.measure(testcase)
            
            >>> # 비동기로 여러 테스트케이스 평가
            >>> testcases = [
            ...     MCQTestCase(
            ...         input="1 + 1 = ?",
            ...         choices=["1", "2", "3", "4"],
            ...         expected_output="B"
            ...     ),
            ...     MCQTestCase(
            ...         input="2 * 2 = ?",
            ...         choices=["2", "3", "4", "5"],
            ...         expected_output="C"
            ...     )
            ... ]
            >>> results = await metric.ameasure(testcases)
        """

        self.answer_model = answer_model
        self.answer_model_name = answer_model.model_name
        self.verbose_mode = verbose_mode
        self.template_language = template_language
        self.generate_template_type = generate_template_type
        
        
        # 템플릿 설정
        if template is None:
            self.template = MCQTemplate(self.template_language, self.generate_template_type)
        else:
            self.template = template
            
        self.template_for_answer = self.template.get_prompt_for_answer()

    def measure(self, testcase : Union[MCQTestCase,  List[MCQTestCase], MCQDataset]) -> MCQResult:
        """
        모델 답변의 정확도를 평가합니다.
        
        Args:
            testcase: 평가할 단일 테스트케이스 또는 테스트케이스 리스트
                
        Returns:
            평가 결과를 담은 딕셔너리 또는 딕셔너리 리스트
            - reasoning 템플릿의 경우: {'answer': str, 'reasoning': str, 'ground_truth': str, 'score': float}
            - only_answer 템플릿의 경우: {'answer': str, 'ground_truth': str, 'score': float}
                
        Raises:
            ValueError: 테스트케이스가 올바른 형식이 아닌 경우
            TypeError: 입력값의 타입이 올바르지 않은 경우
        """
        # MCQDataset 처리
        if isinstance(testcase, MCQDataset):
            testcases = testcase
        # 리스트 처리 
        elif isinstance(testcase, list):
            if not testcase:
                raise ValueError("Empty list provided")
            if not all(isinstance(item, MCQTestCase) for item in testcase):
                raise TypeError("All items in the list must be MCQTestCase instances")
            testcases = testcase
        # 단일 MCQTestCase 처리
        elif isinstance(testcase, MCQTestCase):
            testcases = [testcase]
        else:
            raise TypeError("Invalid input type. Expected MCQTestCase, List[MCQTestCase], or MCQDataset")

        # # Convert single testcase to list
        # testcases = [testcase] if not isinstance(testcase, list) else testcase
        
        # Process all cases synchronously
        results = [self._process_single_case(case) for case in testcases]

        return results[0] if isinstance(testcase, MCQTestCase) else results
    
    async def ameasure(self, testcase: Union[MCQTestCase, List[MCQTestCase], MCQDataset]) -> MCQResult:
        """
        모델 답변의 정확도를 비동기적으로 평가합니다.
        
        Args:
            testcase: 평가할 단일 테스트케이스 또는 테스트케이스 리스트
                
        Returns:
            평가 결과를 담은 딕셔너리 또는 딕셔너리 리스트
            - reasoning 템플릿의 경우: {'answer': str, 'reasoning': str, 'ground_truth': str, 'score': float}
            - only_answer 템플릿의 경우: {'answer': str, 'ground_truth': str, 'score': float}
                
        Raises:
            ValueError: 테스트케이스가 올바른 형식이 아닌 경우
            TypeError: 입력값의 타입이 올바르지 않은 경우
        """
        # MCQDataset 처리
        if isinstance(testcase, MCQDataset):
            testcases = testcase
        # 리스트 처리 
        elif isinstance(testcase, list):
            if not testcase:
                raise ValueError("Empty list provided")
            if not all(isinstance(item, MCQTestCase) for item in testcase):
                raise TypeError("All items in the list must be MCQTestCase instances")
            testcases = testcase
        # 단일 MCQTestCase 처리
        elif isinstance(testcase, MCQTestCase):
            testcases = [testcase]
        else:
            raise TypeError("Invalid input type. Expected MCQTestCase, List[MCQTestCase], or MCQDataset")

        # # 단일 테스트케이스를 리스트로 변환
        # testcases = [testcase] if not isinstance(testcase, list) else testcase

        results = await asyncio.gather(*[self._a_process_single_case(case) for case in testcases])

        return results[0] if isinstance(testcase, MCQTestCase) else results
    
    def _process_single_case(self, case: MCQTestCase) -> MCQResult:
        """Process a single testcase and return result."""
        token_usage = None
        try:
            self._validate_testcase(case)
            print('case validate 검증 완료')
            
            # 답변이 없는 경우 새로 생성
            if not case.output:
                
                case_output = self._generate_answer_one_case(case)
                print(case_output.content)
                case.output = case_output.content
            
                # Store token usage metrics in dictionary format
                token_usage = {
                    'completion_tokens': case_output.response_metadata['token_usage'].get('completion_tokens'),
                    'prompt_tokens': case_output.response_metadata['token_usage'].get('prompt_tokens'),
                    'total_tokens': case_output.response_metadata['token_usage'].get('total_tokens')
                }
                    

                # Parse JSON response and process results
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

                # Process the output and store results
                result = self._process_output(parsed_output, case)
                case.output = result.get('answer', '')
                case.reasoning = parsed_output.get('reasoning', '')
                
                # Print detailed information in verbose mode
                if self.verbose_mode:
                    print(f"Input: {case.input}")
                    print(f"Generated answer: {case.output}")
                    print(f"Expected answer: {case.expected_output}")
                    print(f"Is correct: {case.output == case.expected_output}")
                    print(f"Reasoning: {case.reasoning}")
                

            return MCQResult(
                question=case.input,
                choice=case.choices,
                ground_truth=case.expected_output,
                predicted=case.output,
                reasoning=case.reasoning,
                score =int((case.expected_output == case.output)),
                language=self.template_language,
                token_usage=token_usage
            )
            
        except Exception as e:
            if self.verbose_mode:
                print(f"Error processing test case: {str(e)}")
                
            # Return default MCQResult with empty values in case of error
            return MCQResult(
                question=getattr(case, 'input', ''),
                choice=getattr(case, 'choices', ''),
                ground_truth=getattr(case, 'expected_output', ''),
                predicted=getattr(case, 'output', ''),
                reasoning=getattr(case, 'reasoning', ''),
                score = False,
                language=self.template_language,
                token_usage=token_usage
            )
        
    async def _a_process_single_case(self, case):
        token_usage = None
        try:
            self._validate_testcase(case)
            
            # 답변이 없는 경우 새로 생성
            if not case.output:
                
                case_output = await self._a_generate_answer_one_case(case)
                case.output = case_output.content
            
                # Store token usage metrics in dictionary format
                token_usage = {
                    'completion_tokens': case_output.response_metadata['token_usage'].get('completion_tokens'),
                    'prompt_tokens': case_output.response_metadata['token_usage'].get('prompt_tokens'),
                    'total_tokens': case_output.response_metadata['token_usage'].get('total_tokens')
                }
                    

                # Parse JSON response and process results
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

                # Process the output and store results
                result = self._process_output(parsed_output, case)
                case.output = result.get('answer', '')
                case.reasoning = parsed_output.get('reasoning', '')
                
                # Print detailed information in verbose mode
                if self.verbose_mode:
                    print(f"Input: {case.input}")
                    print(f"Generated answer: {case.output}")
                    print(f"Expected answer: {case.expected_output}")
                    print(f"Is correct: {case.output == case.expected_output}")
                    print(f"Reasoning: {case.reasoning}")
                

            return MCQResult(
                question=case.input,
                choice=case.choices,
                ground_truth=case.expected_output,
                predicted=case.output,
                reasoning=case.reasoning,
                score =int((case.expected_output == case.output)),
                language=self.template_language,
                token_usage=token_usage
            )
            
        except Exception as e:
            if self.verbose_mode:
                print(f"Error processing test case: {str(e)}")
                
            # Return default MCQResult with empty values in case of error
            return MCQResult(
                question=getattr(case, 'input', ''),
                choice=getattr(case, 'choices', ''),
                ground_truth=getattr(case, 'expected_output', ''),
                predicted=getattr(case, 'output', ''),
                reasoning=getattr(case, 'reasoning', ''),
                score = False,
                language=self.template_language,
                token_usage=token_usage
            )
          
    def _generate_answer_one_case(self, case:  MCQTestCase) -> AIMessage:
        """LLM을 사용하여 답변을 생성합니다."""
        try:
            choices_str = '\n'.join(
                f"{chr(65 + i)}: {value}" 
                for i, value in enumerate(case.choices)
            )
            prompt = self.template_for_answer.format_messages(
                question=case.input,
                choices=choices_str
            )
            response = self.answer_model.invoke(prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")
       
    async def _a_generate_answer_one_case(self, case: MCQTestCase) -> AIMessage:
        """LLM을 사용하여 async로 답변을 생성합니다."""
        try:
            choices_str = '\n'.join(
                f"{chr(65 + i)}: {value}" 
                for i, value in enumerate(case.choices)
            )
            prompt = self.template_for_answer.format_messages(
                question=case.input,
                choices=choices_str
            )
            response = await self.answer_model.ainvoke(prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")
         
    def _process_output(self, parsed_output: dict, case: MCQTestCase) -> dict:
        """
        파싱된 출력을 처리하여 결과 딕셔너리를 생성합니다.
        
        Args:
            parsed_output: 파싱된 LLM 출력
            case: 처리 중인 테스트케이스
            
        Returns:
            처리된 결과를 담은 딕셔너리
        """
        answer = parsed_output.get('answer', '').strip()
        
        if not answer:
            if self.verbose_mode:
                print(f"경고: 생성된 답변이 비어있습니다. 입력: {case.input}")
            return {
                'answer': '',
                'ground_truth': case.expected_output,
                **(
                    {'reasoning': parsed_output.get('reasoning', '')} 
                    if self.generate_template_type == 'reasoning' 
                    else {}
                )
            }

        result = {
            'answer': answer,
            'ground_truth': case.expected_output,
        }
        
        if self.generate_template_type == 'reasoning':
            result['reasoning'] = parsed_output.get('reasoning', '')
            
        return result

    def _validate_testcase(self, case: MCQTestCase) -> None:
        """테스트케이스의 유효성을 검사합니다."""
        if not hasattr(case, 'input') or not hasattr(case, 'choices') or not hasattr(case, 'expected_output'):
            raise ValueError("테스트케이스는 'input'과 'choices' 속성을 가져야 합니다.")
        if not case.choices:
            raise ValueError("choices가 비어있습니다.")
        if not case.input:
            raise ValueError("input이 비어있습니다.")
        if not case.expected_output:
            raise ValueError("expected_output이 비어있습니다.")
    @property
    def __name__(self):
        return "MCQMetric"