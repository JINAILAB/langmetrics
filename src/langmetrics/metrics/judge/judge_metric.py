# judge_metric.py
from langmetrics.metrics.base_metric import BaseMetric
from typing import Union, Literal, Optional, List
from langmetrics.llmtestcase import JudgeTestCase
from langmetrics.metrics.judge.judge_template  import JudgeTemplate
from langmetrics.llmdataset import JudgeDataset
from langchain_core.messages import AIMessage
from langmetrics.metrics import BaseTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatClovaX
from langmetrics.utils import trimAndLoadJson, load_json
import json
import asyncio
from langmetrics.metrics.base_result import JudgeResult
from pathlib import Path

class JudgeMetric(BaseMetric):
    def __init__(
        self,
        score_model : Union[ChatOpenAI, ChatAnthropic, ChatClovaX],
        answer_model : Union[ChatOpenAI, ChatAnthropic, ChatClovaX] = None,
        verbose_mode: bool = False,
        template_language : Literal['ko', 'en'] = 'ko',
        generate_template_type : Literal['reasoning', 'only_answer'] = 'reasoning', 
        template : Optional[Union[JudgeTemplate, BaseTemplate]]  = None,
    ):

        self.answer_model = answer_model
        self.score_model = score_model
        self.answer_model_name = answer_model.model_name
        self.verbose_mode = verbose_mode
        self.template_language = template_language
        self.generate_template_type = generate_template_type
        
        
        # 템플릿 설정
        if template is None:
            self.template = JudgeTemplate(self.template_language, self.generate_template_type, self.category)
        else:
            self.template = template
            
        self.template_for_judge = self.template.get_prompt_for_judge()

    def measure(self, testcase : Union[JudgeTestCase,  List[JudgeTestCase], JudgeDataset]) -> JudgeResult:
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
        if isinstance(testcase, JudgeDataset):
            testcases = testcase
        # 리스트 처리 
        elif isinstance(testcase, list):
            if not testcase:
                raise ValueError("Empty list provided")
            if not all(isinstance(item, JudgeTestCase) for item in testcase):
                raise TypeError("All items in the list must be JudgeTestCase instances")
            testcases = testcase
        # 단일 JudgeTestCase 처리
        elif isinstance(testcase, JudgeTestCase):
            testcases = [testcase]
        else:
            raise TypeError("Invalid input type. Expected JudgeTestCase, List[JudgeTestCase], or JudgeDataset")

        
        # Process all cases synchronously
        results = [self._process_single_case(case) for case in testcases]

        return results[0] if isinstance(testcase, JudgeTestCase) else results
    
    async def ameasure(self, testcase: Union[JudgeTestCase, List[JudgeTestCase], JudgeDataset]) -> JudgeResult:
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
        # JudgeDataset 처리
        if isinstance(testcase, JudgeDataset):
            testcases = testcase
        # 리스트 처리 
        elif isinstance(testcase, list):
            if not testcase:
                raise ValueError("Empty list provided")
            if not all(isinstance(item, JudgeTestCase) for item in testcase):
                raise TypeError("All items in the list must be JudgeTestCase instances")
            testcases = testcase
        # 단일 JudgeTestCase 처리
        elif isinstance(testcase, JudgeTestCase):
            testcases = [testcase]
        else:
            raise TypeError("Invalid input type. Expected JudgeTestCase, List[JudgeTestCase], or JudgeDataset")

        # # 단일 테스트케이스를 리스트로 변환
        # testcases = [testcase] if not isinstance(testcase, list) else testcase

        results = await asyncio.gather(*[self._a_process_single_case(case) for case in testcases])

        return results[0] if isinstance(testcase, JudgeTestCase) else results
    
    def _process_single_case(self, case: JudgeTestCase) -> JudgeResult:
        """단일 테스트케이스를 처리하고 결과를 반환합니다."""
        token_usage = None
        try:
            self._validate_testcase(case)
            
            # 답변이 없는 경우 새로 생성
            if not case.output and self.answer_model:
                case_output = self._generate_answer_one_case(case)
                case.output = case_output.content
                
                token_usage = {
                    'completion_tokens': case_output.response_metadata['token_usage'].get('completion_tokens'),
                    'prompt_tokens': case_output.response_metadata['token_usage'].get('prompt_tokens'),
                    'total_tokens': case_output.response_metadata['token_usage'].get('total_tokens')
                }
            
            # LLM으로 답변 평가
            score_result = self._evaluate_answer(case)
            
            if self.verbose_mode:
                print(f"Input: {case.input}")
                print(f"Generated answer: {case.output}")
                print(f"Evaluation result: {score_result}")

            return JudgeResult(
                question=case.input,
                predicted=case.output,
                score=score_result['score'],
                reasoning=score_result.get('reasoning', ''),
                language=self.template_language,
                token_usage=token_usage
            )
            
        except Exception as e:
            if self.verbose_mode:
                print(f"Error processing test case: {str(e)}")
                
            return JudgeResult(
                question=getattr(case, 'input', ''),
                predicted=getattr(case, 'output', ''),
                score=0,
                reasoning='Error occurred during evaluation',
                language=self.template_language,
                token_usage=token_usage
            )
            
        except Exception as e:
            if self.verbose_mode:
                print(f"Error processing test case: {str(e)}")
                
            # Return default JudgeResult with empty values in case of error
            return JudgeResult(
                question=getattr(case, 'input', ''),
                choice=getattr(case, 'choices', ''),
                ground_truth=getattr(case, 'expected_output', ''),
                predicted=getattr(case, 'output', ''),
                reasoning=getattr(case, 'reasoning', ''),
                score = False,
                language=self.template_language,
                token_usage=token_usage
            )
      
    def _generate_answer_one_case(self, case: JudgeTestCase) -> AIMessage:
        """LLM을 사용하여 답변을 생성합니다."""
        try:
            prompt = self.template.format_prompt(question=case.input)
            response = self.answer_model.invoke(prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")
    
    def _evaluate_answer(self, case: JudgeTestCase) -> dict:
        """LLM을 사용하여 답변을 평가합니다."""
        evaluation_tokens = None
        try:
            evaluation_prompt = self.template_for_judge.format_messages(
                question=case.input,
                answer=case.output
            )
            evaluation_response = self.score_model.invoke(evaluation_prompt)
            
            # Store evaluation token usage
            evaluation_tokens = {
                'completion_tokens': evaluation_response.response_metadata['token_usage'].get('completion_tokens'),
                'prompt_tokens': evaluation_response.response_metadata['token_usage'].get('prompt_tokens'),
                'total_tokens': evaluation_response.response_metadata['token_usage'].get('total_tokens')
            }
            
            # Parse the evaluation response
            try:
                parsed_output = trimAndLoadJson(evaluation_response.content)
                
                if isinstance(parsed_output, dict):
                    result = {
                        'score': float(parsed_output.get('score', 0)),
                        'reasoning': parsed_output.get('reasoning', 'No reasoning provided'),
                        'token_usage': evaluation_tokens
                    }
                else:
                    # Handle non-dict responses (e.g., float or int scores)
                    result = {
                        'score': float(parsed_output) if parsed_output is not None else 0,
                        'reasoning': 'Direct score provided without reasoning',
                        'token_usage': evaluation_tokens
                    }
                
                if self.verbose_mode:
                    print(f"Evaluation result: {result}")
                    print(f"Token usage: {evaluation_tokens}")
                
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                if self.verbose_mode:
                    print(f"Warning: Evaluation parsing failed. Raw output: {evaluation_response.content}")
                    print(f"Error: {str(e)}")
                
                return {
                    'score': 0,
                    'reasoning': f'Failed to parse evaluation response: {str(e)}',
                    'token_usage': evaluation_tokens
                }
                
        except Exception as e:
            error_msg = f"답변 평가 중 오류 발생: {str(e)}"
            if self.verbose_mode:
                print(error_msg)
            
            return {
                'score': 0,
                'reasoning': error_msg,
                'token_usage': evaluation_tokens
            }
      
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
                

            return JudgeResult(
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
                
            # Return default JudgeResult with empty values in case of error
            return JudgeResult(
                question=getattr(case, 'input', ''),
                choice=getattr(case, 'choices', ''),
                ground_truth=getattr(case, 'expected_output', ''),
                predicted=getattr(case, 'output', ''),
                reasoning=getattr(case, 'reasoning', ''),
                score = False,
                language=self.template_language,
                token_usage=token_usage
            )
          
       
    async def _a_generate_answer_one_case(self, case: JudgeTestCase) -> AIMessage:
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
         
    def _process_output(self, parsed_output: dict, case: JudgeTestCase) -> dict:
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

    def _validate_testcase(self, case: JudgeTestCase) -> None:
        """테스트케이스의 유효성을 검사합니다."""
        if not hasattr(case, 'input'):
            raise ValueError("테스트케이스는 'input' 속성을 가져야 합니다.")
        if not case.input:
            raise ValueError("input이 비어있습니다.")
    
    
    @classmethod
    def get_score_category():
        json_path = Path(__file__).parent.parent.parent / 'prompt_storage' / 'medical_evaluate_prompt.json'
        data = load_json(json_path)
        return list(data['category'].keys())
    
    
    @property
    def __name__(self):
        return "JudgeMetric"