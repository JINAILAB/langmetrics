# judge_metric.py
from langmetrics.metrics.base_metric import BaseMetric
from typing import Union, Literal, Optional, List
from langmetrics.llmtestcase import LLMTestCase
from langmetrics.metrics.judge.judge_template  import JudgeTemplate
from langmetrics.llmdataset import LLMDataset
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
        score_model : Union[ChatOpenAI, ChatAnthropic, ChatClovaX], # 채점 모델 
        category : str = 'temporal_relevance',
        answer_model : Union[ChatOpenAI, ChatAnthropic, ChatClovaX] = None, # 답변 모델
        verbose_mode: bool = False,
        template_language : Literal['ko', 'en'] = 'ko',
        generate_template_type : Literal['reasoning', 'only_answer'] = 'reasoning', 
        judge_template : Union[str, JudgeTemplate, BaseTemplate] = None, 
    ):

        self.answer_model = answer_model
        self.score_model = score_model
        self.score_model_name = score_model.model_name
        self.verbose_mode = verbose_mode
        self.template_language = template_language
        self.generate_template_type = generate_template_type
        self.category = category
        
        
        
        # 템플릿 설정
        if judge_template is None:
            self.template = JudgeTemplate(self.template_language, self.generate_template_type, self.category)
        else:
            self.template = judge_template
        
        self.template_for_judge = self.template.get_prompt_for_score()

    def measure(self, testcase : Union[LLMTestCase,  List[LLMTestCase], LLMDataset]) -> JudgeResult:
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
        if isinstance(testcase, LLMDataset):
            testcases = testcase
        # 리스트 처리 
        elif isinstance(testcase, list):
            if not testcase:
                raise ValueError("Empty list provided")
            if not all(isinstance(item, LLMTestCase) for item in testcase):
                raise TypeError("All items in the list must be LLMTestCase instances")
            testcases = testcase
        # 단일 LLMTestCase 처리
        elif isinstance(testcase, LLMTestCase):
            testcases = [testcase]
        else:
            raise TypeError("Invalid input type. Expected LLMTestCase, List[LLMTestCase], or LLMDataset")

        
        # Process all cases synchronously
        results = [self._process_single_case(case) for case in testcases]

        return results[0] if isinstance(testcase, LLMTestCase) else results
    
    async def ameasure(self, testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]) -> JudgeResult:
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
        # LLMDataset 처리
        if isinstance(testcase, LLMDataset):
            testcases = testcase
        # 리스트 처리 
        elif isinstance(testcase, list):
            if not testcase:
                raise ValueError("Empty list provided")
            if not all(isinstance(item, LLMTestCase) for item in testcase):
                raise TypeError("All items in the list must be LLMTestCase instances")
            testcases = testcase
        # 단일 LLMTestCase 처리
        elif isinstance(testcase, LLMTestCase):
            testcases = [testcase]
        else:
            raise TypeError("Invalid input type. Expected LLMTestCase, List[LLMTestCase], or LLMDataset")

        # # 단일 테스트케이스를 리스트로 변환
        # testcases = [testcase] if not isinstance(testcase, list) else testcase

        results = await asyncio.gather(*[self._a_process_single_case(case) for case in testcases])

        return results[0] if isinstance(testcase, LLMTestCase) else results
    
    def _process_single_case(self, case: LLMTestCase) -> JudgeResult:
        """단일 테스트케이스를 처리합니다. 답변 모델은 단순 텍스트를 반환하고,
        score 모델이 평가를 진행합니다."""
        token_usage = {}
        try:
            self._validate_testcase(case)
            
            # 답변이 없는 경우 answer model로부터 생성
            if not case.output:
                if self.answer_model is None:
                    raise ValueError("output이 없고 answer_model도 설정되지 않았습니다. output을 직접 제공하거나 answer_model을 설정해주세요.")
                else:
                    case_output = self._generate_answer_one_case(case)
                    case.output = case_output.content
                    token_usage['answer'] = {
                        'completion_tokens': case_output.response_metadata['token_usage'].get('completion_tokens'),
                        'prompt_tokens': case_output.response_metadata['token_usage'].get('prompt_tokens'),
                        'total_tokens': case_output.response_metadata['token_usage'].get('total_tokens')
                    }
            
            # answer model의 마일드한 답변을 그대로 사용하고,
            # score 모델로 채점을 진행 (이때 trimAndLoadJson 등은 사용하지 않습니다)
            evaluation_result = self._evaluate_answer(case)
            token_usage['evaluation'] = evaluation_result.get('token_usage', {})
            score = evaluation_result.get('score', 0)
            reasoning = evaluation_result.get('reasoning', '')
            
            if self.verbose_mode:
                print(f"Input: {case.input}")
                print(f"Generated answer: {case.output}")
                print(f"Expected answer: {case.expected_output}")
                print(f"Score: {score}")
                print(f"Reasoning: {reasoning}")

            return JudgeResult(
                question=case.input,
                predicted=case.output,
                reasoning=reasoning,
                score=score,
                language=self.template_language,
                token_usage=token_usage
            )
            
        except Exception as e:
            print(f"Error processing test case: {str(e)}")
            return JudgeResult(
                question=getattr(case, 'input', ''),
                predicted=getattr(case, 'output', ''),
                reasoning=getattr(case, 'reasoning', ''),
                score=False,
                language=self.template_language,
                token_usage=token_usage
            )
        
    async def _a_process_single_case(self, case: LLMTestCase) -> JudgeResult:
        """단일 테스트케이스를 비동기 방식으로 처리합니다.
        answer model은 단순 텍스트 답변을 반환하며, 이후 score 모델로 평가합니다."""
        token_usage = {}
        try:
            self._validate_testcase(case)
            
            # 답변이 없는 경우 answer model로부터 생성
            if not case.output:
                if self.answer_model is None:
                    raise ValueError("output이 없고 answer_model도 설정되지 않았습니다. output을 직접 제공하거나 answer_model을 설정해주세요.")
                else:
                    case_output = await self._a_generate_answer_one_case(case)
                    case.output = case_output.content
                    token_usage['answer'] = {
                        'completion_tokens': case_output.response_metadata['token_usage'].get('completion_tokens'),
                        'prompt_tokens': case_output.response_metadata['token_usage'].get('prompt_tokens'),
                        'total_tokens': case_output.response_metadata['token_usage'].get('total_tokens')
                    }
            
            # score 모델을 통해 평가
            evaluation_result = await self._a_evaluate_answer(case)
            token_usage['evaluation'] = evaluation_result.get('token_usage', {})
            score = evaluation_result.get('score', 0)
            reasoning = evaluation_result.get('reasoning', '')
            
            if self.verbose_mode:
                print(f"Input: {case.input}")
                print(f"Generated answer: {case.output}")
                print(f"Expected answer: {case.expected_output}")
                print(f"Score: {score}")
                print(f"Reasoning: {reasoning}")

            return JudgeResult(
                question=case.input,
                predicted=case.output,
                reasoning=reasoning,
                score=score,
                language=self.template_language,
                token_usage=token_usage
            )
            
        except Exception as e:
            print(f"Error processing test case: {str(e)}")
            return JudgeResult(
                question=getattr(case, 'input', ''),
                predicted=getattr(case, 'output', ''),
                reasoning=getattr(case, 'reasoning', ''),
                score=False,
                language=self.template_language,
                token_usage=token_usage
            )
      
    def _generate_answer_one_case(self, case: LLMTestCase) -> AIMessage:
        """LLM을 사용하여 답변을 생성합니다."""
        try:
            response = self.answer_model.invoke(case.input)
            return response
        except Exception as e:
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")
    
    def _evaluate_answer(self, case: LLMTestCase) -> dict:
        """
        score 모델을 사용하여 답변을 평가합니다.
        
        입력(case.input)과 answer model의 평문 답변(case.output)을 포함하는 평가 프롬프트를 생성하고,
        score 모델을 호출하여 평가 결과(점수와 평가 근거)를 JSON 형태로 받아옵니다.
        
        Returns:
            dict: {
                'score': float,            # 평가 점수
                'reasoning': str,          # 평가 근거
                'token_usage': dict        # 평가에 사용된 토큰 정보
            }
        """
        evaluation_tokens = {}
        try:
            # 평가 프롬프트 생성 (질문과 answer model의 평문 답변 포함)
            

            evaluation_prompt = self.template.format_prompt(
                question=case.input,
                answer=case.output
            )
            
            # score 모델을 사용하여 평가 수행
            evaluation_response = self.score_model.invoke(evaluation_prompt)
            print(evaluation_response.content)
            
            
            # 토큰 사용량 저장
            evaluation_tokens = {
                'completion_tokens': evaluation_response.response_metadata['token_usage'].get('completion_tokens'),
                'prompt_tokens': evaluation_response.response_metadata['token_usage'].get('prompt_tokens'),
                'total_tokens': evaluation_response.response_metadata['token_usage'].get('total_tokens')
            }
            
            # score 모델의 응답을 JSON으로 파싱 (score 모델은 JSON 형태로 응답함)
            try:
                parsed_output = trimAndLoadJson(evaluation_response.content)
                
                if isinstance(parsed_output, dict):
                    result = {
                        'score': float(parsed_output.get('score', 0)),
                        'reasoning': parsed_output.get('reason', 'No reasoning provided'),
                        'token_usage': evaluation_tokens
                    }
                else:
                    # 만약 dict가 아니라면, 직접 score로 간주
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
                print(f"Warning: 평가 결과 파싱 실패. Raw output: {evaluation_response.content}")
                print(f"Error: {str(e)}")
                return {
                    'score': 0,
                    'reasoning': f'Failed to parse evaluation response: {str(e)}',
                    'token_usage': evaluation_tokens
                }
        
        except Exception as e:
            error_msg = f"답변 평가 중 오류 발생: {str(e)}"
            print(error_msg)
            return {
                'score': 0,
                'reasoning': error_msg,
                'token_usage': evaluation_tokens
            }
          
    def _a_evaluate_answer(self, case: LLMTestCase) -> dict:
        """
        score 모델을 사용하여 답변을 평가합니다.
        
        입력(case.input)과 answer model의 평문 답변(case.output)을 포함하는 평가 프롬프트를 생성하고,
        score 모델을 호출하여 평가 결과(점수와 평가 근거)를 JSON 형태로 받아옵니다.
        
        Returns:
            dict: {
                'score': float,            # 평가 점수
                'reasoning': str,          # 평가 근거
                'token_usage': dict        # 평가에 사용된 토큰 정보
            }
        """
        evaluation_tokens = {}
        try:
            # 평가 프롬프트 생성 (질문과 answer model의 평문 답변 포함)
            evaluation_prompt = self.template.format_prompt(
                question=case.input,
                answer=case.output
            )
            
            # score 모델을 사용하여 평가 수행
            evaluation_response = self.score_model.ainvoke(evaluation_prompt)
            
            # 토큰 사용량 저장
            evaluation_tokens = {
                'completion_tokens': evaluation_response.response_metadata['token_usage'].get('completion_tokens'),
                'prompt_tokens': evaluation_response.response_metadata['token_usage'].get('prompt_tokens'),
                'total_tokens': evaluation_response.response_metadata['token_usage'].get('total_tokens')
            }
            
            # score 모델의 응답을 JSON으로 파싱 (score 모델은 JSON 형태로 응답함)
            try:
                parsed_output = trimAndLoadJson(evaluation_response.content)
                
                if isinstance(parsed_output, dict):
                    result = {
                        'score': float(parsed_output.get('score', 0)),
                        'reasoning': parsed_output.get('reason', 'No reasoning provided'),
                        'token_usage': evaluation_tokens
                    }
                else:
                    # 만약 dict가 아니라면, 직접 score로 간주
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
                print(f"Warning: 평가 결과 파싱 실패. Raw output: {evaluation_response.content}")
                print(f"Error: {str(e)}")
                return {
                    'score': 0,
                    'reasoning': f'Failed to parse evaluation response: {str(e)}',
                    'token_usage': evaluation_tokens
                }
        
        except Exception as e:
            error_msg = f"답변 평가 중 오류 발생: {str(e)}"
            print(error_msg)
            return {
                'score': 0,
                'reasoning': error_msg,
                'token_usage': evaluation_tokens
            }

       
    async def _a_generate_answer_one_case(self, case: LLMTestCase) -> AIMessage:
        """LLM을 사용하여 async로 답변을 생성합니다."""
        try:
            choices_str = '\n'.join(
                f"{chr(65 + i)}: {value}" 
                for i, value in enumerate(case.choices)
            )
            prompt = self.template_for_answer.format_prompt(
                question=case.input,
                choices=choices_str
            )
            response = await self.answer_model.ainvoke(prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")
         
    def _process_output(self, parsed_output: dict, case: LLMTestCase) -> dict:
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
            print(f"경고: 생성된 답변이 비어있습니다. 입력: {case.input}")
            return {
                'answer': '',
                **(
                    {'reasoning': parsed_output.get('reasoning', '')} 
                    if self.generate_template_type == 'reasoning' 
                    else {}
                )
            }

        result = {
            'answer': answer,
        }
        
        if self.generate_template_type == 'reasoning':
            result['reasoning'] = parsed_output.get('reasoning', '')
            
        return result

    def _validate_testcase(self, case: LLMTestCase) -> None:
        """테스트케이스의 유효성을 검사합니다."""
        if not hasattr(case, 'input'):
            raise ValueError("테스트케이스는 'input' 속성을 가져야 합니다.")
        if not case.input:
            raise ValueError("input이 비어있습니다.")
    
    
    @classmethod
    def get_score_category(cls):
        json_path = Path(__file__).parent.parent.parent / 'prompt_storage' / 'medical_evaluate_prompt.json'
        data = load_json(json_path)
        return list(data['category'].keys())
    
    
    @property
    def __name__(self):
        return "JudgeMetric"