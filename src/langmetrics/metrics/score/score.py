from langmetrics.metrics.base_metric import BaseMetric
from typing import Union, Literal, List
from langmetrics.llmtestcase import LLMTestCase
from langmetrics.metrics.judge.judge_template import JudgeTemplate
from langmetrics.llmdataset import LLMDataset, ResultDataset
from langchain_core.messages import AIMessage
from langmetrics.metrics import BaseTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatClovaX
from langmetrics.utils import trimAndLoadJson, load_json
import json
import asyncio
from langmetrics.llmresult import LLMResult
from pathlib import Path


class JudgeMetric(BaseMetric):
    def __init__(
        self,
        score_model: Union[ChatOpenAI, ChatAnthropic, ChatClovaX],
        category: str = 'temporal_relevance',
        answer_model: Union[ChatOpenAI, ChatAnthropic, ChatClovaX] = None,
        verbose_mode: bool = False,
        template_language: Literal['ko', 'en'] = 'ko',
        generate_template_type: Literal['reasoning', 'only_answer'] = 'reasoning',
        judge_template: Union[str, JudgeTemplate, BaseTemplate] = None,
    ):
        self.answer_model = answer_model
        self.score_model = score_model
        self.score_model_name = score_model.model_name
        self.verbose_mode = verbose_mode
        self.template_language = template_language
        self.generate_template_type = generate_template_type
        self.category = category
        # template class
        self.template = (JudgeTemplate(self.template_language, self.generate_template_type, self.category) 
                        if judge_template is None else judge_template)
        # judge를 위한 template string
        self.template_for_judge = self.template.get_prompt_for_score()

    def measure(
        self,
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> Union[LLMResult, List[LLMResult]]:
        """
        동기 방식으로 모델 답변의 정확도를 평가합니다.
        """
        testcases = self._normalize_testcases(testcase)
        results = ResultDataset([self._process_single_case(case) for case in testcases])
        return results[0] if isinstance(testcase, LLMTestCase) else results

    async def ameasure(
        self,
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> Union[LLMResult, List[LLMResult]]:
        """
        비동기 방식으로 모델 답변의 정확도를 평가합니다.
        """
        testcases = self._normalize_testcases(testcase)
        results = await asyncio.gather(*[self._a_process_single_case(case) for case in testcases])
        results = ResultDataset(results)
        return results[0] if isinstance(testcase, LLMTestCase) else results

    def _normalize_testcases(
        self, testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> List[LLMTestCase]:
        """
        입력된 테스트케이스를 리스트 형태로 표준화합니다.
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
        
    def _process_generated_answer(self, case: LLMTestCase, response: AIMessage, evaluate_response: AIMessage) -> dict:
        """
        LLM 응답을 처리하여 JSON 파싱, 메타데이터 업데이트 및 결과 생성까지 수행합니다.
        """
        case.output = response.content
        metadata = {'teacher_template_language': self.template_language}
        metadata['student_model_name'] = response.response_metadata.get('model_name', '')
        metadata['teacher_model_name'] = evaluate_response.response_metadata.get('model_name', '')
        student_token_usage = response.response_metadata.get('token_usage', {})
        metadata['student_token_usage'] = {
            'completion_tokens': student_token_usage.get('completion_tokens'),
            'prompt_tokens': student_token_usage.get('prompt_tokens'),
            'total_tokens': student_token_usage.get('total_tokens')
        }
        teacher_token_usage = evaluate_response.response_metadata.get('token_usage', {})
        metadata['teacher_token_usage'] = {
            'completion_tokens': teacher_token_usage.get('completion_tokens'),
            'prompt_tokens': teacher_token_usage.get('prompt_tokens'),
            'total_tokens': teacher_token_usage.get('total_tokens')
        }
        
        try:
            parsed_output = trimAndLoadJson(evaluate_response.content)
            parsed_output = {
                'score': parsed_output.get('score', ''),
                'reasoning': parsed_output.get('reasoning', '')
            }
        except json.JSONDecodeError:
            if self.verbose_mode:
                print(f"Warning: JSON parsing failed. Raw output: {evaluate_response.content}")
            parsed_output = {'answer': '', 'reasoning': ''}
            metadata['error'] = f"Warning: JSON parsing failed. Raw output: {evaluate_response.content}"
        case.reasoning = parsed_output.get('reasoning', '')
        
        # verbose mode시 case 출력
        self._log_process_info(case, evaluate_response)
            
        return LLMResult(
                input=getattr(case, 'input', ''),
                student_answer=getattr(case, 'output', ''),
                teacher_answer=evaluate_response.content,
                expected_output=None,
                context=None,
                retrieval_context=None,
                score=parsed_output.get('score', ''),
                reasoning=parsed_output.get('reasoning', ''),
                choices=getattr(case, 'choices', ''),
                metadata=metadata
            )

    def _process_single_case(self, case: LLMTestCase) -> LLMResult:
        """
        동기 방식으로 단일 테스트케이스를 처리합니다.
        """
        try:
            self._validate_testcase(case)
            if not case.output:
                if self.answer_model is None:
                    raise ValueError(
                        "output이 없고 answer_model도 설정되지 않았습니다. output을 직접 제공하거나 answer_model을 설정해주세요.")
                response = self._generate_answer_one_case(case)
            else:
                response = AIMessage(
                    content=case.output,
                    response_metadata={
                        'model_name': getattr(case, 'model_name', ''),
                        'token_usage': getattr(case, 'token_usage', {})
                    }
                )
            # evaluate answer 
            evaluate_response = self._evaluate_answer(case, response)
            
            result = self._process_generated_answer(case, response, evaluate_response)
            

            return result

        except Exception as e:
            print(f"Error processing test case: {str(e)}")
            return LLMResult(
                input=getattr(case, 'input', ''),
                student_answer=getattr(case, 'output', ''),
                teacher_answer=evaluate_response.content,
                expected_output=None,
                context=None,
                retrieval_context=None,
                reasoning=getattr(case, 'reasoning', ''),
                score=0,
                metadata=getattr(case, 'metadata', {})
            )

    async def _a_process_single_case(self, case: LLMTestCase) -> LLMResult:
        """
        비동기 방식으로 단일 테스트케이스를 처리합니다.
        """
        try:
            self._validate_testcase(case)
            if not case.output:
                if self.answer_model is None:
                    raise ValueError(
                        "output이 없고 answer_model도 설정되지 않았습니다. output을 직접 제공하거나 answer_model을 설정해주세요.")
                response = self._a_generate_answer_one_case(case)
            else:
                response = AIMessage(
                    content=case.output,
                    response_metadata={
                        'model_name': getattr(case, 'model_name', ''),
                        'token_usage': getattr(case, 'token_usage', {})
                    }
                )
            # evaluate answer 
            evaluate_response = self._evaluate_answer(case, response)
            
            result = self._process_generated_answer(case, response, evaluate_response)
            

            return result

        except Exception as e:
            print(f"Error processing test case: {str(e)}")
            return LLMResult(
                question=getattr(case, 'input', ''),
                student_answer=getattr(case, 'output', ''),
                teacher_answer=evaluate_response.content,
                expected_output=None,
                context=None,
                retrieval_context=None,
                reasoning=getattr(case, 'reasoning', ''),
                score=0,
                metadata=getattr(case, 'metadata', {})
            )

    def _generate_answer_one_case(self, case: LLMTestCase) -> AIMessage:
        """
        LLM을 사용하여 동기 방식으로 답변을 생성합니다.
        """
        try:
            return self.answer_model.invoke(case.input)
        except Exception as e:
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")

    async def _a_generate_answer_one_case(self, case: LLMTestCase) -> AIMessage:
        """
        LLM을 사용하여 비동기 방식으로 답변을 생성합니다.
        """
        try:
            return await self.answer_model.ainvoke(case.input)
        except Exception as e:
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")

    def _evaluate_answer(self, case: LLMTestCase, response : AIMessage) -> dict:
        """
        동기 방식으로 score 모델을 사용하여 답변을 평가합니다.
        """
        try:
            evaluation_prompt = self.template.format_prompt(question=case.input, answer=response.content)
            evaluation_response = self.score_model.invoke(evaluation_prompt)
            return evaluation_response
        except Exception as e:
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")

    async def _a_evaluate_answer(self, case: LLMTestCase, response: AIMessage) -> dict:
        """
        비동기 방식으로 score 모델을 사용하여 답변을 평가합니다.
        """
        try:
            evaluation_prompt = self.template.format_prompt(question=case.input, answer=response.content)
            evaluation_response = self.score_model.ainvoke(evaluation_prompt)
            return evaluation_response
        except Exception as e:
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")

    def _validate_testcase(self, case: LLMTestCase) -> None:
        """
        테스트케이스의 유효성을 검사합니다.
        """
        if not hasattr(case, 'input'):
            raise ValueError("테스트케이스는 'input' 속성을 가져야 합니다.")
        if not case.input:
            raise ValueError("input이 비어있습니다.")
    
    def _log_process_info(self, case: LLMTestCase, evaluate_response: LLMResult):
        if self.verbose_mode:
                print(f"Input: {case.input}")
                print(f"Student answer: {case.output}")
                print(f"teacher answer: {evaluate_response.content}")
                print(f"Reasoning: {case.reasoning}")


    @classmethod
    def get_score_category(cls):
        json_path = Path(__file__).parent.parent.parent / 'prompt_storage' / 'medical_evaluate_prompt.json'
        data = load_json(json_path)
        return list(data['category'].keys())

    @property
    def __name__(self):
        return "JudgeMetric"
