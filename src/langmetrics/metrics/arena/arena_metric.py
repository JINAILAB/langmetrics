from langmetrics.metrics.base_metric import BaseMetric
from typing import Union, Literal, List
from langmetrics.llmtestcase import LLMTestCase
from langmetrics.metrics.arena.arena_template import ArenaTemplate
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


class ArenaMetric(BaseMetric):
    def __init__(
        self,
        judge_model: Union[ChatOpenAI, ChatAnthropic, ChatClovaX],
        category: str = 'general_comparison',
        student_model: Union[ChatOpenAI, ChatAnthropic, ChatClovaX] = None,
        teacher_model: Union[ChatOpenAI, ChatAnthropic, ChatClovaX] = None,
        verbose_mode: bool = False,
        template_language: Literal['ko', 'en'] = 'ko',
        generate_template_type: Literal['reasoning', 'only_answer'] = 'reasoning',
        arena_template: Union[str, ArenaTemplate, BaseTemplate] = None,
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.judge_model = judge_model
        self.judge_model_name = judge_model.model_name
        self.verbose_mode = verbose_mode
        self.template_language = template_language
        self.generate_template_type = generate_template_type
        self.category = category
        # template class
        self.template = (ArenaTemplate(self.template_language, self.generate_template_type, self.category) 
                        if arena_template is None else arena_template)
        # arena 비교를 위한 template string
        self.template_for_comparison = self.template.get_prompt_for_comparison()

    def measure(
        self,
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> Union[LLMResult, List[LLMResult]]:
        """
        동기 방식으로 모델 답변들을 비교 평가합니다.
        """
        testcases = self._normalize_testcases(testcase)
        results = ResultDataset([self._process_single_case(case) for case in testcases])
        return results[0] if isinstance(testcase, LLMTestCase) else results

    async def ameasure(
        self,
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> Union[LLMResult, List[LLMResult]]:
        """
        비동기 방식으로 모델 답변들을 비교 평가합니다.
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
        
    def _process_comparison_result(self, case: LLMTestCase, student_response: AIMessage, 
                                  teacher_response: AIMessage, compare_response: AIMessage) -> LLMResult:
        """
        비교 평가 결과를 처리하여 JSON 파싱, 메타데이터 업데이트 및 결과 생성까지 수행합니다.
        """
        metadata = {'template_language': self.template_language}
        metadata['student_model_name'] = student_response.response_metadata.get('model_name', '')
        metadata['teacher_model_name'] = teacher_response.response_metadata.get('model_name', '')
        metadata['judge_model_name'] = compare_response.response_metadata.get('model_name', '')
        
        student_token_usage = student_response.response_metadata.get('token_usage', {})
        metadata['student_token_usage'] = {
            'completion_tokens': student_token_usage.get('completion_tokens'),
            'prompt_tokens': student_token_usage.get('prompt_tokens'),
            'total_tokens': student_token_usage.get('total_tokens')
        }
        
        teacher_token_usage = teacher_response.response_metadata.get('token_usage', {})
        metadata['teacher_token_usage'] = {
            'completion_tokens': teacher_token_usage.get('completion_tokens'),
            'prompt_tokens': teacher_token_usage.get('prompt_tokens'),
            'total_tokens': teacher_token_usage.get('total_tokens')
        }
        
        judge_token_usage = compare_response.response_metadata.get('token_usage', {})
        metadata['judge_token_usage'] = {
            'completion_tokens': judge_token_usage.get('completion_tokens'),
            'prompt_tokens': judge_token_usage.get('prompt_tokens'),
            'total_tokens': judge_token_usage.get('total_tokens')
        }
        
        try:
            parsed_output = trimAndLoadJson(compare_response.content)
            parsed_output = {
                'reasoning': parsed_output.get('reasoning', ''),
                'winner': parsed_output.get('winner', '')
            }
        except json.JSONDecodeError:
            if self.verbose_mode:
                print(f"Warning: JSON parsing failed. Raw output: {compare_response.content}")
            parsed_output = {'reasoning': '', 'winner': ''}
            metadata['error'] = f"Warning: JSON parsing failed. Raw output: {compare_response.content}"
        
        case.reasoning = parsed_output.get('reasoning', '')
        
        # verbose mode시 case 출력
        self._log_process_info(case, student_response, teacher_response, compare_response)
            
        return LLMResult(
                input=getattr(case, 'input', ''),
                student_answer=student_response.content,
                teacher_answer=teacher_response.content,
                expected_output=getattr(case, 'expected_output', None),
                context=getattr(case, 'context', None),
                retrieval_context=getattr(case, 'retrieval_context', None),
                score=parsed_output.get('winner', ''),
                reasoning=parsed_output.get('reasoning', ''),
                choices=getattr(case, 'choices', None),
                metadata=metadata
            )

    def _process_single_case(self, case: LLMTestCase) -> LLMResult:
        """
        동기 방식으로 단일 테스트케이스를 처리합니다.
        """
        try:
            self._validate_testcase(case)
            
            # 학생 모델 답변 처리
            if not hasattr(case, 'output') or not case.output:
                if self.student_model is None:
                    raise ValueError(
                        "output이 없고 student_model도 설정되지 않았습니다. output을 직접 제공하거나 student_model을 설정해주세요.")
                student_response = self._generate_student_answer(case)
            else:
                student_response = AIMessage(
                    content=case.output,
                    response_metadata={
                        'model_name': getattr(case, 'model_name', ''),
                        'token_usage': getattr(case, 'token_usage', {})
                    }
                )
            
            # 교사 모델 답변 처리
            if not hasattr(case, 'expected_output') or not case.expected_output:
                if self.teacher_model is None:
                    raise ValueError(
                        "expected_output이 없고 teacher_model도 설정되지 않았습니다. expected_output을 직접 제공하거나 teacher_model을 설정해주세요.")
                teacher_response = self._generate_teacher_answer(case)
            else:
                teacher_response = AIMessage(
                    content=case.expected_output,
                    response_metadata={
                        'model_name': getattr(case, 'reference_model_name', ''),
                        'token_usage': getattr(case, 'reference_token_usage', {})
                    }
                )
            
            # 두 답변 비교 평가
            compare_response = self._compare_answers(case, student_response, teacher_response)
            
            result = self._process_comparison_result(case, student_response, teacher_response, compare_response)
            
            return result

        except Exception as e:
            print(f"Error processing test case: {str(e)}")
            return LLMResult(
                input=getattr(case, 'input', ''),
                student_answer=getattr(case, 'output', ''),
                teacher_answer=getattr(case, 'expected_output', ''),
                expected_output=getattr(case, 'expected_output', ''),
                context=getattr(case, 'context', None),
                retrieval_context=getattr(case, 'retrieval_context', None),
                reasoning="처리 중 오류 발생: " + str(e),
                score=0,
                metadata=getattr(case, 'metadata', {})
            )

    async def _a_process_single_case(self, case: LLMTestCase) -> LLMResult:
        """
        비동기 방식으로 단일 테스트케이스를 처리합니다.
        """
        try:
            self._validate_testcase(case)
            
            # 학생 모델 답변 처리
            if not hasattr(case, 'output') or not case.output:
                if self.student_model is None:
                    raise ValueError(
                        "output이 없고 student_model도 설정되지 않았습니다. output을 직접 제공하거나 student_model을 설정해주세요.")
                student_response = await self._a_generate_student_answer(case)
            else:
                student_response = AIMessage(
                    content=case.output,
                    response_metadata={
                        'model_name': getattr(case, 'model_name', ''),
                        'token_usage': getattr(case, 'token_usage', {})
                    }
                )
            
            # 교사 모델 답변 처리
            if not hasattr(case, 'expected_output') or not case.expected_output:
                if self.teacher_model is None:
                    raise ValueError(
                        "expected_output이 없고 teacher_model도 설정되지 않았습니다. expected_output을 직접 제공하거나 teacher_model을 설정해주세요.")
                teacher_response = await self._a_generate_teacher_answer(case)
            else:
                teacher_response = AIMessage(
                    content=case.expected_output,
                    response_metadata={
                        'model_name': getattr(case, 'reference_model_name', ''),
                        'token_usage': getattr(case, 'reference_token_usage', {})
                    }
                )
            
            # 두 답변 비교 평가
            compare_response = await self._a_compare_answers(case, student_response, teacher_response)
            
            result = self._process_comparison_result(case, student_response, teacher_response, compare_response)
            
            return result

        except Exception as e:
            print(f"Error processing test case: {str(e)}")
            return LLMResult(
                input=getattr(case, 'input', ''),
                student_answer=getattr(case, 'output', ''),
                teacher_answer=getattr(case, 'expected_output', ''),
                expected_output=getattr(case, 'expected_output', ''),
                context=getattr(case, 'context', None),
                retrieval_context=getattr(case, 'retrieval_context', None),
                reasoning="처리 중 오류 발생: " + str(e),
                score=0,
                metadata=getattr(case, 'metadata', {})
            )

    def _generate_student_answer(self, case: LLMTestCase) -> AIMessage:
        """
        학생 LLM을 사용하여 동기 방식으로 답변을 생성합니다.
        """
        try:
            return self.student_model.invoke(case.input)
        except Exception as e:
            raise RuntimeError(f"학생 모델 답변 생성 중 오류 발생: {str(e)}")

    async def _a_generate_student_answer(self, case: LLMTestCase) -> AIMessage:
        """
        학생 LLM을 사용하여 비동기 방식으로 답변을 생성합니다.
        """
        try:
            return await self.student_model.ainvoke(case.input)
        except Exception as e:
            raise RuntimeError(f"학생 모델 답변 생성 중 오류 발생: {str(e)}")

    def _generate_teacher_answer(self, case: LLMTestCase) -> AIMessage:
        """
        교사 LLM을 사용하여 동기 방식으로 답변을 생성합니다.
        """
        try:
            return self.teacher_model.invoke(case.input)
        except Exception as e:
            raise RuntimeError(f"교사 모델 답변 생성 중 오류 발생: {str(e)}")

    async def _a_generate_teacher_answer(self, case: LLMTestCase) -> AIMessage:
        """
        교사 LLM을 사용하여 비동기 방식으로 답변을 생성합니다.
        """
        try:
            return await self.teacher_model.ainvoke(case.input)
        except Exception as e:
            raise RuntimeError(f"교사 모델 답변 생성 중 오류 발생: {str(e)}")

    def _compare_answers(self, case: LLMTestCase, student_response: AIMessage, teacher_response: AIMessage) -> AIMessage:
        """
        동기 방식으로 judge 모델을 사용하여 두 답변을 비교 평가합니다.
        """
        try:
            comparison_prompt = self.template.format_prompt(
                question=case.input, 
                student_answer=student_response.content, 
                teacher_answer=teacher_response.content
            )
            comparison_response = self.judge_model.invoke(comparison_prompt)
            return comparison_response
        except Exception as e:
            raise RuntimeError(f"답변 비교 평가 중 오류 발생: {str(e)}")

    async def _a_compare_answers(self, case: LLMTestCase, student_response: AIMessage, teacher_response: AIMessage) -> AIMessage:
        """
        비동기 방식으로 judge 모델을 사용하여 두 답변을 비교 평가합니다.
        """
        try:
            comparison_prompt = self.template.format_prompt(
                question=case.input, 
                student_answer=student_response.content, 
                teacher_answer=teacher_response.content
            )
            comparison_response = await self.judge_model.ainvoke(comparison_prompt)
            return comparison_response
        except Exception as e:
            raise RuntimeError(f"답변 비교 평가 중 오류 발생: {str(e)}")

    def _validate_testcase(self, case: LLMTestCase) -> None:
        """
        테스트케이스의 유효성을 검사합니다.
        """
        if not hasattr(case, 'input'):
            raise ValueError("테스트케이스는 'input' 속성을 가져야 합니다.")
        if not case.input:
            raise ValueError("input이 비어있습니다.")
    
    def _log_process_info(self, case: LLMTestCase, student_response: AIMessage, 
                         teacher_response: AIMessage, compare_response: AIMessage):
        if self.verbose_mode:
            print(f"Input: {case.input}")
            print(f"Student answer: {student_response.content}")
            print(f"Teacher answer: {teacher_response.content}")
            print(f"Comparison result: {compare_response.content}")
            print(f"Reasoning: {case.reasoning}")

    @classmethod
    def get_comparison_category(cls):
        json_path = Path(__file__).parent.parent.parent / 'prompt_storage' / 'arena_comparison_prompt.json'
        data = load_json(json_path)
        return list(data['category'].keys())

    @property
    def __name__(self):
        return "ArenaMetric"