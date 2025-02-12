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
    def __init__(
        self,
        answer_model: Union[ChatOpenAI, ChatAnthropic, ChatClovaX] = None,
        verbose_mode: bool = False,
        template_language: Literal['ko', 'en'] = 'ko',
        generate_template_type: Literal['reasoning', 'only_answer'] = 'reasoning',
        template: Optional[Union[MCQTemplate, BaseTemplate]] = None,
    ):
        """
        객관식 문제 평가를 위한 메트릭 클래스

        Args:
            answer_model: LLM 답변 생성 모델
            verbose_mode: 상세 로그 출력 여부
            template_language: 템플릿 언어 ('ko' 또는 'en')
            generate_template_type: 템플릿 유형 ('reasoning' 또는 'only_answer')
            template: 답변 생성에 사용될 프롬프트 템플릿 (없으면 기본 MCQTemplate 사용)
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
        """
        testcases = self._normalize_testcases(testcase)
        results = ([self._process_single_case(case) for case in testcases])
        return results[0] if isinstance(testcase, LLMTestCase) else ResultDataset(results)

    async def ameasure(
        self, 
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> Union[LLMResult, List[LLMResult]]:
        """
        모델 답변의 정확도를 비동기적으로 평가합니다.
        """
        testcases = self._normalize_testcases(testcase)
        results = await asyncio.gather(*(self._a_process_single_case(case) for case in testcases))
        return results[0] if isinstance(testcase, LLMTestCase) else ResultDataset(results)

    def _normalize_testcases(
        self, 
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> List[LLMTestCase]:
        """입력된 테스트케이스를 리스트 형태로 반환합니다."""
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

    def _process_generated_answer(self, response: AIMessage, case: LLMTestCase) -> dict:
        """
        LLM 응답을 처리하여 JSON 파싱, 메타데이터 업데이트 및 결과 생성까지 수행합니다.
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
        """단일 테스트케이스를 동기적으로 처리합니다."""
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
        """단일 테스트케이스를 비동기적으로 처리합니다."""
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
        """LLM을 사용하여 동기적으로 답변을 생성합니다."""
        try:
            prompt = self._build_prompt(case)
            response = self.answer_model.invoke(prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")

    async def _a_generate_answer_one_case(self, case: LLMTestCase) -> AIMessage:
        """LLM을 사용하여 비동기적으로 답변을 생성합니다."""
        try:
            prompt = self._build_prompt(case)
            response = await self.answer_model.ainvoke(prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"답변 생성 중 오류 발생: {str(e)}")

    def _build_prompt(self, case: LLMTestCase) -> str:
        """테스트케이스의 choices를 포맷팅하여 프롬프트를 생성합니다."""
        choices_str = '\n'.join(f"{chr(65 + i)}: {value}" for i, value in enumerate(case.choices))
        return self.template_for_answer.format_messages(question=case.input, choices=choices_str)

    def _validate_testcase(self, case: LLMTestCase) -> None:
        """테스트케이스의 유효성을 검사합니다."""
        if not hasattr(case, 'input') or not hasattr(case, 'choices') or not hasattr(case, 'expected_output'):
            raise ValueError("테스트케이스는 'input'과 'choices', 'expected_output' 속성을 가져야 합니다.")
        if not case.choices:
            raise ValueError("choices가 비어있습니다.")
        if not case.input:
            raise ValueError("input이 비어있습니다.")
        if not case.expected_output:
            raise ValueError("expected_output이 비어있습니다.")
        
    def _log_process_info(self, case: LLMTestCase):
        if self.verbose_mode:
            print(f"Input: {case.input}")
            print(f"Generated answer: {case.output}")
            print(f"Expected answer: {case.expected_output}")
            print(f"Is correct: {case.output == case.expected_output}")
            print(f"Reasoning: {case.reasoning}")
            

    @property
    def __name__(self):
        return "MCQMetric"
