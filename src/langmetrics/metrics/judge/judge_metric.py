from langmetrics.metrics.base_metric import BaseMetric
from typing import Union, Literal, List
from langmetrics.llmtestcase import LLMTestCase
from langmetrics.metrics.judge.judge_template import JudgeTemplate
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

        if judge_template is None:
            self.template = JudgeTemplate(self.template_language, self.generate_template_type, self.category)
        else:
            self.template = judge_template

        self.template_for_judge = self.template.get_prompt_for_score()

    def measure(
        self,
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> Union[JudgeResult, List[JudgeResult]]:
        """
        동기 방식으로 모델 답변의 정확도를 평가합니다.
        """
        testcases = self._normalize_testcases(testcase)
        results = [self._process_single_case(case) for case in testcases]
        return results[0] if isinstance(testcase, LLMTestCase) else results

    async def ameasure(
        self,
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> Union[JudgeResult, List[JudgeResult]]:
        """
        비동기 방식으로 모델 답변의 정확도를 평가합니다.
        """
        testcases = self._normalize_testcases(testcase)
        results = await asyncio.gather(*[self._a_process_single_case(case) for case in testcases])
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

    def _process_single_case(self, case: LLMTestCase) -> JudgeResult:
        """
        동기 방식으로 단일 테스트케이스를 처리합니다.
        """
        metadata = {}
        metadata = {'student_template_language': self.template_language}
        metadata['student_model_name'] = response.response_metadata.get('model_name', '')
        try:
            self._validate_testcase(case)

            if not case.output:
                if self.answer_model is None:
                    raise ValueError(
                        "output이 없고 answer_model도 설정되지 않았습니다. "
                        "output을 직접 제공하거나 answer_model을 설정해주세요."
                    )
                case_output = self._generate_answer_one_case(case)
                case.output = case_output.content
                token_usage['answer'] = {
                    'completion_tokens': case_output.response_metadata['token_usage'].get('completion_tokens'),
                    'prompt_tokens': case_output.response_metadata['token_usage'].get('prompt_tokens'),
                    'total_tokens': case_output.response_metadata['token_usage'].get('total_tokens')
                }

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
                student_answer=case.output,
                reasoning=reasoning,
                score=score,
                language=self.template_language,
                token_usage=token_usage
            )

        except Exception as e:
            print(f"Error processing test case: {str(e)}")
            return JudgeResult(
                question=getattr(case, 'input', ''),
                student_answer=getattr(case, 'output', ''),
                reasoning=getattr(case, 'reasoning', ''),
                score=False,
                language=self.template_language,
                token_usage=token_usage
            )

    async def _a_process_single_case(self, case: LLMTestCase) -> JudgeResult:
        """
        비동기 방식으로 단일 테스트케이스를 처리합니다.
        """
        token_usage = {}
        try:
            self._validate_testcase(case)

            if not case.output:
                if self.answer_model is None:
                    raise ValueError(
                        "output이 없고 answer_model도 설정되지 않았습니다. "
                        "output을 직접 제공하거나 answer_model을 설정해주세요."
                    )
                case_output = await self._a_generate_answer_one_case(case)
                case.output = case_output.content
                token_usage['answer'] = {
                    'completion_tokens': case_output.response_metadata['token_usage'].get('completion_tokens'),
                    'prompt_tokens': case_output.response_metadata['token_usage'].get('prompt_tokens'),
                    'total_tokens': case_output.response_metadata['token_usage'].get('total_tokens')
                }

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
                student_answer=case.output,
                reasoning=reasoning,
                score=score,
                language=self.template_language,
                token_usage=token_usage
            )

        except Exception as e:
            print(f"Error processing test case: {str(e)}")
            return JudgeResult(
                question=getattr(case, 'input', ''),
                student_answer=getattr(case, 'output', ''),
                reasoning=getattr(case, 'reasoning', ''),
                score=False,
                language=self.template_language,
                token_usage=token_usage
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

    def _parse_evaluation_response(self, response_content: str, token_usage: dict) -> dict:
        """
        평가 모델의 응답을 파싱하여 점수와 평가 근거를 반환합니다.
        """
        try:
            parsed_output = trimAndLoadJson(response_content)
            if isinstance(parsed_output, dict):
                score = float(parsed_output.get('score', 0))
                reasoning = parsed_output.get('reason', 'No reasoning provided')
            else:
                score = float(parsed_output) if parsed_output is not None else 0
                reasoning = 'Direct score provided without reasoning'

            if self.verbose_mode:
                print(f"Parsed evaluation result: {{'score': {score}, 'reasoning': {reasoning}}}")
                print(f"Token usage: {token_usage}")

            return {'score': score, 'reasoning': reasoning, 'token_usage': token_usage}
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: 평가 결과 파싱 실패. Raw output: {response_content}")
            print(f"Error: {str(e)}")
            return {'score': 0, 'reasoning': f'Failed to parse evaluation response: {str(e)}', 'token_usage': token_usage}

    def _evaluate_answer(self, case: LLMTestCase) -> dict:
        """
        동기 방식으로 score 모델을 사용하여 답변을 평가합니다.
        """
        evaluation_tokens = {}
        try:
            evaluation_prompt = self.template.format_prompt(question=case.input, answer=case.output)
            evaluation_response = self.score_model.invoke(evaluation_prompt)
            print(evaluation_response.content)

            evaluation_tokens = {
                'completion_tokens': evaluation_response.response_metadata['token_usage'].get('completion_tokens'),
                'prompt_tokens': evaluation_response.response_metadata['token_usage'].get('prompt_tokens'),
                'total_tokens': evaluation_response.response_metadata['token_usage'].get('total_tokens')
            }

            return self._parse_evaluation_response(evaluation_response.content, evaluation_tokens)
        except Exception as e:
            error_msg = f"답변 평가 중 오류 발생: {str(e)}"
            print(error_msg)
            return {'score': 0, 'reasoning': error_msg, 'token_usage': evaluation_tokens}

    async def _a_evaluate_answer(self, case: LLMTestCase) -> dict:
        """
        비동기 방식으로 score 모델을 사용하여 답변을 평가합니다.
        """
        evaluation_tokens = {}
        try:
            evaluation_prompt = self.template.format_prompt(question=case.input, answer=case.output)
            evaluation_response = await self.score_model.ainvoke(evaluation_prompt)

            evaluation_tokens = {
                'completion_tokens': evaluation_response.response_metadata['token_usage'].get('completion_tokens'),
                'prompt_tokens': evaluation_response.response_metadata['token_usage'].get('prompt_tokens'),
                'total_tokens': evaluation_response.response_metadata['token_usage'].get('total_tokens')
            }

            return self._parse_evaluation_response(evaluation_response.content, evaluation_tokens)
        except Exception as e:
            error_msg = f"답변 평가 중 오류 발생: {str(e)}"
            print(error_msg)
            return {'score': 0, 'reasoning': error_msg, 'token_usage': evaluation_tokens}

    def _process_output(self, parsed_output: dict, case: LLMTestCase) -> dict:
        """
        파싱된 출력을 처리하여 결과 딕셔너리를 생성합니다.
        """
        answer = parsed_output.get('answer', '').strip()
        if not answer:
            print(f"경고: 생성된 답변이 비어있습니다. 입력: {case.input}")
            result = {'answer': ''}
            if self.generate_template_type == 'reasoning':
                result['reasoning'] = parsed_output.get('reasoning', '')
            return result

        result = {'answer': answer}
        if self.generate_template_type == 'reasoning':
            result['reasoning'] = parsed_output.get('reasoning', '')
        return result

    def _validate_testcase(self, case: LLMTestCase) -> None:
        """
        테스트케이스의 유효성을 검사합니다.
        """
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
