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
from langmetrics.result import LLMResult
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
        """
        JudgeMetric 클래스 초기화 메서드
        
        매개변수:
            score_model (Union[ChatOpenAI, ChatAnthropic, ChatClovaX]): 
                답변을 평가하는 LLM 모델
            category (str, 선택 사항): 
                평가 카테고리 (예: 'temporal_relevance', 'accuracy' 등)
                기본값은 'temporal_relevance'
            answer_model (Union[ChatOpenAI, ChatAnthropic, ChatClovaX], 선택 사항): 
                답변을 생성하는 LLM 모델, None이면 답변이 이미 제공되어야 함
                기본값은 None
            verbose_mode (bool, 선택 사항): 
                상세 로깅 활성화 여부
                기본값은 False
            template_language (Literal['ko', 'en'], 선택 사항): 
                템플릿 언어 ('ko': 한국어, 'en': 영어)
                기본값은 'ko'
            generate_template_type (Literal['reasoning', 'only_answer'], 선택 사항): 
                템플릿 유형 ('reasoning': 이유 포함, 'only_answer': 답변만)
                기본값은 'reasoning'
            judge_template (Union[str, JudgeTemplate, BaseTemplate], 선택 사항): 
                평가에 사용할 사용자 정의 템플릿
                기본값은 None (기본 템플릿 사용)
                
        입력 예시:
            score_model = ChatOpenAI(model_name="gpt-4")
            judge_metric = JudgeMetric(
                score_model=score_model,
                category='temporal_relevance',
                template_language='ko'
            )
        """
        self.answer_model = answer_model
        self.score_model = score_model
        self.score_model_name = score_model.model_name
        self.verbose_mode = verbose_mode
        self.template_language = template_language
        self.generate_template_type = generate_template_type
        self.category = category
        # 템플릿 클래스 초기화
        self.template = (JudgeTemplate(self.template_language, self.generate_template_type, self.category) 
                        if judge_template is None else judge_template)
        # judge를 위한 템플릿 문자열 가져오기
        self.template_for_judge = self.template.get_prompt_for_score()

    def measure(
        self,
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> Union[LLMResult, List[LLMResult]]:
        """
        동기 방식으로 모델 답변의 정확도를 평가합니다.
        
        이 메서드는 입력된 테스트 케이스(단일 또는 다수)에 대해 LLM 응답을 평가합니다.
        테스트 케이스에 output이 없고 answer_model이 설정되어 있으면 답변을 먼저 생성한 후 평가합니다.
        
        매개변수:
            testcase (Union[LLMTestCase, List[LLMTestCase], LLMDataset]): 
                평가할 테스트 케이스(단일 테스트 케이스, 테스트 케이스 리스트 또는 LLMDataset)
                
        반환값:
            Union[LLMResult, List[LLMResult]]: 
                단일 테스트 케이스의 경우 LLMResult 객체,
                다수 테스트 케이스의 경우 ResultDataset(LLMResult 리스트)
                
        예외:
            ValueError: 테스트 케이스가 유효하지 않은 경우
            TypeError: 테스트 케이스 유형이 잘못된 경우
            
        입력 예시:
            testcase = LLMTestCase(
                input="2023년 노벨 물리학상 수상자는 누구인가요?",
                output="2023년 노벨 물리학상은 앤 레러, 피에르 아고스티니, 페렌츠 크라우스가 수상했습니다."
            )
            result = judge_metric.measure(testcase)
            
        출력 예시:
            LLMResult(
                input="2023년 노벨 물리학상 수상자는 누구인가요?",
                student_answer="2023년 노벨 물리학상은 앤 레러, 피에르 아고스티니, 페렌츠 크라우스가 수상했습니다.",
                teacher_answer="{\"score\": 5, \"reasoning\": \"해당 답변은 2023년 노벨 물리학상 수상자 정보를 정확하게 제공하고 있습니다.\"}",
                score=5,
                reasoning="해당 답변은 2023년 노벨 물리학상 수상자 정보를 정확하게 제공하고 있습니다.",
                metadata={...}
            )
        """
        # 테스트 케이스를 표준화된 형식으로 변환
        testcases = self._normalize_testcases(testcase)
        # 각 테스트 케이스를 처리하여 결과 생성
        results = ResultDataset([self._process_single_case(case) for case in testcases])
        # 입력이 단일 테스트 케이스였다면 단일 결과 반환, 아니면 ResultDataset 반환
        return results[0] if isinstance(testcase, LLMTestCase) else results

    async def ameasure(
        self,
        testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> Union[LLMResult, List[LLMResult]]:
        """
        비동기 방식으로 모델 답변의 정확도를 평가합니다.
        
        이 메서드는 입력된 테스트 케이스(단일 또는 다수)에 대해 비동기적으로 LLM 응답을 평가합니다.
        테스트 케이스에 output이 없고 answer_model이 설정되어 있으면 답변을 먼저 비동기적으로 생성한 후 평가합니다.
        
        매개변수:
            testcase (Union[LLMTestCase, List[LLMTestCase], LLMDataset]): 
                평가할 테스트 케이스(단일 테스트 케이스, 테스트 케이스 리스트 또는 LLMDataset)
                
        반환값:
            Union[LLMResult, List[LLMResult]]: 
                단일 테스트 케이스의 경우 LLMResult 객체,
                다수 테스트 케이스의 경우 ResultDataset(LLMResult 리스트)
                
        예외:
            ValueError: 테스트 케이스가 유효하지 않은 경우
            TypeError: 테스트 케이스 유형이 잘못된 경우
            
        입력 예시:
            testcases = [
                LLMTestCase(input="2023년 노벨 물리학상 수상자는 누구인가요?"),
                LLMTestCase(input="2023년 노벨 화학상 수상자는 누구인가요?")
            ]
            results = await judge_metric.ameasure(testcases)
            
        출력 예시:
            ResultDataset([
                LLMResult(input="2023년 노벨 물리학상 수상자는 누구인가요?", score=5, ...),
                LLMResult(input="2023년 노벨 화학상 수상자는 누구인가요?", score=4, ...)
            ])
        """
        # 테스트 케이스를 표준화된 형식으로 변환
        testcases = self._normalize_testcases(testcase)
        results = await asyncio.gather(*[self._a_process_single_case(case) for case in testcases])
        results = ResultDataset(results)
        return results[0] if isinstance(testcase, LLMTestCase) else results

    def _normalize_testcases(
        self, testcase: Union[LLMTestCase, List[LLMTestCase], LLMDataset]
    ) -> List[LLMTestCase]:
        """
        입력된 테스트케이스를 리스트 형태로 표준화합니다.
        
        다양한 형태(단일 테스트 케이스, 테스트 케이스 리스트, LLMDataset)로 제공될 수 있는
        테스트 케이스 입력을 일관된 형식의 리스트로 변환합니다.
        
        매개변수:
            testcase (Union[LLMTestCase, List[LLMTestCase], LLMDataset]): 
                표준화할 테스트 케이스
                
        반환값:
            List[LLMTestCase]: 표준화된 테스트 케이스 리스트
            
        예외:
            ValueError: 빈 리스트가 제공된 경우
            TypeError: 잘못된 입력 유형이 제공된 경우
        """
        if isinstance(testcase, LLMDataset):
            # LLMDataset 객체는 그대로 반환
            return testcase
        elif isinstance(testcase, list):
            # 리스트가 비어있는지 확인
            if not testcase:
                raise ValueError("Empty list provided")
            # 리스트의 모든 항목이 LLMTestCase인지 확인
            if not all(isinstance(item, LLMTestCase) for item in testcase):
                raise TypeError("All items in the list must be LLMTestCase instances")
            return testcase
        elif isinstance(testcase, LLMTestCase):
            # 단일 테스트 케이스를 리스트로 변환
            return [testcase]
        else:
            # 지원되지 않는 입력 유형
            raise TypeError("Invalid input type. Expected LLMTestCase, List[LLMTestCase], or LLMDataset")
        
    def _process_generated_answer(self, case: LLMTestCase, response: AIMessage, evaluate_response: AIMessage) -> dict:
        """
        LLM 응답을 처리하여 JSON 파싱, 메타데이터 업데이트 및 결과 생성까지 수행합니다.
        
        생성된 답변(response)과 그 평가(evaluate_response)를 처리하여 최종 LLMResult 객체를 생성합니다.
        평가 응답에서 JSON을 추출하고, 토큰 사용량 등의 메타데이터를 수집합니다.
        
        매개변수:
            case (LLMTestCase): 처리 중인 테스트 케이스
            response (AIMessage): 학생 모델(answer_model)의 응답
            evaluate_response (AIMessage): 평가 모델(score_model)의 응답
            
        반환값:
            LLMResult: 처리된 결과 객체
            
        입력 예시:
            case = LLMTestCase(input="최근 기후 변화가 생태계에 미치는 영향은?")
            response = AIMessage(content="기후 변화는 여러 생물 종의 멸종 위기를 초래하고 있습니다...")
            evaluate_response = AIMessage(content="{\"score\": 4, \"reasoning\": \"포괄적인 답변이지만 구체적인 사례가 부족합니다.\"}")
            
        출력 예시:
            LLMResult(
                input="최근 기후 변화가 생태계에 미치는 영향은?",
                student_answer="기후 변화는 여러 생물 종의 멸종 위기를 초래하고 있습니다...",
                teacher_answer="{\"score\": 4, \"reasoning\": \"포괄적인 답변이지만 구체적인 사례가 부족합니다.\"}",
                score=4,
                reasoning="포괄적인 답변이지만 구체적인 사례가 부족합니다.",
                metadata={...}
            )
        """
        # 테스트 케이스의 output 필드 업데이트
        case.output = response.content
        
        # 메타데이터 초기화 및 채우기
        metadata = {'teacher_template_language': self.template_language}
        metadata['student_model_name'] = response.response_metadata.get('model_name', '')
        metadata['teacher_model_name'] = evaluate_response.response_metadata.get('model_name', '')
        
        # 학생 모델(answer_model)의 토큰 사용량 정보 수집
        student_token_usage = response.response_metadata.get('token_usage', {})
        metadata['student_token_usage'] = {
            'completion_tokens': student_token_usage.get('completion_tokens'),
            'prompt_tokens': student_token_usage.get('prompt_tokens'),
            'total_tokens': student_token_usage.get('total_tokens')
        }
        
        # 평가 모델(score_model)의 토큰 사용량 정보 수집
        teacher_token_usage = evaluate_response.response_metadata.get('token_usage', {})
        metadata['teacher_token_usage'] = {
            'completion_tokens': teacher_token_usage.get('completion_tokens'),
            'prompt_tokens': teacher_token_usage.get('prompt_tokens'),
            'total_tokens': teacher_token_usage.get('total_tokens')
        }
        
        # 평가 응답에서 JSON 추출 시도
        try:
            parsed_output = trimAndLoadJson(evaluate_response.content)
            parsed_output = {
                'score': parsed_output.get('score', ''),
                'reasoning': parsed_output.get('reasoning', '')
            }
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 경고 메시지 출력(verbose_mode가 활성화된 경우)
            if self.verbose_mode:
                print(f"Warning: JSON parsing failed. Raw output: {evaluate_response.content}")
            parsed_output = {'answer': '', 'reasoning': ''}
            metadata['error'] = f"Warning: JSON parsing failed. Raw output: {evaluate_response.content}"
        
        # 테스트 케이스의 reasoning 필드 업데이트
        case.reasoning = parsed_output.get('reasoning', '')
        
        # verbose mode가 활성화된 경우 로그 출력
        self._log_process_info(case, evaluate_response)
            
        # 최종 LLMResult 객체 생성 및 반환
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
        
        테스트 케이스를 검증하고, 필요한 경우 답변을 생성한 후,
        그 답변을 평가하여 최종 결과를 반환합니다.
        
        매개변수:
            case (LLMTestCase): 처리할 테스트 케이스
            
        반환값:
            LLMResult: 처리된 결과 객체
            
        예외:
            ValueError: 테스트 케이스가 유효하지 않거나 필요한 모델이 설정되지 않은 경우
            
        입력 예시:
            case = LLMTestCase(input="인공지능의 윤리적 문제점은 무엇인가요?")
            
        출력 예시:
            LLMResult(
                input="인공지능의 윤리적 문제점은 무엇인가요?",
                student_answer="인공지능의 윤리적 문제점으로는 프라이버시 침해, 편향성, 의사결정의 투명성 부족 등이 있습니다...",
                teacher_answer="{\"score\": 5, \"reasoning\": \"주요 윤리적 문제점들을 포괄적으로 다루고 있음\"}"
                score=5,
                reasoning="주요 윤리적 문제점들을 포괄적으로 다루고 있음",
                metadata={...}
            )
        """
        try:
            # 테스트 케이스 검증
            self._validate_testcase(case)
            
            # output이 없는 경우 answer_model을 사용하여 답변 생성
            if not case.output:
                if self.answer_model is None:
                    raise ValueError(
                        "output이 없고 answer_model도 설정되지 않았습니다. output을 직접 제공하거나 answer_model을 설정해주세요.")
                response = self._generate_answer_one_case(case)
            else:
                # output이 이미 있는 경우 AIMessage 객체로 변환
                response = AIMessage(
                    content=case.output,
                    response_metadata={
                        'model_name': getattr(case, 'model_name', ''),
                        'token_usage': getattr(case, 'token_usage', {})
                    }
                )
                
            # 답변 평가
            evaluate_response = self._evaluate_answer(case, response)
            
            # 생성된 답변 처리
            result = self._process_generated_answer(case, response, evaluate_response)
            return result

        except Exception as e:
            # 오류 발생 시 로그 출력 및 기본 결과 반환
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
        
        테스트 케이스를 검증하고, 필요한 경우 비동기적으로 답변을 생성한 후,
        그 답변을 평가하여 최종 결과를 반환합니다.
        
        매개변수:
            case (LLMTestCase): 처리할 테스트 케이스
            
        반환값:
            LLMResult: 처리된 결과 객체
            
        예외:
            ValueError: 테스트 케이스가 유효하지 않거나 필요한 모델이 설정되지 않은 경우
            
        입력 예시:
            case = LLMTestCase(input="양자 컴퓨팅의 미래 전망은 어떻게 되나요?")
            
        출력 예시:
            LLMResult(
                input="양자 컴퓨팅의 미래 전망은 어떻게 되나요?",
                student_answer="양자 컴퓨팅은 암호화, 신약 개발, 기계 학습 등의 분야에서 혁신을 가져올 것으로 예상됩니다...",
                teacher_answer="{\"score\": 4, \"reasoning\": \"주요 응용 분야는 언급했으나, 기술적 한계에 대한 설명이 부족합니다.\"}"
                score=4,
                reasoning="주요 응용 분야는 언급했으나, 기술적 한계에 대한 설명이 부족합니다.",
                metadata={...}
            )
        """
        try:
            # 테스트 케이스 검증
            self._validate_testcase(case)
            
            # output이 없는 경우 answer_model을 사용하여 비동기적으로 답변 생성
            if not case.output:
                if self.answer_model is None:
                    raise ValueError(
                        "output이 없고 answer_model도 설정되지 않았습니다. output을 직접 제공하거나 answer_model을 설정해주세요.")
                response = self._a_generate_answer_one_case(case)
            else:
                # output이 이미 있는 경우 AIMessage 객체로 변환
                response = AIMessage(
                    content=case.output,
                    response_metadata={
                        'model_name': getattr(case, 'model_name', ''),
                        'token_usage': getattr(case, 'token_usage', {})
                    }
                )
                
            # 답변 평가
            evaluate_response = self._evaluate_answer(case, response)
            
            # 생성된 답변 처리
            result = self._process_generated_answer(case, response, evaluate_response)
            
            return result

        except Exception as e:
            # 오류 발생 시 로그 출력 및 기본 결과 반환
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
        
        answer_model을 사용하여 테스트 케이스의 입력에 대한 답변을 생성합니다.
        
        매개변수:
            case (LLMTestCase): 답변을 생성할 테스트 케이스
            
        반환값:
            AIMessage: 생성된 답변을 포함하는 AIMessage 객체
            
        예외:
            RuntimeError: 답변 생성 중 오류가 발생한 경우
            
        입력 예시:
            case = LLMTestCase(input="자연어 처리란 무엇인가요?")
            
        출력 예시:
            AIMessage(
                content="자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리하는 인공지능의 한 분야입니다...",
                response_metadata={...}
            )
        """
        try:
            # answer_model을 사용하여 답변 생성
            return self.answer_model.invoke(case.input)
        except Exception as e:
            # 오류 발생 시 예외 발생
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
