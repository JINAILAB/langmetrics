from abc import ABC, abstractmethod
from langevaluate.config import ModelConfig
from typing import Any, Dict
from langchain_core.rate_limiters import InMemoryRateLimiter
import math
import asyncio
import json
from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError
from langchain_core.messages import AIMessage
from langevaluate.utils import trimAndLoadJson

class BaseFactory(ABC):
    """LLM 생성을 위한 추상 팩토리 클래스
    
    각 AI 제공업체별 LLM 인스턴스 생성을 위한 인터페이스를 정의합니다.
    팩토리 메서드 패턴을 사용하여 구체적인 LLM 구현체 생성을 서브클래스에 위임합니다.
    """
    def create_llm(self, config: ModelConfig, temperature: float, rpm: int = None, max_retries: int = 60, **kwargs) -> Any:
        """LLM 인스턴스를 생성하고 ainvoke 메서드를 process_single_input으로 래핑합니다."""
        if rpm:
            rate_limiter = self._create_rate_limiter(rpm)
            kwargs['rate_limiter'] = rate_limiter
            
        # 원래 LLM 인스턴스 생성
        llm = self._create_llm(config, temperature, **kwargs)
        
        # 원래 ainvoke 메서드 저장
        original_ainvoke = llm.ainvoke
        
        
        # process_single_input 함수로 ainvoke 메서드 대체
        async def wrapped_ainvoke(input_data: Dict[str, Any], config=None, parse_json=False, **kwargs) -> str:
            """
            단일 입력에 대한 번역 결과를 얻습니다.
            (에러 처리 및 재시도 로직 포함)
            """
            e = None
            retries = 0
            while retries < max_retries:
                try:
                    result = await original_ainvoke(input_data, config, **kwargs)
                    
                    if parse_json:
                        try:
                            trimAndLoadJson(result.content)  # 출력 파싱
                        except json.JSONDecodeError as json_err:
                            print(f"JSONDecodeError in trimAndLoadJson: {json_err}, retry {retries+1}/{max_retries}")
                            e = json_err
                            retries += 1
                            await asyncio.sleep(10)
                            continue  # 다시 재시도
                        
                        except Exception as err:
                            print(f"Unexpected error in trimAndLoadJson: {err}, retry {retries+1}/{max_retries}")
                            retries += 1
                            e = err
                            await asyncio.sleep(10)
                            continue
                        

                    return result

                except json.JSONDecodeError as json_err:
                    print(f"JSONDecodeError in ainvoke: {json_err}, retry {retries+1}/{max_retries}")
                    e = json_err
                    retries += 1
                    await asyncio.sleep(10)

                except (APIError, RateLimitError, APITimeoutError, APIConnectionError) as err:
                    e = err
                    print(f"API Error: {err}, retry {retries+1}/{max_retries}")
                    retries += 1
                    await asyncio.sleep(10)

                except Exception as err:
                    print(f"Unexpected error: {err}, retry {retries+1}/{max_retries}")
                    retries += 1
                    e = err
                    await asyncio.sleep(10)

            return AIMessage(content="", answer_metadata={'error': e})
        
        # 원래 객체의 ainvoke 메서드를 래핑된 버전으로 교체
        object.__setattr__(llm, "ainvoke", wrapped_ainvoke)
        
        return llm
    
    
    
    @abstractmethod
    def _create_llm(self, config: ModelConfig, temperature: float, **kwargs) -> Any:
        """구체적인 LLM 인스턴스를 생성하는 메서드입니다. 하위 클래스에서 구현해야 합니다."""
        pass
    
    def _create_rate_limiter(self, rpm: int) -> InMemoryRateLimiter:
        """RPM 기반의 rate limiter를 생성합니다."""
        g = math.gcd(60, rpm)
        check_every_n_seconds = 60 // g
        requests_per_second = rpm // g
        max_bucket_size = rpm
        
        return InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=check_every_n_seconds,
            max_bucket_size=max_bucket_size,
        )