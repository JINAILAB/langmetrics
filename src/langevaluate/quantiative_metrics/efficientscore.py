import torch
import torch.nn as nn
import tiktoken
import numpy as np
from typing import List, Union, Dict, Optional


class EfficientScore(nn.Module):
    """
    EfficientScore - 텍스트의 토큰 길이 기반 효율성 점수
    
    공식: score = (1 - (L/10000)^2)^(1/2)
    여기서 L은 토큰 길이입니다.
    
    점수가 높을수록 더 효율적(짧고 간결한) 텍스트를 의미합니다.
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Args:
            model_name: tiktoken 인코딩에 사용할 모델 이름
                       ("gpt-4", "gpt-3.5-turbo", "text-davinci-003" 등)
        """
        super(EfficientScore, self).__init__()
        
        # tiktoken 인코더 초기화
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # 모델이 없으면 기본 인코더 사용
            self.encoder = tiktoken.get_encoding("cl100k_base")
        
        self.max_tokens = 10000  # 공식에서 사용하는 최대 토큰 수
    
    def _calculate_score(self, token_length: int) -> float:
        """
        토큰 길이를 기반으로 효율성 점수 계산
        
        Args:
            token_length: 텍스트의 토큰 개수
            
        Returns:
            float: 효율성 점수 (0~1 사이)
        """
        # 토큰 길이가 max_tokens를 초과하면 0 반환
        if token_length >= self.max_tokens:
            return 0.0
        
        # 공식: score = (1 - (L/10000)^2)^(1/2)
        ratio = token_length / self.max_tokens
        score = np.sqrt(1 - ratio ** 2)
        
        return score
    
    def _count_tokens(self, text: str) -> int:
        """텍스트의 토큰 개수 계산"""
        if not text:
            return 0
        return len(self.encoder.encode(text))
    
    def forward(
        self, 
        texts: Union[str, List[str]]
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        텍스트의 효율성 점수를 계산
        
        Args:
            texts: 평가할 텍스트 (단일 문자열 또는 리스트)
            
        Returns:
            dict 또는 list of dict: {
                'score': 효율성 점수,
                'token_count': 토큰 개수,
                'efficiency_ratio': 효율성 비율 (1 - token_count/max_tokens)
            }
        """
        # 단일 텍스트 처리
        if isinstance(texts, str):
            token_count = self._count_tokens(texts)
            score = self._calculate_score(token_count)
            
            return score
        
        # 여러 텍스트 처리
        results = []
        for text in texts:
            token_count = self._count_tokens(text)
            score = self._calculate_score(token_count)
            
            results.append(score)
    
        return results
    

if __name__ == "__main__":
    # EfficientScore 초기화
    scorer = EfficientScore(model_name="gpt-4")
    
    # 예제 1: 단일 텍스트 평가
    text = "The patient presents with acute chest pain, shortness of breath, and elevated troponin levels. ECG shows ST-segment elevation in leads V1-V4. Immediate cardiac catheterization is recommended."
    
    result = scorer(text)
    print("=== Single Text Evaluation ===")
    print(f"Text: {text[:100]}...")
    print(f"Efficiency Score: {result['score']:.4f}")
    print(f"Token Count: {result['token_count']}")
    print(f"Efficiency Ratio: {result['efficiency_ratio']:.4f}")
    print()
    
    # 예제 2: 여러 텍스트 평가
    texts = [
        "Patient has fever.",
        "The patient is presenting with an elevated body temperature that is above the normal range, which is commonly referred to as fever or pyrexia in medical terminology.",
        "The individual who is currently under medical observation and care has been found to exhibit a physiological condition characterized by an abnormal elevation of the core body temperature beyond the typical homeostatic range of approximately 36.5-37.5 degrees Celsius, a state which is medically recognized and commonly designated as fever, febrile response, or pyrexia."
    ]
    
    results = scorer(texts)
    print("=== Multiple Texts Evaluation ===")
    for i, result in enumerate(results['individual_results']):
        print(f"Text {i+1}: Score={result['score']:.4f}, Tokens={result['token_count']}")
    print(f"\nMean Score: {results['mean_score']:.4f}")
    print(f"Std Score: {results['std_score']:.4f}")
    print(f"Total Tokens: {results['total_tokens']}")
    print()
