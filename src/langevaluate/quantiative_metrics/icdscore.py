from typing import List
import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
import re

def parse_icd_codes(icd_string: str) -> List[str]:
    """ICD 코드 문자열을 파싱하여 리스트로 변환"""
    if not icd_string or pd.isna(icd_string):
        return []
    
    # 문자열이 리스트 형태인 경우 (예: "['I259', 'E119']")
    if icd_string.startswith('[') and icd_string.endswith(']'):
        codes = icd_string.strip('[]').replace("'", "").replace('"', '').split(',')
    else:
        # 일반 문자열인 경우 (예: "I259,E119" 또는 "I259 E119")
        codes = re.split('[,\s]+', icd_string)
    
    # 공백 제거 및 빈 문자열 필터링
    cleaned_codes = [code.strip().upper() for code in codes if code.strip()]
    return cleaned_codes

def icd_similarity(code1: str, code2: str) -> float:
    """ICD-10 계층 기반 유사도 계산"""
    if not code1 or not code2:
        return 0.0
    
    # 점(.) 제거하여 통일
    clean_code1 = code1.replace('.', '')
    clean_code2 = code2.replace('.', '')
    
    max_len = min(len(clean_code1), len(clean_code2))
    common = 0
    
    # 공통 prefix 길이 계산
    for i in range(max_len):
        if clean_code1[i] == clean_code2[i]:
            common += 1
        else:
            break
    
    # 최대 깊이로 정규화
    max_depth = max(len(clean_code1), len(clean_code2))
    return common / max_depth if max_depth > 0 else 0.0

def hierarchical_f1(y_true: List[str], y_pred: List[str]) -> float:
    """계층적 부분 점수를 반영한 F1-score 계산"""
    # 빈 리스트 처리
    if len(y_true) == 0 and len(y_pred) == 0:
        return 1.0
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    
    # 유사도 매트릭스 계산
    sim_matrix = np.zeros((len(y_true), len(y_pred)))
    for i, true_code in enumerate(y_true):
        for j, pred_code in enumerate(y_pred):
            sim_matrix[i, j] = icd_similarity(true_code, pred_code)
    
    # Hungarian algorithm으로 최적 매칭 찾기
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    matched_score = sim_matrix[row_ind, col_ind].sum()
    
    # 부분 TP, FP, FN 계산
    partial_TP = matched_score
    FP = len(y_pred) - partial_TP
    FN = len(y_true) - partial_TP
    
    # Precision, Recall, F1 계산
    precision = partial_TP / len(y_pred) if len(y_pred) > 0 else 0
    recall = partial_TP / len(y_true) if len(y_true) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

class ICDScore:
    """ICD-10 계층적 F1-score 평가 클래스"""
    
    def __init__(self):
        pass
    
    def __call__(self, refs: List[List[str]], hyps: List[List[str]]) -> List[float]:
        """
        ICD 코드 리스트들에 대한 계층적 F1-score 계산
        
        Args:
            refs: 정답 ICD 코드 리스트들 [[code1, code2], [code3], ...]
            hyps: 예측 ICD 코드 리스트들 [[code1, code2], [code3], ...]
        
        Returns:
            List[float]: 각 샘플별 F1-score 리스트
        """
        if len(refs) != len(hyps):
            raise ValueError(f"참조와 예측 데이터의 길이가 다릅니다: {len(refs)} vs {len(hyps)}")
        
        scores = []
        for ref, hyp in zip(refs, hyps):
            score = hierarchical_f1(ref, hyp)
            scores.append(score)
        
        return scores