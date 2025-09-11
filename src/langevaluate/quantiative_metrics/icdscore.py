import re
import numpy as np
from typing import List, Union, Dict

def is_icd(code: str) -> bool:
    """
    코드 검증
    """
    if not isinstance(code, str):
        return False
    
    # 기본 ICD-10 패턴들 지원
    patterns = [
        r'^[A-Z]\d{2}\.\d$',      # A12.3 (ICD(2).py 원본)
        r'^[A-Z]\d{2}\.\d{1,2}$', # A12.34, A12.3
        r'^[A-Z]\d{2}$',          # A12
        r'^[A-Z]\d{2}\.\d{1,2}[A-Z]?$'  # A12.3A (추가 확장자)
    ]
    
    return any(re.match(pattern, code) for pattern in patterns)

def separate(code: str):
    """
    계층 분리
    """
    if not is_icd(code):
        raise ValueError(f"Invalid ICD code format: {code}")
    
    # 기본 분리: 챕터, 블록, 세부코드
    chapter = code[0]  # A
    
    if '.' in code:
        block = code[1:3]  # 12
        detail = code[4:]  # 3 또는 34
    else:
        block = code[1:3] if len(code) >= 3 else code[1:]
        detail = ""
    
    return chapter, block, detail

def hierarchical_match_score(pred_code: str, true_code: str) -> float:
    """
    계층적 부분 점수
    
    Returns:
        1.0: 완전 매칭 (level 3)
        0.8: 블록 매칭 (level 2) 
        0.4: 챕터 매칭 (level 1)
        0.0: 매칭 없음
    """
    if not pred_code or not true_code:
        return 0.0
    
    try:
        p_ch, p_bl, p_dt = separate(pred_code)
        t_ch, t_bl, t_dt = separate(true_code)
    except ValueError:
        # 잘못된 코드는 0점
        return 0.0
    
    # Level 3: 완전 매칭
    if p_ch == t_ch and p_bl == t_bl and p_dt == t_dt:
        return 1.0
    
    # Level 2: 챕터 + 블록 매칭
    if p_ch == t_ch and p_bl == t_bl:
        return 0.8
    
    # Level 1: 챕터만 매칭
    if p_ch == t_ch:
        return 0.4
    
    # 매칭 없음
    return 0.0

def compute_multilabel_hierarchical_score(pred_codes: List[str], true_codes: List[str]) -> float:
    """
    멀티라벨 계층적 부분 점수 계산
    각 예측 코드에 대해 가장 좋은 매칭을 찾아 평균 계산
    """
    if not pred_codes or not true_codes:
        return 0.0
    
    total_score = 0.0
    for pred_code in pred_codes:
        # 각 예측 코드에 대해 실제 코드들 중 가장 좋은 매칭 점수 찾기
        best_match = max(
            hierarchical_match_score(pred_code, true_code) 
            for true_code in true_codes
        )
        total_score += best_match
    
    # 예측 코드 개수로 나누어 평균 점수 반환
    return total_score / len(pred_codes)

def compute_bidirectional_hierarchical_score(pred_codes: List[str], true_codes: List[str]) -> dict:
    """개선된 계층적 점수 계산 (양방향 점수 포함)
    
    Args:
        pred_codes: 예측된 ICD 코드들
        true_codes: 실제 ICD 코드들
    
    Returns:
        dict: 정밀도, 재현율, F1 점수를 포함한 양방향 점수
    """
    if not pred_codes or not true_codes:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'hierarchical_partial': 0.0
        }
    
    # 정밀도: 예측 코드들이 얼마나 정확한가 (pred -> true 방향)
    precision = compute_multilabel_hierarchical_score(pred_codes, true_codes)
    
    # 재현율: 실제 코드들을 얼마나 잘 찾아냈는가 (true -> pred 방향)
    recall = compute_multilabel_hierarchical_score(true_codes, pred_codes)
    
    # F1 점수: 정밀도와 재현율의 조화평균
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    # 기본 계층적 부분 점수
    hierarchical_partial = precision
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'hierarchical_partial': hierarchical_partial
    }

class ICDScore:
    """
    계층적 부분 점수 계산기
    
    멀티라벨 __call__ 인터페이스:
    - input: list[str] or list[list[str]]
    - output: score or list[score]
    """
    
    def __init__(self):
        """계층적 부분 점수만 지원"""
        pass
    
    def validate_inputs(self, predictions, references):
        """
        입력 검증 및 타입 판단
        Returns: (predictions, references, is_single_case)
        """
        if predictions is None or references is None:
            raise ValueError("predictions and references cannot be None")
        
        if not predictions or not references:
            return [], [], True
        
        # 타입 구조 판단
        pred_is_single = all(isinstance(item, str) for item in predictions)
        pred_is_multi = all(isinstance(item, list) for item in predictions)
        
        ref_is_single = all(isinstance(item, str) for item in references)
        ref_is_multi = all(isinstance(item, list) for item in references)
        
        # 일관성 검사
        if not ((pred_is_single and ref_is_single) or (pred_is_multi and ref_is_multi)):
            raise ValueError("predictions and references must have same structure")
        
        # 단일 케이스를 다중 케이스로 변환
        if pred_is_single and ref_is_single:
            return [predictions], [references], True
        else:
            return predictions, references, False
    
    def compute_single_score(self, pred_codes: List[str], true_codes: List[str]) -> float:
        """단일 케이스 점수 계산"""
        return compute_multilabel_hierarchical_score(pred_codes, true_codes)
    
    def __call__(
        self,
        predictions: Union[List[str], List[List[str]]],
        references: Union[List[str], List[List[str]]]
    ) -> Union[float, List[float]]:
        """
        멀티라벨 계층적 부분 점수 계산
        
        Args:
            predictions: 예측 ICD 코드
                - List[str]: 단일 케이스 ['A12.3', 'B45.6']
                - List[List[str]]: 다중 케이스 [['A12.3', 'B45.6'], ['C78.9']]
            references: 실제 ICD 코드 (동일 구조)
        
        Returns:
            Union[float, List[float]]: 계층적 점수 또는 점수 리스트
        """
        try:
            predictions, references, is_single = self.validate_inputs(predictions, references)
            
            # 길이 검사
            if len(predictions) != len(references):
                raise ValueError(f"Length mismatch: {len(predictions)} vs {len(references)}")
            
            # 점수 계산
            scores = []
            for pred, ref in zip(predictions, references):
                single_score = self.compute_single_score(pred, ref)
                scores.append(single_score)
            
            # 반환 타입 결정
            return scores[0] if is_single else scores
            
        except Exception as e:
            # 에러 시 안전한 기본값 반환
            if hasattr(self, '_debug') and self._debug:
                print(f"ICDScore Error: {e}")
            
            # 입력 타입에 따른 기본값
            try:
                if predictions and all(isinstance(item, str) for item in predictions):
                    return 0.0  # 단일 케이스
                elif predictions and all(isinstance(item, list) for item in predictions):
                    return [0.0] * len(predictions)  # 다중 케이스
                else:
                    return 0.0
            except:
                return 0.0
    
    def set_debug(self, debug: bool = True):
        """디버그 모드"""
        self._debug = debug
        return self
    
    def _get_detailed_scores(
        self,
        predictions: Union[List[str], List[List[str]]],
        references: Union[List[str], List[List[str]]]
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """내부용 - 개선된 계층적 점수 계산 (양방향 점수 포함)
        
        이 메서드는 내부적으로 양방향 점수를 계산하지만,
        외부 사용자에게는 계층적 점수만 노출.
        """
        try:
            predictions, references, is_single = self.Validate_inputs(predictions, references)
            
            # 길이 검사
            if len(predictions) != len(references):
                raise ValueError(f"Length mismatch: {len(predictions)} vs {len(references)}")
            
            # 각 케이스별 상세 점수 계산 (내부적으로만 사용)
            detailed_scores = []
            for pred, ref in zip(predictions, references):
                scores_dict = compute_bidirectional_hierarchical_score(pred, ref)
                detailed_scores.append(scores_dict)
            
            # 반환 타입 결정
            return detailed_scores[0] if is_single else detailed_scores
            
        except Exception as e:
            # 에러 시 안전한 기본값 반환 
            default_score = {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'hierarchical_partial': 0.0
            }
            
            if hasattr(self, '_debug') and self._debug:
                print(f"ICDScore Internal Error: {e}")
            
            try:
                if predictions and all(isinstance(item, str) for item in predictions):
                    return default_score  # 단일 케이스
                elif predictions and all(isinstance(item, list) for item in predictions):
                    return [default_score] * len(predictions)  # 다중 케이스
                else:
                    return default_score
            except:
                return default_score

def icd_hierarchical_metric(pred: List[str], true: List[str]) -> float:
    """
    ICD(2).py 스타일의 단일 리스트 계층적 점수 계산
    """
    scorer = ICDScore()
    return scorer(pred, true)

if __name__ == "__main__":
    print("=" * 70)
    print("수정된 ICDScore - ICD(2).py 구조 기반 (계층적 부분 점수만)")
    print("=" * 70)
    
    # 1. 멀티라벨 __call__ 인터페이스 검증
    print("\n1. 멀티라벨 __call__ 인터페이스 검증:")
    print("-" * 40)
    
    scorer = ICDScore()
    
    # 단일 케이스: list[str] → score
    single_pred = ['A12.3', 'B45.6']
    single_ref = ['A12.4', 'B45.6']  # 부분 매칭
    single_result = scorer(single_pred, single_ref)
    
    print(f"단일 케이스:")
    print(f"  입력 타입: list[str]")
    print(f"  예측: {single_pred}")
    print(f"  실제: {single_ref}")
    print(f"  출력: {single_result:.3f} (타입: {type(single_result).__name__})")
    
    # 다중 케이스: list[list[str]] → list[score]
    multi_pred = [
        ['A12.3', 'B45.6'],  # Case 1
        ['C78.9'],           # Case 2
        ['A12.3', 'D64.9']  # Case 3
    ]
    multi_ref = [
        ['A12.4', 'B45.6'],  # Case 1: 부분 매칭
        ['C78.9'],           # Case 2: 완전 매칭  
        ['F32.1', 'D64.9']  # Case 3: 부분 매칭
    ]
    multi_result = scorer(multi_pred, multi_ref)
    
    print(f"\n다중 케이스:")
    print(f"  입력 타입: list[list[str]]")
    print(f"  케이스 수: {len(multi_pred)}")
    print(f"  출력 타입: {type(multi_result).__name__}")
    print(f"  출력 길이: {len(multi_result)}")
    print(f"  점수들: {[f'{s:.3f}' for s in multi_result]}")
    
    print(f"\n2. 계층별 점수 확인 (ICD(2).py 기반):")
    print("-" * 40)
    
    test_cases = [
        (['A12.3'], ['A12.3'], "완전 매칭 (Level 3)", 1.0),
        (['A12.3'], ['A12.4'], "블록 매칭 (Level 2)", 0.8),
        (['A12.3'], ['A45.6'], "챕터 매칭 (Level 1)", 0.4),
        (['A12.3'], ['B45.6'], "매칭 없음", 0.0),
    ]
    
    for pred, ref, desc, expected in test_cases:
        result = scorer(pred, ref)
        status = "✓" if abs(result - expected) < 0.01 else "✗"
        print(f"  {status} {desc}: {result:.1f} (예상: {expected})")
    
    # 3. 멀티라벨 시나리오 테스트
    print(f"\n3. 멀티라벨 시나리오:")
    print("-" * 30)
    
    # 실제 의료진 진단 시뮬레이션
    doctor_predictions = [
        ['A12.3', 'B45.6', 'C78.9'],  # 3개 진단
        ['D64.9'],                     # 1개 진단
        ['A12.3', 'F32.1']            # 2개 진단
    ]
    
    actual_diagnoses = [
        ['A12.4', 'B45.6', 'C78.8'],  # 부분 매칭들
        ['D64.9'],                     # 완전 매칭
        ['E11.9', 'F32.1']            # 챕터 불일치 + 완전 매칭
    ]
    
    results = scorer(doctor_predictions, actual_diagnoses)
    
    print("의료진 진단 평가:")
    for i, (pred, actual, score) in enumerate(zip(doctor_predictions, actual_diagnoses, results)):
        print(f"  환자 {i+1}: {pred} vs {actual}")
        print(f"          점수: {score:.3f}")
    
    print(f"\n평균 점수: {np.mean(results):.3f}")
    
    # 4. 양방향 점수 계산 확인 (내부적으로만 사용)
    print(f"\n4. 개선된 계층적 점수 계산 (양방향 점수 포함 - 내부용):")
    print("-" * 55)
    
    # 내부 양방향 점수는 계산되지만 최종 출력은 계층적 점수만
    test_pred = ['A12.3', 'B45.6'] 
    test_ref = ['A12.4', 'B45.6']
    
    # 사용자에게는 계층적 점수만 보임
    final_score = scorer(test_pred, test_ref)
    print(f"최종 출력 (계층적 점수만): {final_score:.3f}")
    
    # 내부적으로는 양방향 점수가 계산됨을 보여주기 위해
    internal_scores = scorer._get_detailed_scores(test_pred, test_ref)
    print(f"내부 계산 (참고용):")
    print(f"  - 정밀도: {internal_scores['precision']:.3f}")
    print(f"  - 재현율: {internal_scores['recall']:.3f}")  
    print(f"  - F1 점수: {internal_scores['f1_score']:.3f}")
    print(f"  - 계층적 점수: {internal_scores['hierarchical_partial']:.3f}")
    print(f"→ 최종 계층적 점수({final_score:.3f})")
    
    
    pred_codes = ['A12.3', 'B45.6', 'C78.9']
    true_codes = ['A12.4', 'B45.6', 'C78.8']
    
    compat_score = icd_hierarchical_metric(pred_codes, true_codes)
    print(f"ICD(2).py 스타일 함수: {compat_score:.3f}")
    
    # 7. 에러 처리 검증
    print(f"\n7. 에러 처리:")
    print("-" * 15)
    
    error_cases = [
        ([], ['A12.3'], "빈 예측"),
        (['INVALID'], ['A12.3'], "잘못된 형식"),
        ([['A12.3'], ['B45.6']], [['A12.3']], "길이 불일치")
    ]
    
    for pred, ref, desc in error_cases:
        try:
            result = scorer(pred, ref)
            print(f"  {desc}: {result} (처리됨)")
        except Exception as e:
            print(f"  {desc}: 에러 - {str(e)[:30]}...")
    
    print(f"\n{'=' * 70}")
    print("✓ input: list[str] or list[list[str]]")
    print("✓ output: score or list[score] (계층적 점수만)")
    print("=" * 70)