from __future__ import annotations 
from ..llmtestcase import LLMTestCase
from dataclasses import dataclass, field
import pandas as pd
from typing import List, Iterator, Union, Optional, Tuple, Dict
from langmetrics.result import LLMResult


@dataclass
class LLMDataset:
    test_cases: Union[List[LLMTestCase], LLMTestCase, None] = field(default_factory=list, repr=False)
    _df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        self._validate_and_initialize()

    def _validate_and_initialize(self) -> None:
        """테스트 케이스 유효성 검증 및 DataFrame 초기화"""
        # None인 경우 빈 리스트로 초기화
        if self.test_cases is None:
            self.test_cases = []
        
        # 단일 LLMTestCase인 경우 리스트로 변환
        if isinstance(self.test_cases, LLMTestCase):
            self.test_cases = [self.test_cases]
            
        # 리스트가 아닌 경우 타입 에러
        if not isinstance(self.test_cases, list):
            raise TypeError("'test_cases'는 리스트여야 합니다.")

        # 데이터프레임 초기화
        data = [case.to_dict() for case in self.test_cases]
        self._df = pd.DataFrame(data)
        self.test_cases = None

    def _get_row_as_dict(self, idx: int) -> Dict:
        """특정 인덱스의 행을 딕셔너리로 반환"""
        return self._df.iloc[idx].to_dict()

    def _validate_test_case(self, test_case: LLMTestCase) -> None:
        """단일 테스트 케이스 유효성 검증"""
        if not isinstance(test_case, LLMTestCase):
            raise TypeError("테스트 케이스는 LLMTestCase 인스턴스여야 합니다.")
        
    @property
    def df(self) -> pd.DataFrame:
        """Pandas DataFrame에 대한 getter"""
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        """
        Pandas DataFrame에 대한 setter
        - DataFrame 타입 검증
        - 필수 컬럼 존재 여부 검증
        """
        # Pandas DataFrame 타입 검증
        if not isinstance(value, pd.DataFrame):
            raise TypeError("입력은 pandas.DataFrame 타입이어야 합니다.")
        
        # 필수 컬럼 정의
        required_columns = {
            'input', 'output', 'expected_output', 'context',
            'retrieval_context', 'reasoning', 'choices'
        }
        
        # 현재 DataFrame의 컬럼
        current_columns = set(value.columns)
        
        # 필수 컬럼 존재 여부 검증
        missing_columns = required_columns - current_columns
        if missing_columns:
            raise ValueError(f"다음 필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}")
        
        # 검증 통과 시 DataFrame 저장
        self._df = value

    # Collection 인터페이스 구현
    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx) -> Union[LLMTestCase, List[LLMTestCase]]:
        if isinstance(idx, int):
            # 음수 인덱스 처리
            if idx < 0:
                idx = len(self) + idx
            if not (0 <= idx < len(self)):
                raise IndexError("인덱스가 범위를 벗어났습니다.")
            return LLMTestCase(**self._df.iloc[idx].to_dict())
        elif isinstance(idx, slice):
            df_slice = self._df.iloc[idx]
            return [LLMTestCase(**row) for _, row in df_slice.iterrows()]
        raise TypeError("인덱스는 정수 또는 슬라이스여야 합니다.")

    def __iter__(self) -> Iterator[LLMTestCase]:
        return (self[i] for i in range(len(self)))

    def __str__(self) -> str:
        return str(self._df)

    def __repr__(self) -> str:
        return f"LLMDataset(Pandas DataFrame with {len(self)} rows)"

    # 데이터 조작 메서드
    def append(self, test_case: LLMTestCase) -> None:
        self._validate_test_case(test_case)
        self._df = pd.concat([self._df, pd.DataFrame([test_case.to_dict()])])

    def extend(self, other: Union[List[LLMTestCase], 'LLMDataset']) -> None:
        if isinstance(other, LLMDataset):
            other_df = other.df
        elif isinstance(other, list):
            if not all(isinstance(item, LLMTestCase) for item in other):
                raise TypeError("리스트의 모든 아이템은 LLMTestCase 인스턴스여야 합니다.")
            other_df = pd.DataFrame([item.to_dict() for item in other])
        else:
            raise TypeError(f"리스트 또는 LLMDataset 타입이어야 합니다. (받은 타입: {type(other)})")
        self._df = pd.concat([self._df, other_df])

    def insert(self, index: int, test_case: LLMTestCase) -> None:
        self._validate_test_case(test_case)
        before = self._df.iloc[:index]
        after = self._df.iloc[index:]
        inserted = pd.DataFrame([test_case.to_dict()])
        self._df = pd.concat([before, inserted, after]).reset_index(drop=True)

    def remove(self, test_case: LLMTestCase) -> None:
        idx = self.index(test_case)
        self._df = self._df.drop(self._df.index[idx]).reset_index(drop=True)

    def pop(self, index: int = -1) -> LLMTestCase:
        index = len(self) + index if index < 0 else index
        result = LLMTestCase(**self._df.iloc[index].to_dict())
        self._df = self._df.drop(self._df.index[index]).reset_index(drop=True)
        return result

    def clear(self) -> None:
        self._df = pd.DataFrame(columns=self._df.columns)

    def index(self, test_case: LLMTestCase) -> int:
        target = test_case.to_dict()
        for i, row in self._df.iterrows():
            if row.to_dict() == target:
                return i
        raise ValueError("테스트 케이스를 찾을 수 없습니다.")

    def count(self, test_case: LLMTestCase) -> int:
        target = test_case.to_dict()
        return sum(1 for _, row in self._df.iterrows() if row.to_dict() == target)

    def reverse(self) -> None:
        self._df = self._df.iloc[::-1].reset_index(drop=True)

    # 데이터 변환 메서드
    def to_list(self, dict_format: bool = False, include_attrs: Optional[List[str]] = None) -> List[Union[LLMTestCase, dict]]:
        if include_attrs:
            rows = self._df[include_attrs].to_dict('records')
        else:
            rows = self._df.to_dict('records')
        return rows if dict_format else [LLMTestCase(**row) for row in rows]
    
    def to_dict(self) -> Dict:
        return {col: self._df[col].tolist() for col in self._df.columns}
        
    # 데이터 분할 및 샘플링 메서드
    def split(self, test_size: float = 0.2, random_state: Optional[int] = None, shuffle: bool = True) -> Tuple['LLMDataset', 'LLMDataset']:
        if not 0 <= test_size <= 1:
            raise ValueError("test_size는 0과 1 사이의 값이어야 합니다.")
        from sklearn.model_selection import train_test_split
        cases = self.to_list(dict_format=False)
        train_cases, test_cases = train_test_split(
            cases, test_size=test_size, 
            random_state=random_state, shuffle=shuffle
        )
        return LLMDataset(test_cases=train_cases), LLMDataset(test_cases=test_cases)

    def sample(self, n: Optional[int] = None, frac: Optional[float] = None, random_state: Optional[int] = None) -> 'LLMDataset':
        if (n is None and frac is None) or (n is not None and frac is not None):
            raise ValueError("n 또는 frac 중 정확히 하나를 지정해야 합니다.")
        
        sampled_df = self._df.sample(
            n=n, 
            frac=frac, 
            random_state=random_state
        ).reset_index(drop=True)
        
        result = LLMDataset(test_cases=None)
        result.df = sampled_df
        return result

    def push_to_hub(self, repo_id: str, commit_message: Optional[str] = None, 
                    private: bool = False, token: Optional[str] = None, 
                    split: Optional[str] = None, **kwargs) -> str:
        try:
            from datasets import Dataset
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "이 메서드를 사용하려면 'datasets' 패키지가 필요합니다. "
                "'pip install datasets'로 설치하세요."
            )

        if '/' not in repo_id:
            raise ValueError("repo_id는 'username/dataset-name' 형식이어야 합니다.")
            
        # Pandas DataFrame에서 모든 값이 null인 열 찾기
        null_counts = self._df.isnull().sum()
        total_rows = len(self)
        
        # 모든 값이 null인 열 찾기
        all_null_columns = [col for col, count in null_counts.items() if count == total_rows]
        
        if all_null_columns:
            print(f"다음 열들이 모두 Null이어서 제외됩니다: {', '.join(all_null_columns)}")
            # Null인 열 제거
            cleaned_df = self._df.drop(columns=all_null_columns)
        else:
            cleaned_df = self._df
            
        # Pandas DataFrame을 딕셔너리 리스트로 변환
        cleaned_data = cleaned_df.to_dict('records')
        
        # Dataset 생성
        dataset = Dataset.from_list(cleaned_data)
        commit_message = commit_message or f"{len(self)}개의 예시가 포함된 데이터셋 업로드"

        api = HfApi()
        if token:
            api.set_access_token(token)
            
        api.create_repo(repo_id=repo_id, repo_type="dataset", 
                    private=private, exist_ok=True)
            
        return dataset.push_to_hub(
            repo_id,
            split=split,
            private=private,
            commit_message=commit_message,
            token=token,
            **kwargs
        )

    @classmethod
    def from_huggingface_hub(cls, path: str, field_mapping: dict = None, **kwargs) -> 'LLMDataset':
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "이 메서드를 사용하려면 'datasets' 패키지가 필요합니다. "
                "'pip install datasets'로 설치하세요."
            )
        
        # 기본 필드 매핑 설정
        default_mapping = {
            'input': 'input',
            'output': 'output',
            'expected_output': 'expected_output', 
            'context': 'context',
            'retrieval_context': 'retrieval_context',
            'reasoning': 'reasoning',
            'choices': 'choices',
        }
        
        # 사용자가 제공한 매핑으로 기본 매핑 업데이트
        if field_mapping:
            default_mapping.update(field_mapping)
        
        # 데이터셋 로드
        dataset = load_dataset(path, **kwargs)
        
        # 데이터셋이 DatasetDict인 경우 첫 번째 split 사용
        if hasattr(dataset, 'keys'):
            first_split = list(dataset.keys())[0]
            dataset = dataset[first_split]
        
        # 필드 매핑 및 데이터 변환
        data_dicts = []
        available_fields = dataset.features.keys()
        
        for item in dataset:
            mapped_item = {}
            for target_field, source_field in default_mapping.items():
                # 소스 필드가 있으면 값을 가져오고, 없으면 None 설정
                mapped_item[target_field] = item.get(source_field) if source_field in available_fields else None
            data_dicts.append(mapped_item)
        
        # Pandas DataFrame 생성
        df = pd.DataFrame(data_dicts)
        
        # 빈 인스턴스 생성
        instance = cls(test_cases=None) 
        
        # DataFrame 직접 설정
        instance._df = df
        
        return instance


class ResultDataset(LLMDataset):
    def __init__(self, test_cases: Optional[List[LLMResult]] = None):
        super().__init__(test_cases)

    def _validate_and_initialize(self) -> None:
        """테스트 케이스 유효성 검증 및 DataFrame 초기화"""
        if self.test_cases is None:
            self.test_cases = []
        
        if isinstance(self.test_cases, LLMResult):
            self.test_cases = [self.test_cases]
            
        if not isinstance(self.test_cases, list):
            raise TypeError("'test_cases'는 리스트여야 합니다.")

        data = [case.to_dict() for case in self.test_cases]
        self._df = pd.DataFrame(data)
        self.test_cases = None

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("입력은 pandas.DataFrame 타입이어야 합니다.")
        
        required_columns = {
            'input', 'student_answer', 
            'teacher_answer', 'expected_output', 
            'context', 'retrieval_context', 
            'reasoning', 'choices',
            'score', 'metadata'
        }
        
        current_columns = set(value.columns)
        missing_columns = required_columns - current_columns
        
        if missing_columns:
            raise ValueError(f"다음 필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}")
        
        self._df = value

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0:
                idx = len(self) + idx
            if not (0 <= idx < len(self)):
                raise IndexError("인덱스가 범위를 벗어났습니다.")
            row_dict = self._df.iloc[idx].to_dict()
            return LLMResult(
                input=row_dict['input'],
                student_answer=row_dict['student_answer'],
                teacher_answer=row_dict['teacher_answer'],
                expected_output=row_dict['expected_output'],
                context=row_dict['context'],
                retrieval_context=row_dict['retrieval_context'],
                reasoning=row_dict['reasoning'],
                choices=row_dict['choices'],
                score=row_dict['score'],
                metadata=row_dict['metadata']
            )
        elif isinstance(idx, slice):
            df_slice = self._df.iloc[idx]
            return [LLMResult(
                input=row['input'],
                student_answer=row['student_answer'],
                teacher_answer=row['teacher_answer'],
                expected_output=row['expected_output'],
                context=row['context'],
                retrieval_context=row['retrieval_context'],
                reasoning=row['reasoning'],
                choices=row['choices'],
                score=row['score'],
                metadata=row['metadata']
            ) for _, row in df_slice.iterrows()]
        raise TypeError("인덱스는 정수 또는 슬라이스여야 합니다.")

    # 결과 데이터셋에 특화된 메서드들
    def get_average_score(self) -> float:
        """전체 데이터셋의 평균 점수를 반환"""
        return self._df['score'].mean()

    def get_score_distribution(self) -> Dict[str, int]:
        """점수 분포를 반환"""
        score_counts = self._df['score'].value_counts().to_dict()
        return {str(score): count for score, count in score_counts.items()}

    def filter_by_score_range(self, min_score: float, max_score: float) -> 'ResultDataset':
        """특정 점수 범위의 결과만 필터링"""
        filtered_df = self._df[(self._df['score'] >= min_score) & (self._df['score'] <= max_score)]
        new_dataset = ResultDataset()
        new_dataset.df = filtered_df.reset_index(drop=True)
        return new_dataset

    def get_metadata_summary(self) -> Dict:
        """메타데이터 필드의 요약 통계를 반환"""
        metadata_list = self._df['metadata'].tolist()
        summary = {}
        for metadata in metadata_list:
            if metadata is None:
                continue
            for key, value in metadata.items():
                if key not in summary:
                    summary[key] = []
                summary[key].append(value)
        return {key: list(set(values)) for key, values in summary.items()}
    
    @classmethod
    def from_huggingface_hub(cls, path: str, field_mapping: dict = None, **kwargs) -> 'ResultDataset':
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "이 메서드를 사용하려면 'datasets' 패키지가 필요합니다. "
                "'pip install datasets'로 설치하세요."
            )
        
        # 기본 필드 매핑 설정 (ResultDataset에 맞게 수정)
        default_mapping = {
            'input': 'input',
            'output': 'student_answer',  # student_answer를 output으로 매핑
            'teacher_answer': 'teacher_answer',
            'expected_output': 'expected_output',
            'context': 'context',
            'retrieval_context': 'retrieval_context',
            'reasoning': 'reasoning',
            'choices': 'choices',
            'score': 'score',
            'metadata': 'metadata'
        }
        
        # 사용자가 제공한 매핑으로 기본 매핑 업데이트
        if field_mapping:
            default_mapping.update(field_mapping)
        
        # 데이터셋 로드
        dataset = load_dataset(path, **kwargs)
        
        # 데이터셋이 DatasetDict인 경우 첫 번째 split 사용
        if hasattr(dataset, 'keys'):
            first_split = list(dataset.keys())[0]
            dataset = dataset[first_split]
        
        # 필드 매핑 및 데이터 변환
        data_dicts = []
        available_fields = dataset.features.keys()
        
        for item in dataset:
            mapped_item = {}
            for target_field, source_field in default_mapping.items():
                # 소스 필드가 있으면 값을 가져오고, 없으면 기본값 설정
                value = item.get(source_field) if source_field in available_fields else None
                
                # 특별한 필드들에 대한 기본값 처리
                if value is None:
                    if target_field == 'score':
                        value = 0
                    elif target_field == 'metadata':
                        value = None
                
                mapped_item[target_field] = value
            
            # student_answer를 output으로 변환 (LLMDataset 호환성)
            mapped_item['output'] = mapped_item.pop('student_answer')
            data_dicts.append(mapped_item)
        
        # Pandas DataFrame 생성
        df = pd.DataFrame(data_dicts)
        
        # 빈 인스턴스 생성
        instance = cls(test_cases=None)
        
        # DataFrame 직접 설정
        instance._df = df
        
        return instance