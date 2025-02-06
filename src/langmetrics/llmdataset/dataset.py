from __future__ import annotations 
from ..llmtestcase import LLMTestCase
from dataclasses import dataclass, field
import polars as pl
from abc import ABC, abstractmethod
from typing import List, Iterator, Union, TypeVar, Generic, Optional, Tuple, Callable, Dict
from pathlib import Path
import json
import csv
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets


@dataclass
class LLMDataset:
    test_cases: Union[List[LLMTestCase], LLMTestCase, None] = field(default_factory=list, repr=False)
    _df: pl.DataFrame = field(init=False, repr=False)

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
        self._df = pl.DataFrame(data)
        self.test_cases = None

    def _get_row_as_dict(self, idx: int) -> Dict:
        """특정 인덱스의 행을 딕셔너리로 반환"""
        row = self._df.row(idx)
        return dict(zip(self._df.columns, row))

    def _validate_test_case(self, test_case: LLMTestCase) -> None:
        """단일 테스트 케이스 유효성 검증"""
        if not isinstance(test_case, LLMTestCase):
            raise TypeError("테스트 케이스는 LLMTestCase 인스턴스여야 합니다.")
        
    @property
    def df(self) -> pl.DataFrame:
        """Polars DataFrame에 대한 getter"""
        return self._df

    @df.setter
    def df(self, value: pl.DataFrame) -> None:
        """
        Polars DataFrame에 대한 setter
        - DataFrame 타입 검증
        - 필수 컬럼 존재 여부 검증
        """
        # Polars DataFrame 타입 검증
        if not isinstance(value, pl.DataFrame):
            raise TypeError("입력은 polars.DataFrame 타입이어야 합니다.")
        
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
        return self._df.height

    def __getitem__(self, idx) -> Union[LLMTestCase, List[LLMTestCase]]:
        if isinstance(idx, int):
            # 음수 인덱스 처리
            if idx < 0:
                idx = len(self) + idx
            if not (0 <= idx < len(self)):
                raise IndexError("인덱스가 범위를 벗어났습니다.")
            return LLMTestCase(**{name: val for name, val in zip(self._df.columns, self._df.row(idx))})
        elif isinstance(idx, slice):
            _df_slice = self._df[idx]
            return [LLMTestCase(**{name: val for name, val in zip(_df_slice.columns, row)}) 
                   for row in _df_slice.iter_rows()]
        raise TypeError("인덱스는 정수 또는 슬라이스여야 합니다.")

    def __iter__(self) -> Iterator[LLMTestCase]:
        return (self[i] for i in range(len(self)))

    def __str__(self) -> str:
        return str(self._df)

    def __repr__(self) -> str:
        return f"LLMDataset(Polars DataFrame with {len(self)} rows)"

    # 데이터 조작 메서드
    def append(self, test_case: LLMTestCase) -> None:
        self._validate_test_case(test_case)
        self._df = self._df.vstack(pl.DataFrame([test_case.to_dict()]))

    def extend(self, other: Union[List[LLMTestCase], 'LLMDataset']) -> None:
        if isinstance(other, LLMDataset):
            other_list = other.to_list(dict_format=True)
        elif isinstance(other, list):
            if not all(isinstance(item, LLMTestCase) for item in other):
                raise TypeError("리스트의 모든 아이템은 LLMTestCase 인스턴스여야 합니다.")
            other_list = [item.to_dict() for item in other]
        else:
            raise TypeError(f"리스트 또는 LLMDataset 타입이어야 합니다. (받은 타입: {type(other)})")
        self._df = self._df.vstack(pl.DataFrame(other_list))

    def insert(self, index: int, test_case: LLMTestCase) -> None:
        self._validate_test_case(test_case)
        self._df = pl.concat([
            self._df[:index],
            pl.DataFrame([test_case.to_dict()]),
            self._df[index:]
        ])

    def remove(self, test_case: LLMTestCase) -> None:
        idx = self.index(test_case)
        self._df = self._df.drop([idx])

    def pop(self, index: int = -1) -> LLMTestCase:
        index = len(self) + index if index < 0 else index
        result = LLMTestCase(**self._get_row_as_dict(index))
        self._df = self._df.drop([index])
        return result

    def clear(self) -> None:
        self._df = pl.DataFrame(schema=self._df.schema)

    def index(self, test_case: LLMTestCase) -> int:
        target = test_case.to_dict()
        for i, row in enumerate(self._df.to_dicts()):
            if row == target:
                return i
        raise ValueError("테스트 케이스를 찾을 수 없습니다.")

    def count(self, test_case: LLMTestCase) -> int:
        target = test_case.to_dict()
        return sum(1 for row in self._df.to_dicts() if row == target)

    def reverse(self) -> None:
        self._df = self._df.reverse()

    # 데이터 변환 메서드
    def to_list(self, dict_format: bool = False, include_attrs: Optional[List[str]] = None) -> List[Union[LLMTestCase, dict]]:
        rows = self._df.to_dicts()
        if include_attrs:
            rows = [{k: v for k, v in row.items() if k in include_attrs} for row in rows]
        return rows if dict_format else [LLMTestCase(**row) for row in rows]
    
    def to_dict(self) -> Dict:
        return self._df.to_dict(as_series=False)
        

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
            
        import random
        if random_state is not None:
            random.seed(random_state)
            
        cases = self.to_list(dict_format=False)
        if frac is not None:
            if not 0 <= frac <= 1:
                raise ValueError("frac은 0과 1 사이의 값이어야 합니다.")
            n = int(len(cases) * frac)
            
        return LLMDataset(test_cases=random.sample(cases, n))
    


    def push_to_hub(self, repo_id: str, commit_message: Optional[str] = None, 
                    private: bool = False, token: Optional[str] = None, 
                    split: Optional[str] = None, **kwargs) -> str:
            try:
                from datasets import Dataset
                from huggingface_hub import HfApi
                import polars as pl
            except ImportError:
                raise ImportError(
                    "이 메서드를 사용하려면 'datasets' 패키지가 필요합니다. "
                    "'pip install datasets'로 설치하세요."
                )

            if '/' not in repo_id:
                raise ValueError("repo_id는 'username/dataset-name' 형식이어야 합니다.")
                
            # Polars DataFrame에서 모든 값이 null인 열 찾기
            null_counts = self._df.null_count()
            total_rows = len(self)
            
            # 모든 값이 null인 열 찾기 (수정된 부분)
            all_null_columns = []
            for col_name, null_count in zip(self._df.columns, null_counts):
                if null_count.item() == total_rows:  # .item()을 사용하여 스칼라 값으로 변환
                    all_null_columns.append(col_name)
            
            if all_null_columns:
                print(f"다음 열들이 모두 Null이어서 제외됩니다: {', '.join(all_null_columns)}")
                # Null인 열 제거
                cleaned_df = self._df.drop(all_null_columns)
            else:
                cleaned_df = self._df
                
            # Polars DataFrame을 딕셔너리 리스트로 변환
            cleaned_data = cleaned_df.to_dicts()
            
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
            import polars as pl
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
        
        # Polars DataFrame 생성
        df = pl.DataFrame(data_dicts)
        
        # 빈 인스턴스 생성
        instance = cls(test_cases=None) 
        
        # DataFrame 직접 설정
        instance._df = df
        
        return instance



