from __future__ import annotations 
from ..llmtestcase import LLMTestCase, MCQTestCase
from dataclasses import dataclass

from abc import ABC, abstractmethod
from typing import List, Iterator, Union, TypeVar, Generic, Optional, Tuple, Callable
from pathlib import Path
import json
import csv
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets



T = TypeVar('T')

@dataclass
class BaseDataset(ABC, Generic[T]):
    test_cases: List[T]
    
    def __post_init__(self):
        if not isinstance(self.test_cases, list):
            raise TypeError("'test_cases' must be a list")
    
    def __len__(self) -> int:
        return len(self.test_cases)
    
    def __getitem__(self, idx) -> Union[T, List[T]]:
        return self.test_cases[idx]
    
    def __iter__(self) -> Iterator[T]:
        return iter(self.test_cases)
    
    def __str__(self) -> str:
        return str(self.test_cases)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.test_cases})"
    
    def append(self, test_case: T) -> None:
        """리스트의 append와 동일한 동작을 합니다."""
        self.test_cases.append(test_case)
    
    def extend(self, other: Union[List[T], BaseDataset[T]]) -> None:
        """
        리스트나 BaseDatasets 객체를 현재 데이터셋에 확장합니다.
        
        Args:
            other: List[T] 또는 BaseDatasets[T] 타입의 객체
                - List[T]: 테스트 케이스의 리스트
                - BaseDatasets[T]: 다른 데이터셋 객체
                
        Raises:
            TypeError: other가 List[T] 또는 BaseDatasets[T] 타입이 아닌 경우
            TypeError: 리스트의 요소가 올바른 타입이 아닌 경우
        """
        if isinstance(other, BaseDataset):
            if not all(isinstance(item, type(self.test_cases[0])) for item in other.test_cases):
                raise TypeError("All items in the other dataset must match the type of existing test cases")
            self.test_cases.extend(other.test_cases)
        elif isinstance(other, list):
            if not other:
                return
            if not all(isinstance(item, type(self.test_cases[0])) for item in other):
                raise TypeError("All items in the list must match the type of existing test cases")
            self.test_cases.extend(other)
        else:
            raise TypeError(f"Expected List[T] or BaseDatasets[T], got {type(other)}")
        
    def insert(self, index: int, test_case: T) -> None:
        """리스트의 insert와 동일한 동작을 합니다."""
        self.test_cases.insert(index, test_case)
    
    def remove(self, test_case: T) -> None:
        """리스트의 remove와 동일한 동작을 합니다."""
        self.test_cases.remove(test_case)
    
    def pop(self, index: int = -1) -> T:
        """리스트의 pop과 동일한 동작을 합니다."""
        return self.test_cases.pop(index)
    
    def clear(self) -> None:
        """리스트의 clear와 동일한 동작을 합니다."""
        self.test_cases.clear()
    
    def index(self, test_case: T, *args) -> int:
        """리스트의 index와 동일한 동작을 합니다."""
        return self.test_cases.index(test_case, *args)
    
    def count(self, test_case: T) -> int:
        """리스트의 count와 동일한 동작을 합니다."""
        return self.test_cases.count(test_case)
    
    def reverse(self) -> None:
        """리스트의 reverse와 동일한 동작을 합니다."""
        self.test_cases.reverse()
        
    @classmethod
    def from_huggingface_hub(cls, path: str, field_mapping: dict = None, **kwargs) -> 'BaseDataset[T]':
        """
        Hugging Face 데이터셋으로부터 테스트 케이스를 생성합니다.
        
        Args:
            path (str): Hugging Face 데이터셋의 이름
            field_mapping (dict): 데이터셋 필드와 표준 필드 간의 매핑
                예: {
                    'input': 'question',  # 데이터셋의 'question' 필드를 'input'으로 매핑
                    'expected_output': 'answer',  # 데이터셋의 'answer' 필드를 'expected_output'으로 매핑
                    'choices': 'options'  # 데이터셋의 'options' 필드를 'choices'로 매핑
                }
            **kwargs: datasets.load_dataset에 전달할 추가 인자들
            
        Returns:
            BaseDatasets[T]: 생성된 데이터셋 객체
        """
        from datasets import load_dataset
        
        # 기본 필드 매핑 설정
        default_mapping = {
            'input': 'input',
            'expected_output': 'expected_output', 
            'choices': 'choices'
        }
        
        # 사용자가 제공한 매핑으로 기본 매핑 업데이트
        if field_mapping:
            default_mapping.update(field_mapping)
        
        dataset = load_dataset(path, **kwargs)
        
        def map_fields(item):
            # 필드 매핑 적용
            mapped_item = {}
            for target_field, source_field in default_mapping.items():
                if source_field not in item:
                    raise KeyError(f"데이터셋에 '{source_field}' 필드가 없습니다. 올바른 필드 매핑을 확인해주세요.")
                mapped_item[target_field] = item[source_field]
            return mapped_item
        
        # 각 아이템의 필드를 매핑한 후 테스트 케이스로 변환
        test_cases = [cls._convert_to_test_case(map_fields(item)) for item in dataset]
        return cls(test_cases=test_cases)

    
    @classmethod
    def from_json(cls, 
                 file_path: Union[str, Path], 
                 encoding: str = 'utf-8',
                 **kwargs) -> 'BaseDataset[T]':
        """
        JSON 파일로부터 테스트 케이스를 생성합니다.
        
        Args:
            file_path (Union[str, Path]): JSON 파일 경로
            encoding (str): 파일 인코딩 (기본값: 'utf-8')
            **kwargs: json.load에 전달할 추가 인자들
            
        Returns:
            BaseDatasets[T]: 생성된 데이터셋 객체
        
        Raises:
            FileNotFoundError: 파일을 찾을 수 없는 경우
            JSONDecodeError: JSON 파싱에 실패한 경우
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f, **kwargs)
            
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of test cases")
            
        test_cases = [cls._convert_to_test_case(item) for item in data]
        return cls(test_cases=test_cases)
    
    @classmethod
    def from_csv(cls, 
            file_path: Union[str, Path], 
            encoding: str = 'utf-8',
            **kwargs) -> 'BaseDataset[T]':
        """
        CSV 파일로부터 테스트 케이스를 생성합니다.
        
        Args:
            file_path (Union[str, Path]): CSV 파일 경로
            encoding (str): 파일 인코딩 (기본값: 'utf-8')
            **kwargs: csv.DictReader에 전달할 추가 인자들
            
        Returns:
            BaseDatasets[T]: 생성된 데이터셋 객체
            
        Raises:
            FileNotFoundError: 파일을 찾을 수 없는 경우
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        test_cases = []
        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f, **kwargs)
            for row in reader:
                test_cases.append(cls._convert_to_test_case(row))
                
        return cls(test_cases=test_cases)
    
    @classmethod
    def from_dataframe(cls, 
                    df: pd.DataFrame,
                    **kwargs) -> 'BaseDataset[T]':
        """
        pandas DataFrame으로부터 테스트 케이스를 생성합니다.
        
        Args:
            df (pd.DataFrame): 테스트 케이스가 포함된 DataFrame
            **kwargs: DataFrame 처리에 필요한 추가 인자들
            
        Returns:
            BaseDatasets[T]: 생성된 데이터셋 객체
            
        Raises:
            TypeError: DataFrame이 아닌 타입이 입력된 경우
        """
        import pandas as pd
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        test_cases = []
        for _, row in df.iterrows():
            test_cases.append(cls._convert_to_test_case(row.to_dict()))
                
        return cls(test_cases=test_cases)
    
    @classmethod
    @abstractmethod
    def _convert_to_test_case(cls, item: dict) -> T:
        """
        Hugging Face 데이터셋의 아이템을 테스트 케이스로 변환합니다.
        
        Args:
            item (dict): Hugging Face 데이터셋의 단일 아이템
            
        Returns:
            T: 변환된 테스트 케이스
        """
        pass
    
    def filter(self, predicate: Callable[[T], bool]) -> 'BaseDataset[T]':
        """
        지정된 조건에 맞는 테스트 케이스만 포함하는 새로운 데이터셋을 반환합니다.
        
        Args:
            predicate (Callable[[T], bool]): 각 테스트 케이스를 평가할 함수
            
        Returns:
            BaseDatasets[T]: 필터링된 새로운 데이터셋
        """
        filtered_cases = [case for case in self.test_cases if predicate(case)]
        return self.__class__(test_cases=filtered_cases)
    
    def map(self, func: Callable[[T], T]) -> 'BaseDataset[T]':
        """
        각 테스트 케이스를 변환하여 새로운 데이터셋을 생성합니다.
        
        Args:
            func (Callable[[T], T]): 각 테스트 케이스를 변환할 함수
            
        Returns:
            BaseDatasets[T]: 변환된 새로운 데이터셋
            
        Raises:
            TypeError: 변환된 객체가 올바른 타입이 아닌 경우
        """
        mapped_cases = []
        for case in self.test_cases:
            transformed = func(case)
            if not isinstance(transformed, type(case)):
                raise TypeError(f"Mapped item must be of type {type(case)}, got {type(transformed)}")
            mapped_cases.append(transformed)
        return self.__class__(test_cases=mapped_cases)
    
    def to_list(self, 
            dict_format: bool = True,
            include_attrs: Optional[List[str]] = None) -> List[Union[T, dict]]:
        """
        데이터셋의 테스트 케이스들을 리스트 형태로 변환합니다.
        
        Args:
            dict_format (bool): 테스트 케이스를 사전 형태로 변환할지 여부 (기본값: True)
            include_attrs (Optional[List[str]]): 사전에 포함할 속성들의 리스트.
                None인 경우 모든 속성을 포함 (기본값: None)
                
        Returns:
            List[Union[T, dict]]: 테스트 케이스들의 리스트
                - dict_format=True인 경우: 사전 형태의 리스트
                - dict_format=False인 경우: 원래 객체들의 리스트
                
        Examples:
            >>> dataset = BaseDataset(test_cases=[...])
            >>> # 모든 테스트 케이스를 사전 형태로 변환
            >>> dict_list = dataset.to_list()
            >>> # 원본 객체 리스트 반환
            >>> obj_list = dataset.to_list(dict_format=False)
            >>> # 특정 속성만 포함하여 사전 형태로 변환
            >>> filtered_list = dataset.to_list(include_attrs=['id', 'text'])
        """
        if not dict_format:
            return self.test_cases.copy()
            
        result = []
        for case in self.test_cases:
            if hasattr(case, 'to_dict'):
                case_dict = case.to_dict()
            else:
                case_dict = vars(case)
                
            if include_attrs is not None:
                case_dict = {k: v for k, v in case_dict.items() if k in include_attrs}
                
            result.append(case_dict)
            
        return result
    
    
    def to_json(self, 
                file_path: Union[str, Path], 
                encoding: str = 'utf-8',
                ensure_ascii: bool = False,
                **kwargs) -> None:
        """
        데이터셋을 JSON 파일로 저장합니다.
        
        Args:
            file_path (Union[str, Path]): 저장할 파일 경로
            encoding (str): 파일 인코딩 (기본값: 'utf-8')
            ensure_ascii (bool): ASCII 문자만 사용할지 여부 (기본값: False)
            **kwargs: json.dump에 전달할 추가 인자들
        """
        file_path = Path(file_path)
        data = [case.to_dict() if hasattr(case, 'to_dict') else vars(case) 
                for case in self.test_cases]
        
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=2, **kwargs)
            
    def to_csv(self, 
               file_path: Union[str, Path],
               encoding: str = 'utf-8',
               **kwargs) -> None:
        """
        데이터셋을 CSV 파일로 저장합니다.
        
        Args:
            file_path (Union[str, Path]): 저장할 파일 경로
            encoding (str): 파일 인코딩 (기본값: 'utf-8')
            **kwargs: csv.DictWriter에 전달할 추가 인자들
        """
        file_path = Path(file_path)
        data = [case.to_dict() if hasattr(case, 'to_dict') else vars(case) 
                for case in self.test_cases]
        
        if not data:
            raise ValueError("Cannot save empty dataset to CSV")
            
        fieldnames = data[0].keys()
        with open(file_path, 'w', encoding=encoding, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, **kwargs)
            writer.writeheader()
            writer.writerows(data)
            
    def to_dataframe(self) -> 'pd.DataFrame':
        """
        데이터셋을 pandas DataFrame으로 변환합니다.
        
        Returns:
            pd.DataFrame: 변환된 DataFrame
        """
        import pandas as pd
        
        data = [case.to_dict() if hasattr(case, 'to_dict') else vars(case) 
                for case in self.test_cases]
        return pd.DataFrame(data)
    
    def split(self, 
             test_size: float = 0.2, 
             random_state: Optional[int] = None,
             shuffle: bool = True) -> Tuple['BaseDataset[T]', 'BaseDataset[T]']:
        """
        데이터셋을 훈련셋과 테스트셋으로 분할합니다.
        
        Args:
            test_size (float): 테스트셋의 비율 (0과 1 사이, 기본값: 0.2)
            random_state (Optional[int]): 난수 생성을 위한 시드값
            shuffle (bool): 분할 전 데이터를 섞을지 여부 (기본값: True)
            
        Returns:
            Tuple[BaseDatasets[T], BaseDatasets[T]]: (훈련셋, 테스트셋)
            
        Raises:
            ValueError: test_size가 0과 1 사이의 값이 아닌 경우
        """
        if not 0 <= test_size <= 1:
            raise ValueError("test_size must be between 0 and 1")
            
        from sklearn.model_selection import train_test_split
        
        train_cases, test_cases = train_test_split(
            self.test_cases,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle
        )
        
        return (
            self.__class__(test_cases=train_cases),
            self.__class__(test_cases=test_cases)
        )

    def sample(self, 
               n: Optional[int] = None,
               frac: Optional[float] = None,
               random_state: Optional[int] = None) -> 'BaseDataset[T]':
        """
        데이터셋에서 무작위로 샘플을 추출합니다.
        
        Args:
            n (Optional[int]): 추출할 샘플의 개수
            frac (Optional[float]): 추출할 샘플의 비율 (0과 1 사이)
            random_state (Optional[int]): 난수 생성을 위한 시드값
            
        Returns:
            BaseDatasets[T]: 샘플링된 새로운 데이터셋
            
        Raises:
            ValueError: n과 frac이 모두 None이거나 모두 지정된 경우
            ValueError: frac이 0과 1 사이의 값이 아닌 경우
        """
        if (n is None and frac is None) or (n is not None and frac is not None):
            raise ValueError("Exactly one of 'n' or 'frac' must be specified")
            
        import random
        if random_state is not None:
            random.seed(random_state)
            
        if frac is not None:
            if not 0 <= frac <= 1:
                raise ValueError("frac must be between 0 and 1")
            n = int(len(self.test_cases) * frac)
            
        sampled_cases = random.sample(self.test_cases, n)
        return self.__class__(test_cases=sampled_cases)

    def push_to_hub(self,
                   repo_id: str,
                   commit_message: Optional[str] = None,
                   private: bool = False,
                   token: Optional[str] = None,
                   split: Optional[str] = None,
                   **kwargs) -> str:
        """
        데이터셋을 Hugging Face Hub에 업로드합니다.
        
        Args:
            repo_id (str): 'username/dataset-name' 형식의 저장소 ID
            commit_message (Optional[str]): 커밋 메시지 (기본값: 자동 생성)
            private (bool): 비공개 저장소 여부 (기본값: False)
            token (Optional[str]): Hugging Face 토큰 (기본값: 환경변수나 로컬 설정에서 가져옴)
            split (Optional[str]): 데이터셋 분할 이름 (예: 'train', 'test', 'validation')
            **kwargs: datasets.Dataset.push_to_hub에 전달할 추가 인자들
            
        Returns:
            str: 업로드된 저장소의 URL
            
        Raises:
            ImportError: datasets 라이브러리가 설치되지 않은 경우
            ValueError: 잘못된 입력값이 제공된 경우
            RuntimeError: 업로드 실패 시
        """
        try:
            from datasets import Dataset
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "This method requires the 'datasets' package. "
                "Please install it with 'pip install datasets'"
            )
            
        # 데이터를 리스트로 변환
        data = [case.to_dict() if hasattr(case, 'to_dict') else vars(case)
                for case in self.test_cases]
        
        # Dataset 객체 생성
        dataset = Dataset.from_list(data)
        
        # 기본 커밋 메시지 설정
        if commit_message is None:
            commit_message = f"Upload dataset with {len(self.test_cases)} examples"
            
        # repo_id 형식 검증
        if '/' not in repo_id:
            raise ValueError(
                "repo_id must be in the format 'username/dataset-name'"
            )
            
        try:
            # HfApi 객체 생성 및 토큰 설정
            api = HfApi()
            if token:
                api.set_access_token(token)
                
            # 저장소 생성 (없는 경우)
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True
            )
            
            # 데이터셋 업로드
            url = dataset.push_to_hub(
                repo_id,
                split=split,
                private=private,
                commit_message=commit_message,
                token=token,
                **kwargs
            )
            
            return url
            
        except Exception as e:
            raise RuntimeError(f"Failed to push dataset to hub: {str(e)}")
        
    

@dataclass
class LLMDataset(BaseDataset[LLMTestCase]):
    def __post_init__(self):
        super().__post_init__()
        if not all(isinstance(case, LLMTestCase) for case in self.test_cases):
            raise TypeError("All items in 'test_cases' must be LLMTestCase instances")
    
    def append(self, test_case: LLMTestCase) -> None:
        if not isinstance(test_case, LLMTestCase):
            raise TypeError("Appended item must be an LLMTestCase instance")
        super().append(test_case)
    
    
    

@dataclass
class MCQDataset(BaseDataset[MCQTestCase]):
    def __post_init__(self):
        super().__post_init__()
        if not all(isinstance(case, MCQTestCase) for case in self.test_cases):
            raise TypeError("All items in 'test_cases' must be MCQTestCase instances")
    
    def append(self, test_case: MCQTestCase) -> None:
        if not isinstance(test_case, MCQTestCase):
            raise TypeError("Appended item must be an MCQTestCase instance")
        super().append(test_case)
        
    @classmethod
    def _convert_to_test_case(cls, item: dict) -> MCQTestCase:
        """
        Hugging Face 데이터셋의 딕셔너리를 MCQTestCase 인스턴스로 변환합니다.
        
        Args:
            item (dict): 테스트 케이스 데이터를 포함하는 딕셔너리
                필수 키값들:
                - 'input': 입력 질문
                - 'choices': 가능한 답변들의 리스트
                - 'expected_output': 정답
                선택적 키값들:
                - 'output': LLM이 실제로 답변한 답
                - 'reasoning': 답을 생성하기 위한 추론
                
        Returns:
            MCQTestCase: 새로운 테스트 케이스 인스턴스
            
        Raises:
            KeyError: 필수 필드가 없는 경우
            ValueError: 데이터 형식이 유효하지 않은 경우
        """
        # 입력 필드 처리
        if 'input' not in item:
            raise KeyError("데이터셋에 'input' 필드가 반드시 포함되어야 합니다")
        input_field = item['input']

        # 선택지 필드 처리
        if 'choices' not in item:
            raise KeyError("데이터셋에 'choices' 필드가 반드시 포함되어야 합니다")
        choices_field = item['choices']
        
        # 선택지가 문자열 리스트인지 확인
        if not isinstance(choices_field, list):
            raise ValueError("선택지는 리스트 형태여야 합니다")
        choices_field = [str(choice) for choice in choices_field]

        # 기대출력 필드 처리
        if 'expected_output' not in item:
            raise KeyError("데이터셋에 'expected_output' 필드가 반드시 포함되어야 합니다")
        answer_field = str(item['expected_output'])

        # 선택적 필드 처리
        output_field = str(item['output']) if 'output' in item else None
        reasoning_field = str(item['reasoning']) if 'reasoning' in item else None

        # MCQTestCase 인스턴스 생성 및 반환
        return MCQTestCase(
            input=input_field,
            choices=choices_field,
            expected_output=answer_field,
            output=output_field,
            reasoning=reasoning_field
        )
            
    @classmethod
    def from_huggingface_hub(cls, 
                            path: str, 
                            field_mapping: dict = None, 
                            split: str = None,
                            include_output: bool = True,
                            include_reasoning: bool = True,
                            **kwargs) -> 'BaseDataset[T]':
        """
        Hugging Face 데이터셋으로부터 테스트 케이스를 생성합니다.

        Args:
            path (str): Hugging Face 데이터셋의 이름
            field_mapping (dict): 데이터셋 필드와 표준 필드 간의 매핑
                예: {
                    'input': 'question',     # 데이터셋의 'question' 필드를 'input'으로 매핑
                    'expected_output': 'answer',  # 데이터셋의 'answer' 필드를 'expected_output'으로 매핑
                    'choices': 'options',    # 데이터셋의 'options' 필드를 'choices'로 매핑
                    'output': 'output',      # 데이터셋의 'output' 필드를 'output'으로 매핑 (선택)
                    'reasoning': 'reasoning'  # 데이터셋의 'reasoning' 필드를 'reasoning'으로 매핑 (선택)
                }
            split (str, optional): 로드할 특정 split. None일 경우 모든 split을 합칩니다.
            include_output (bool): output 필드를 포함할지 여부
            include_reasoning (bool): reasoning 필드를 포함할지 여부
            **kwargs: datasets.load_dataset에 전달할 추가 인자들

        Returns:
            BaseDatasets[T]: 생성된 데이터셋 객체
        """

        # 기본 필드 매핑 설정
        default_mapping = {
            'input': 'input',
            'expected_output': 'expected_output',
            'choices': 'choices'
        }

        # 선택적 필드 추가
        if include_output:
            default_mapping['output'] = 'output'
        if include_reasoning:
            default_mapping['reasoning'] = 'reasoning'

        # 사용자가 제공한 매핑으로 기본 매핑 업데이트
        if field_mapping:
            default_mapping.update(field_mapping)

        # 데이터셋 로드
        dataset = load_dataset(path, **kwargs)

        def map_fields(item):
            # 필드 매핑 적용
            mapped_item = {}
            for target_field, source_field in default_mapping.items():
                # 필수 필드 체크
                if source_field not in item and target_field in ['input', 'expected_output', 'choices']:
                    raise KeyError(f"데이터셋에 필수 필드 '{source_field}'가 없습니다. 올바른 필드 매핑을 확인해주세요.")
                
                # 선택적 필드 체크
                if source_field in item:
                    mapped_item[target_field] = item[source_field]
                elif target_field in ['output', 'reasoning']:
                    # 선택적 필드가 없을 경우 무시
                    continue
                else:
                    raise KeyError(f"데이터셋에 '{source_field}' 필드가 없습니다. 올바른 필드 매핑을 확인해주세요.")
                    
            return mapped_item

        # DatasetDict 처리
        if isinstance(dataset, DatasetDict):
            if split is not None:
                if split not in dataset:
                    raise KeyError(f"데이터셋에 '{split}' split이 없습니다. 가능한 split: {list(dataset.keys())}")
                dataset = dataset[split]
            else:
                # 모든 split의 데이터를 하나로 합치기
                dataset = concatenate_datasets(list(dataset.values()))
        elif not isinstance(dataset, Dataset):
            raise TypeError("지원하지 않는 데이터셋 타입입니다. Dataset 또는 DatasetDict이어야 합니다.")

        # 각 아이템의 필드를 매핑한 후 테스트 케이스로 변환
        test_cases = [cls._convert_to_test_case(map_fields(item)) for item in dataset]
        
        return cls(test_cases=test_cases)
    
    