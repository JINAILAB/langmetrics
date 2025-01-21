from abc import ABC, abstractmethod
from typing import Dict, List

    
class BaseDBManager(ABC):
    @abstractmethod
    def save_evaluation_results(self, evaluation_summary: Dict, results: List[Dict]) -> None:
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass
    