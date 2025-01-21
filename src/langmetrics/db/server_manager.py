from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base, Evaluation, EvaluationResult
from .base_manager import BaseDBManager
from typing import List, Dict

class ServerDBManager(BaseDBManager):
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def save_evaluation_results(self, evaluation_summary: Dict, results: List[Dict]) -> None:
        session = self.Session()
        try:
            evaluation = Evaluation(
                dataset_name=evaluation_summary['dataset_name'],
                task_type=evaluation_summary['task_type'],
                evaluation_method=evaluation_summary['evaluation_method'],
                evaluation_model=evaluation_summary['evaluation_model'],
                score_type=evaluation_summary['score_type'],
                total_samples=evaluation_summary['total_samples'],
                score=evaluation_summary['score']
            )
            session.add(evaluation)
            session.flush()
            
            for result in results:
                evaluation_result = EvaluationResult(
                    evaluation_id=evaluation.evaluation_id,
                    question=result['question'],
                    student_answer=result['student_answer'],
                    teacher_answer=result['teacher_answer'],
                    is_correct=result['is_correct'],
                    feedback=result.get('feedback'),
                    confidence_score=result.get('confidence_score')
                )
                session.add(evaluation_result)
            
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def close(self) -> None:
        self.engine.dispose()
