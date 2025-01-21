import duckdb
from datetime import datetime
from typing import List, Dict
from .base_manager import BaseDBManager

class DuckDBManager(BaseDBManager):
    def __init__(self, db_path: str):
        self.conn = duckdb.connect(db_path)
        self._create_tables()
    
    def _check_table_exists(self, table_name: str) -> bool:
        result = self.conn.execute(f"""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' AND name='{table_name}'
        """).fetchone()
        return result is not None

    def _create_tables(self):
        # 테이블이 모두 존재하는지 확인
        tables_exist = all(
            self._check_table_exists(table) 
            for table in ['evaluations', 'evaluation_results', 'evaluation_metrics']
        )
        
        if not tables_exist:
            # 테이블이 하나라도 없다면 모든 테이블을 생성
            # Create evaluations table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id INTEGER PRIMARY KEY,
                    dataset_name VARCHAR(255) NOT NULL,
                    task_type VARCHAR(50) NOT NULL,
                    evaluation_method VARCHAR(100) NOT NULL,
                    evaluation_model VARCHAR(100) NOT NULL,
                    score_type VARCHAR(50) NOT NULL,
                    total_samples INTEGER NOT NULL,
                    score DOUBLE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create evaluation_results table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    result_id INTEGER PRIMARY KEY,
                    evaluation_id INTEGER,
                    question TEXT NOT NULL,
                    student_answer TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (evaluation_id) REFERENCES evaluations(evaluation_id)
                )
            """)
            
            # Create evaluation_metrics table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_metrics (
                    metric_id INTEGER PRIMARY KEY,
                    evaluation_id INTEGER,
                    metric_name VARCHAR(100) NOT NULL,
                    score DOUBLE NOT NULL,
                    teacher_feedback TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (evaluation_id) REFERENCES evaluations(evaluation_id)
                )
            """)
            
    
    def save_evaluation_results(self, evaluation_summary: Dict, results: List[Dict]) -> None:
        try:
            # Insert evaluation and get the generated id
            self.conn.execute("""
                INSERT INTO evaluations (
                    dataset_name, task_type, evaluation_method, 
                    evaluation_model, score_type, total_samples, score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, 
                (evaluation_summary['dataset_name'],
                 evaluation_summary['task_type'],
                 evaluation_summary['evaluation_method'],
                 evaluation_summary['evaluation_model'],
                 evaluation_summary['score_type'],
                 evaluation_summary['total_samples'],
                 evaluation_summary['score'])
            )
            
            evaluation_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            
            # Insert evaluation results
            for result in results:
                self.conn.execute("""
                    INSERT INTO evaluation_results (
                        evaluation_id, question, student_answer, teacher_answer,
                        is_correct, feedback, confidence_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (evaluation_id,
                     result['question'],
                     result['student_answer'],
                     result.get('teacher_answer'),
                     result['is_correct'],
                     result.get('feedback'),
                     result.get('confidence_score'))
                )
            
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def close(self) -> None:
        self.conn.close()
