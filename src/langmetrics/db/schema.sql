-- evaluations 테이블: 평가 세션 정보
CREATE TABLE evaluations (
    evaluation_id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(30) NOT NULL,
    task_type VARCHAR(30) NOT NULL,
    evaluation_method VARCHAR(30) NOT NULL,
    evaluation_model VARCHAR(30) NOT NULL,
    score_type VARCHAR(30) NOT NULL,
    total_samples INTEGER NOT NULL,
    score FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- evaluation_results 테이블: 개별 평가 결과
CREATE TABLE evaluation_results (
    result_id SERIAL PRIMARY KEY,
    evaluation_id INTEGER REFERENCES evaluations(evaluation_id),
    question TEXT NOT NULL,
    student_answer TEXT NOT NULL,
    teacher_answer TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL,
    feedback TEXT,
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- evaluation_metrics 테이블: 세부 평가 지표
CREATE TABLE evaluation_metrics (
    metric_id SERIAL PRIMARY KEY,
    evaluation_id INTEGER REFERENCES evaluations(evaluation_id),
    metric_name VARCHAR(30) NOT NULL,
    metric_value FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
