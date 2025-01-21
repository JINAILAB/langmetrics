from .manager import create_db_manager
from .duckdb_manager import DuckDBManager
from .server_manager import ServerDBManager

__all__ = ['create_db_manager', 'DuckDBManager', 'ServerDBManager']


# create_db_manager 사용법
# config = DB_Config(
#     db_type='duckdb',  # or 'server'
#     db_name='evaluation.db',
#     db_https='https://example.com'
# )

# db_manager = create_db_manager(config)
# try:
#     db_manager.save_evaluation_results(evaluation_summary, results)
# finally:
#     db_manager.close()