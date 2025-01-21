# db/manager.py
from typing import Union
from .base_manager import BaseDBManager
# from .server_manager import ServerDBManager
from .duckdb_manager import DuckDBManager
from ..config.db_config import DB_Config

def create_db_manager(config: DB_Config) -> BaseDBManager:
    if config.db_type == 'server':
        connection_string = f"{config.db_https}/{config.db_name}"
        # return ServerDBManager(connection_string)
    elif config.db_type == 'duckdb':
        return DuckDBManager(config.db_name)
    else:
        raise ValueError(f"Unsupported database type: {config.db_type}")