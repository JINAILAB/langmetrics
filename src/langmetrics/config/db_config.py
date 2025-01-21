from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class DB_Config():
    db_type : Literal['duckdb', 'server']
    db_name : str
    db_https : Optional[str] = None