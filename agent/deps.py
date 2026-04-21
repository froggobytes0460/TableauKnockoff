"""Injects dependency for langgraph product."""

from functools import lru_cache

from database import DatabaseHandler
from database.schemas import DatabaseCredential


class DependencyFactory:
    def __init__(self):
        pass

    @lru_cache(maxsize=32)
    @staticmethod
    def get_db(db_creds: DatabaseCredential) -> DatabaseHandler:
        return DatabaseHandler.from_credentials(db_creds)
