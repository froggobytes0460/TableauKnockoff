"""Injects dependency for langgraph product."""

from functools import lru_cache
from typing import Annotated

from typing_extensions import Doc

from database import DatabaseHandler
from database.schemas import DatabaseCredential


class DependencyFactory:
    def __init__(self):
        pass

    @lru_cache(maxsize=32)
    def get_db(
        self,
        db_creds: Annotated[
            DatabaseCredential, Doc("The credentials to connect to the database.")
        ],
    ) -> DatabaseHandler:
        return DatabaseHandler.from_credentials(db_creds)
