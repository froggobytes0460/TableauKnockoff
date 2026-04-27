"""Injects dependency for langgraph product."""

from functools import lru_cache
from typing import Annotated

from typing_extensions import Doc

from database import DatabaseHandler
from database import DatabaseCredential


@lru_cache(maxsize=32)
def _get_db(
    db_creds_json: Annotated[
        str, Doc("Serialized credentials as JSON string to connect to the database.")
    ],
) -> DatabaseHandler:
    """Gets a cached database handler based on the provided credentials, if existing, otherwise creates a new one."""
    db_creds = DatabaseCredential.model_validate_json(db_creds_json)
    return DatabaseHandler.from_credentials(db_creds)


class DependencyFactory:
    def __init__(self):
        pass

    def get_db(
        self,
        db_creds: Annotated[
            DatabaseCredential, Doc("The credentials to connect to the database.")
        ],
    ) -> DatabaseHandler:
        """Wrapper for cached database handler."""
        return _get_db(db_creds.model_dump_json())
