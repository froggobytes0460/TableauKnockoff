from enum import Enum


class APITag(str, Enum):
    """Enum for FastAPI endpoint tags with optional metadata.

    FastAPI accepts a list of strings or Enum members for the `tags` argument.
    Using an Enum provides type-safety and makes the tags discoverable in the
    codebase.
    """

    # Core functional areas – expand as needed
    USER = "User"
    API = "API"
    UI = "UI"
    DB = "Database"
    METADATA = "Metadata"
    UI_HANDLER = "UI Handler"


OPENAPI_TAGS = [
    {
        "name": APITag.DB.value,
        "description": "Database connection/execution",
    },
    {
        "name": APITag.API.value,
        "description": "API endpoints",
    },
    {
        "name": APITag.UI.value,
        "description": "UI endpoints",
    },
    {
        "name": APITag.UI_HANDLER.value,
        "description": "Handles UI endpoints",
    },
    {
        "name": APITag.USER.value,
        "description": "User input",
    },
    {
        "name": APITag.METADATA.value,
        "description": "Extra information about site uptime etc.",
    },
]
