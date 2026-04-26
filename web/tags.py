from enum import Enum


class APITag(str, Enum):
    """Enum for FastAPI endpoint tags with optional metadata.

    FastAPI accepts a list of strings or Enum members for the `tags` argument.
    Using an Enum provides type-safety and makes the tags discoverable in the
    codebase.
    """

    # Core functional areas – expand as needed
    AUTH = "auth"
    SQL = "sql"
    CHART = "chart"
    USER = "user"
    CONFIG = "config"
    METRICS = "metrics"


def openapi_tags() -> list[dict[str, str]]:
    """Generate the OpenAPI `tags` configuration for FastAPI.

    Returns a list of dictionaries understood by FastAPI's `openapi_tags`
    parameter. Each dict contains a ``name`` and an optional ``description``.
    """
    return [
        {
            "name": APITag.AUTH.value,
            "description": "Authentication and security endpoints",
        },
        {
            "name": APITag.SQL.value,
            "description": "SQL generation and validation endpoints",
        },
        {
            "name": APITag.CHART.value,
            "description": "Chart generation and visualization endpoints",
        },
        {
            "name": APITag.USER.value,
            "description": "User profile and preferences endpoints",
        },
        {
            "name": APITag.CONFIG.value,
            "description": "Application configuration endpoints",
        },
        {
            "name": APITag.METRICS.value,
            "description": "Metrics and monitoring endpoints",
        },
    ]
