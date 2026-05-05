"""
Application settings loaded via pydantic-settings.

All environment variables are defined here with appropriate types.
"""

from pathlib import Path
from typing import Annotated
from pydantic.functional_validators import AfterValidator
from pydantic.types import SecretStr, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: Annotated[
        str,
        StringConstraints(pattern=r"^gsk_.*"),
        AfterValidator(lambda s: SecretStr(s)),  # pyright: ignore[reportAny]
    ]
    internal_api_key: SecretStr

    model_config = (  # pyright: ignore[reportUnannotatedClassAttribute]
        SettingsConfigDict(
            env_file=Path(".env"),
            extra="ignore",
            env_file_encoding="utf-8",
        )
    )


settings = Settings()  # pyright: ignore[reportCallIssue]
