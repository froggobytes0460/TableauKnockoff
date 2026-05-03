"""
Application settings loaded via pydantic-settings.

All environment variables are defined here with appropriate types.
"""

from pathlib import Path
from typing import Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConf(BaseSettings):
    type: Literal["postgresql", "mysql", "sqlite"]
    name: str = Field(min_length=1)
    username: str | None = None
    password: SecretStr | None = None
    host: str | None = None
    port: int | None = Field(default=None, gt=0)

    model_config = (  # pyright: ignore[reportUnannotatedClassAttribute]
        SettingsConfigDict(env_file=Path(".env"), extra="ignore")
    )


class Settings(BaseSettings):
    groq_api_key: SecretStr
    internal_api_key: SecretStr

    db: DatabaseConf = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=DatabaseConf  # pyright: ignore[reportArgumentType]
    )

    model_config = (  # pyright: ignore[reportUnannotatedClassAttribute]
        SettingsConfigDict(
            env_file=Path(".env"),
            extra="ignore",
            env_file_encoding="utf-8",
            env_nested_delimiter="__",
        )
    )


settings = Settings()  # pyright: ignore[reportCallIssue]
