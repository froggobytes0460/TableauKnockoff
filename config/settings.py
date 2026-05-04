"""
Application settings loaded via pydantic-settings.

All environment variables are defined here with appropriate types.
"""

from pathlib import Path
from typing import Annotated, Literal
from pydantic import AfterValidator, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConf(BaseSettings):
    type: Literal["postgresql", "mysql", "sqlite"] | None = None
    name: str | None = None
    username: str | None = None
    password: SecretStr | None = None
    host: str | None = None
    port: int | None = Field(default=None, gt=0)


class Settings(BaseSettings):
    groq_api_key: Annotated[
        str,
        AfterValidator(lambda s: SecretStr(s)),  # pyright: ignore[reportAny]
    ] = Field(pattern=r"^gsk_.*")
    internal_api_key: SecretStr

    db: DatabaseConf = Field(default_factory=DatabaseConf)

    model_config = (  # pyright: ignore[reportUnannotatedClassAttribute]
        SettingsConfigDict(
            env_file=Path(".env"),
            extra="ignore",
            env_file_encoding="utf-8",
            env_nested_delimiter="__",
        )
    )


settings = Settings()  # pyright: ignore[reportCallIssue]
