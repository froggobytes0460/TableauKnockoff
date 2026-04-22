"""
The module to allow LLM interaction with the agent.
"""

import os
from pathlib import Path
from string.templatelib import Interpolation, Template
from typing import Annotated, Any, cast

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_groq import ChatGroq
from prettytable import PrettyTable, TableStyle
from typing_extensions import Doc

from agent.prompts import (
    CHART_CONFIG_PROMPT,
    PLANNER_PROMPT,
    PLANNER_RETRY_PROMPT,
    VALIDATOR_PROMPT,
)
from agent.states import ChartConfig, ValidationResult
from database.schemas import SQLBlueprint, TableSchema

if not (path := Path(".env").resolve()).exists():
    raise FileNotFoundError(f"Missing .env file at path: {path}")

_ = load_dotenv(path)

MODEL_ID = "openai/gpt-oss-120b"


class BaseLLM:
    """Base LLM class to be inherited by specific LLM implementations."""

    def __init__(
        self,
        api_key: Annotated[
            str | None, Doc(f"The API key for accessing Groq API.")
        ] = None,
    ):
        resolved_api_key: str | None = api_key if api_key else os.getenv("GROQ_API_KEY")
        self.llm: ChatGroq = ChatGroq(
            model=MODEL_ID,
            api_key=resolved_api_key,  # pyright: ignore[reportArgumentType]
            temperature=0.0,
            max_retries=2,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    @staticmethod
    def format_tabular_data(
        data: Annotated[
            list[dict[str, Any]],  # pyright: ignore[reportExplicitAny]
            Doc("The tabular data to be formatted into a markdown table string."),
        ],
    ) -> str:
        if not data:
            return ""

        table = PrettyTable()
        table.field_names = list(data[0].keys())

        for entry in data:
            table.add_row(list(entry.values()))

        table.set_style(TableStyle.MARKDOWN)
        return str(table.get_string())  # pyright: ignore[reportUnknownMemberType]

    @staticmethod
    def format_schema(
        schema: Annotated[list[TableSchema], Doc("The schema to format.")],
    ) -> str:
        def schema_processor(template: Template) -> str:
            """Processes template items using structural pattern matching."""
            parts: list[str] = []
            for item in template:
                match item:
                    case Interpolation(
                        value=(
                            col_name,  # pyright: ignore[reportAny, reportUnusedVariable]
                            str(col_type),
                        )
                    ):
                        role = {"numeric": "metric", "time": "time"}.get(
                            col_type, "dimension"
                        )
                        parts.append(f"({col_type}, {role})")

                    case Interpolation(value=val):  # pyright: ignore[reportAny]
                        processed_val = (
                            f'"{val}"'
                            if "-" in str(val)  # pyright: ignore[reportAny]
                            or "@" in str(val)  # pyright: ignore[reportAny]
                            else str(val)  # pyright: ignore[reportAny]
                        )
                        parts.append(processed_val)

                    case str(text):
                        parts.append(text)

            return "".join(parts)

        lines: list[str] = []
        for table in schema:
            type_map = {c["name"]: c["type"] for c in table.column_types}

            line = schema_processor(t"{table.name}: [{
                ', '.join(
                    schema_processor(t'{col} {(col, type_map.get(col, "unknown"))}')
                    for col in table.columns
                )
            }]")

            if table.primary_keys:
                line += schema_processor(t" | PK: ({', '.join(table.primary_keys)})")

            if table.foreign_keys:
                line += schema_processor(t" | FK: ({', '.join(
                    schema_processor(t'{fk["column"]} -> {fk["references"]}')
                    for fk in table.foreign_keys
                )})")

            lines.append(line)

        return "\n".join(lines)


class LLMPlanner(BaseLLM):
    """The LLM planner to generate SQL blueprint from the question and database schema."""

    planner_chain: RunnableSerializable[
        dict[str, Any], SQLBlueprint  # pyright: ignore[reportExplicitAny]
    ]
    retry_planner_chain: RunnableSerializable[
        dict[str, Any], SQLBlueprint  # pyright: ignore[reportExplicitAny]
    ]

    def __init__(
        self,
        api_key: Annotated[
            str | None, Doc(f"The API key for accessing Groq API.")
        ] = None,
    ):
        super().__init__(api_key=api_key)
        structured_llm = cast(
            RunnableSerializable[
                Any, SQLBlueprint  # pyright: ignore[reportExplicitAny]
            ],
            self.llm.with_structured_output(  # pyright: ignore[reportUnknownMemberType]
                SQLBlueprint, method="json_schema", strict=True
            ),
        )

        self.planner_chain = PLANNER_PROMPT | structured_llm
        self.retry_planner_chain = PLANNER_RETRY_PROMPT | structured_llm

    def __call__(
        self,
        question: Annotated[str, Doc("The natural language question from the user.")],
        db_schema: Annotated[list[TableSchema], Doc("The database schema.")],
        retry: Annotated[
            bool, Doc("Whether if this is a retry attempt or an initial attempt.")
        ] = False,
        previous_blueprint: Annotated[
            SQLBlueprint | None,
            Doc("The previous SQL blueprint generated by the AI, if existing."),
        ] = None,
        error_message: Annotated[
            str | None,
            Doc("The error message from the previous execution attempt, if existing."),
        ] = None,
        config: Annotated[
            RunnableConfig | None,
            Doc(
                "Optional configuration passed by agent invokation to manage runnables"
            ),
        ] = None,
    ) -> SQLBlueprint:
        """Generate SQL blueprint from the question and database schema."""
        if retry and previous_blueprint:
            return self.retry_planner_chain.invoke(
                input={
                    "db_schema": self.format_schema(db_schema),
                    "user_question": question,
                    "previous_blueprint": previous_blueprint.model_dump(),
                    "error": error_message,
                },
                config=config,
            )
        return self.planner_chain.invoke(
            input={
                "db_schema": self.format_schema(db_schema),
                "user_question": question,
            },
            config=config,
        )


class LLMValidator(BaseLLM):
    """The LLM validator to validate the generated SQL query against the question and database schema."""

    llm_chain: RunnableSerializable[
        dict[str, Any], ValidationResult  # pyright: ignore[reportExplicitAny]
    ]

    def __init__(
        self,
        api_key: Annotated[
            str | None, Doc(f"The API key for accessing Groq API.")
        ] = None,
    ):
        super().__init__(api_key=api_key)
        structured_llm = cast(
            RunnableSerializable[
                Any, ValidationResult  # pyright: ignore[reportExplicitAny]
            ],
            self.llm.with_structured_output(  # pyright: ignore[reportUnknownMemberType]
                ValidationResult,
                method="json_schema",
                strict=True,
            ),
        )
        self.llm_chain = VALIDATOR_PROMPT | structured_llm

    def __call__(
        self,
        question: Annotated[str, Doc("The natural language question from the user.")],
        sql_dialect: Annotated[
            str,
            Doc("The SQL dialect to be used for query generation."),
        ],
        sql_query: Annotated[str, Doc("The compiled SQL query to be validated.")],
        preview: Annotated[
            list[dict[str, Any]],  # pyright: ignore[reportExplicitAny]
            Doc("A preview of the SQL query results."),
        ],
        db_schema: Annotated[list[TableSchema], Doc("The database schema.")],
        config: Annotated[
            RunnableConfig | None,
            Doc(
                "Optional configuration passed by agent invokation to manage runnables"
            ),
        ] = None,
    ) -> ValidationResult:
        """Validate generated SQL query against the question and database schema."""
        return self.llm_chain.invoke(
            input={
                "question": question,
                "sql_dialect": sql_dialect,
                "sql_query": sql_query,
                "preview": self.format_tabular_data(preview),
                "db_schema": self.format_schema(db_schema),
            },
            config=config,
        )


class LLMChart(BaseLLM):
    """The LLM for generating chart configuration based on the SQL query results and the original question."""

    llm_chain: RunnableSerializable[
        dict[str, Any], ChartConfig  # pyright: ignore[reportExplicitAny]
    ]

    def __init__(
        self,
        api_key: Annotated[
            str | None, Doc(f"The API key for accessing Groq API.")
        ] = None,
    ):
        super().__init__(api_key)

        structured_llm = cast(
            RunnableSerializable[
                Any, ChartConfig  # pyright: ignore[reportExplicitAny]
            ],
            self.llm.with_structured_output(  # pyright: ignore[reportUnknownMemberType]
                ChartConfig, method="json_schema", strict=True
            ),
        )

        self.llm_chain = CHART_CONFIG_PROMPT | structured_llm

    def __call__(
        self,
        question: Annotated[str, Doc("The natural language question from the user.")],
        preview: Annotated[
            list[dict[str, Any]],  # pyright: ignore[reportExplicitAny]
            Doc("A preview of the SQL query results."),
        ],
        config: Annotated[
            RunnableConfig | None,
            Doc(
                "Optional configuration passed by agent invokation to manage runnables"
            ),
        ] = None,
    ) -> ChartConfig:
        return self.llm_chain.invoke(
            input={
                "question": question,
                "preview": self.format_tabular_data(preview),
            },
            config=config,
        )
