"""Schemas for database interactions."""

from typing import Literal, override
from pydantic import SecretStr
from pydantic.config import ConfigDict
from pydantic.fields import Field, computed_field
from pydantic.functional_validators import model_validator
from pydantic.main import BaseModel


class QueryParam(BaseModel):
    """Represents a single query parameter for database connection URLs."""

    model_config = ConfigDict(  # pyright: ignore[reportUnannotatedClassAttribute]
        frozen=True
    )
    key: str = Field(
        min_length=1,
        description="The name of the query parameter.",
        examples=["sslmode", "connect_timeout"],
    )
    value: str | int | float | bool = Field(
        description="The value of the query parameter. Must be JSON-serializable and URL-safe."
    )

    @override
    def __hash__(self) -> int:
        return hash((type(self), *self.__dict__.values()))


class DatabaseCredential(BaseModel):
    """Represents the credentials required to connect to a database."""

    model_config = ConfigDict(  # pyright: ignore[reportUnannotatedClassAttribute]
        frozen=True
    )

    db_type: Literal["postgresql", "mysql", "sqlite"] = Field(
        description="Type of database."
    )

    database: str = Field(
        min_length=1,
        description="The name of the database to connect to.",
    )

    username: str | None = Field(
        default=None,
        description="Username for authentication (not required for SQLite).",
    )

    password: SecretStr | None = Field(
        default=None,
        description="Password for authentication (not required for SQLite).",
        json_schema_extra={"format": "password"},
    )

    host: str | None = Field(
        default=None,
        description="Database host (required for PostgreSQL/MySQL).",
        examples=["localhost", "db.example.com"],
    )

    port: int | None = Field(
        default=None,
        gt=0,
        description="Database port (required for PostgreSQL/MySQL).",
        examples=[5432, 3306],
    )

    query: tuple[QueryParam, ...] = Field(
        default_factory=tuple,
        description="Optional query parameters (SSL, timeouts, etc.).",
    )

    @model_validator(mode="after")
    def validate_by_db_type(self):
        if self.db_type in {"postgresql", "mysql"}:
            missing = [
                field for field in ["host", "port"] if getattr(self, field) is None
            ]
            if missing:
                raise ValueError(
                    f"{self.db_type} requires the following fields: {', '.join(missing)}"
                )

        if self.db_type == "sqlite":
            if any([self.host, self.port, self.username, self.password]):
                raise ValueError(
                    "SQLite must not include host, port, username, or password"
                )

        return self

    @computed_field
    @property
    def requires_network(self) -> bool:
        """Computed property instead of a mutated private attribute."""
        return self.db_type in {"postgresql", "mysql"}

    def to_query_dict(self) -> dict[str, str]:
        """Convert query params into dict for SQLAlchemy URL."""
        return {param.key: str(param.value) for param in self.query}

    @override
    def __hash__(self) -> int:
        return hash((type(self), *self.__dict__.values()))


class TableSchema(BaseModel):
    """Represents the schema of a database table."""

    name: str = Field(description="The name of the database table.")
    columns: list[str] = Field(
        description="The list of column names in the database table.",
    )
    column_types: list[dict[str, str]] = Field(
        description="The list of column names along with their data types in the database table.",
    )
    primary_keys: list[str] = Field(
        default_factory=list,
        description="The list of column names that are primary keys in the database table.",
    )
    foreign_keys: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "The list of foreign key relationships in the database table. Each relationship is represented as a dictionary with 'column' and 'references' keys."
        ),
    )


class ColumnRef(BaseModel):
    """Mapping of column name with its table."""

    table: str = Field(description="Name of the database table.")
    column: str = Field(description="Column belonging to database table.")


class Filter(BaseModel):
    """Represents a filter condition to be applied to a column in a SQL query."""

    column: ColumnRef = Field(
        description="The column (with table reference) to filter on."
    )
    operator: Literal["=", "!=", ">", "<", ">=", "<=", "LIKE", "IN"] = Field(
        description="The logical operator to apply to the column and value."
    )
    value: str | int | float | list[str | int | float] = Field(
        description="The value to compare the column against. Use a list for the 'in' operator."
    )

    @model_validator(mode="after")
    def validate_value_for_in(self):
        if self.operator == "IN" and not isinstance(self.value, list):
            raise ValueError("IN operator requires list value")
        return self


class Metric(BaseModel):
    """Represents a mathematical aggregation to be performed on a column in a SQL query."""

    column: ColumnRef = Field(
        description="The column (with table reference) to perform a calculation on."
    )
    aggregation: Literal["sum", "avg", "count", "min", "max", "count_distinct"] = Field(
        description="The aggregation to apply to the metric column."
    )
    alias: str | None = Field(
        default=None,
        description="An optional alias for the resulting metric column in the output.",
    )


class OrderBy(BaseModel):
    """Represents a sorting technique to be applied to a column in a SQL query result."""

    column: ColumnRef | str = Field(
        description="The column (with table reference) OR metric alias used to sort the final result set."
    )
    sort_type: Literal["asc", "desc"] = Field(
        description="The sort order: ascending (asc) or descending (desc)."
    )


class SQLBlueprint(BaseModel):
    """Simplified structured plan for generating a SQL query."""

    dimensions: list[ColumnRef] = Field(
        default_factory=list,
        description="Columns used for grouping and selection (no need for separate group_by).",
    )

    metrics: list[Metric] = Field(
        default_factory=list,
        description="Aggregations to compute.",
    )

    filters: list[Filter] = Field(
        default_factory=list,
        description="Filtering conditions.",
    )

    order_by: list[OrderBy] = Field(
        default_factory=list,
        description="Sorting rules.",
    )

    limit: int = Field(
        default=100,
        description="Maximum rows to return.",
    )

    @model_validator(mode="after")
    def validate_structure(self):
        if not self.dimensions and not self.metrics:
            raise ValueError("At least one dimension or metric required")

        return self
