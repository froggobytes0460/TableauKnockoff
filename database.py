"""Handle database operations for the application."""

from typing import Annotated, Any, Literal, cast
from typing_extensions import Doc
from sqlalchemy import (
    URL,
    ColumnElement,
    Engine,
    Select,
    create_engine,
    select,
    func,
    MetaData,
    Table,
    and_,
)
from agent.state import SQLBlueprint


class DatabaseHandler:
    def __init__(
        self,
        db_url: Annotated[
            str | URL,
            Doc("SQLAlchemy database URL."),
        ] = "sqlite:///default.db",
    ) -> None:
        """
        Do not use this, use the `from_credentials` class method.
        """
        self.engine: Engine = create_engine(db_url)
        self.metadata: MetaData = MetaData()

    def test_connection(self) -> bool:
        """Test connection to database by executing `SELECT 1`."""
        try:
            with self.engine.connect() as conn:
                _ = conn.execute(select(1))
                return True
        except Exception:
            return False

    def get_schema(self) -> dict[str, dict[str, list[str | dict[str, str]]]]:
        self.metadata.reflect(bind=self.engine)
        schema: dict[str, dict[str, list[str | dict[str, str]]]] = {}
        for name, table in self.metadata.tables.items():
            schema[name] = {
                "columns": [f"{c.name} ({c.type})" for c in table.columns],
                "primary_key": [c.name for c in table.primary_key],
                "foreign_keys": [
                    {
                        "column": fk.parent.name,
                        "references": f"{fk.column.table.name}.{fk.column.name}",
                    }
                    for fk in table.foreign_keys
                ],
            }
        return schema

    @classmethod
    def from_credentials(
        cls,
        db_type: Annotated[
            Literal["postgresql", "mysql", "sqlite"], Doc("Different RDBMS.")
        ],
        username: Annotated[str, Doc("Database username.")],
        password: Annotated[str, Doc("Database password.")],
        host: Annotated[str, Doc("Database host.")],
        port: Annotated[int, Doc("Database port.")],
        database: Annotated[str, Doc("Database name.")],
        **query_kwargs: Annotated[  # pyright: ignore[reportAny]
            Any, "Additional query parameters."  # pyright: ignore[reportExplicitAny]
        ],
    ):
        """
        Generates a database handler from given credentials.
        """
        db_url = cls.make_conninfo(
            db_type=db_type,
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
            **query_kwargs,
        )
        return cls(db_url=db_url)

    def generate_sql_query(
        self,
        blueprint: Annotated[
            SQLBlueprint, Doc("AI-generated SQL query generation blueprint.")
        ],
    ) -> Select[tuple[Any, ...]]:  # pyright: ignore[reportExplicitAny]
        """
        Generate a SQLAlchemy Select statement from the `SQLBlueprint`.
        """
        table = Table(blueprint.table, self.metadata, autoload_with=self.engine)
        hue_cols: list[ColumnElement[Any]] = [  # pyright: ignore[reportExplicitAny]
            table.c[d] for d in blueprint.hue_cols
        ]

        metrics: list[ColumnElement[Any]] = []  # pyright: ignore[reportExplicitAny]
        for m in blueprint.metrics:
            col = table.c[m.column]
            match m.aggregation:
                case "sum":
                    agg = func.sum(col).label(m.column)
                case "avg":
                    agg = func.avg(col).label(m.column)
                case "count":
                    agg = func.count(col).label(m.column)
                case "count_distinct":
                    agg = func.count(func.distinct(col)).label(m.column)
                case "min":
                    agg = func.min(col).label(m.column)
                case "max":
                    agg = func.max(col).label(m.column)
            metrics.append(
                cast(ColumnElement[Any], agg)  # pyright: ignore[reportExplicitAny]
            )

        stmt: Select[tuple[Any, ...]] = select(  # pyright: ignore[reportExplicitAny]
            *(hue_cols + metrics)
        )

        filters: list[ColumnElement[bool]] = []
        for f in blueprint.filters:
            col = table.c[f.column]
            match f.operator:
                case "==":
                    filter = cast(ColumnElement[bool], col == f.value)
                case "!=":
                    filter = cast(ColumnElement[bool], col != f.value)
                case ">=":
                    filter = cast(ColumnElement[bool], col >= f.value)
                case "<=":
                    filter = cast(ColumnElement[bool], col <= f.value)
                case ">":
                    filter = cast(ColumnElement[bool], col > f.value)
                case "<":
                    filter = cast(ColumnElement[bool], col < f.value)
                case "like":
                    filter = cast(
                        ColumnElement[bool],
                        col.like(f.value),  # pyright: ignore[reportAny]
                    )
                case "in":
                    filter = cast(
                        ColumnElement[bool],
                        col.in_(f.value),  # pyright: ignore[reportAny]
                    )

            filters.append(filter)

        if filters:
            stmt = stmt.where(and_(*filters))

        if hue_cols and metrics:
            stmt = stmt.group_by(*hue_cols)

        for o in blueprint.order_by:
            col = table.c[o.column]
            match o.sort_type:
                case "desc":
                    stmt = stmt.order_by(col.desc())
                case "asc":
                    stmt = stmt.order_by(col.asc())

        return stmt.limit(blueprint.limit)

    def execute_query(
        self,
        stmt: Annotated[
            Select[tuple[Any, ...]],  # pyright: ignore[reportExplicitAny]
            Doc("The SQL statement to execute."),
        ],
    ) -> list[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
        """Executes the statement and returns a list of dicts for the GraphState."""

        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            return [dict(row) for row in result.mappings()]

    @staticmethod
    def make_conninfo(
        db_type: Annotated[
            Literal["postgresql", "mysql", "sqlite"], "Different RDBMS."
        ],
        database: Annotated[str, "Database name."],
        username: Annotated[str | None, "Database username."] = None,
        password: Annotated[str | None, "Database password."] = None,
        host: Annotated[str | None, "Database host."] = None,
        port: Annotated[int | None, "Database port."] = None,
        **query_kwargs: Annotated[  # pyright: ignore[reportAny]
            Any, "Additional query parameters."  # pyright: ignore[reportExplicitAny]
        ],
    ) -> URL:
        """
        Build a SQLAlchemy connection URL from credentials using SQLAlchemy URL parsing.
        Supports: postgresql, mysql, sqlite.
        """

        if db_type in ["postgresql", "mysql"]:
            return URL.create(
                drivername=db_type,
                username=username,
                password=password,
                host=host,
                port=port,
                database=database,
                query=query_kwargs,
            )
        else:
            return URL.create(
                drivername=db_type,
                database=database,
            )
