"""Handle database operations for the application."""

from typing import Annotated, Any, Literal
from typing_extensions import Doc
from sqlalchemy import (
    URL,
    ColumnElement,
    Date,
    DateTime,
    Engine,
    Float,
    Integer,
    Numeric,
    Select,
    Time,
    create_engine,
    select,
    func,
    MetaData,
    Table,
    and_,
)
from agent.schemas import SQLBlueprint, TableSchema


class DatabaseHandler:
    def __init__(
        self,
        db_url: Annotated[URL, Doc("SQLAlchemy database URL.")],
    ) -> None:
        """
        Do not use this, use the `from_credentials` class method.
        """
        self.engine: Engine = create_engine(db_url)
        self.metadata: MetaData = MetaData()

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
            Any,  # pyright: ignore[reportExplicitAny]
            Doc("Additional query parameters."),
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

    def test_connection(self) -> bool:
        """Test connection to database by executing `SELECT 1`."""
        try:
            with self.engine.connect() as conn:
                _ = conn.execute(select(1))
                return True
        except Exception:
            return False

    def get_schema(self) -> list[TableSchema]:
        self.metadata.reflect(bind=self.engine)

        schema: list[TableSchema] = []

        for name, table in self.metadata.tables.items():

            schema.append(
                TableSchema(
                    name=name,
                    columns=[c.name for c in table.columns],
                    column_types=[
                        {
                            "name": c.name,
                            "type": (
                                "numeric"
                                if isinstance(c.type, (Integer, Float, Numeric))
                                else (
                                    "time"
                                    if isinstance(c.type, (Date, DateTime, Time))
                                    else (
                                        "category"
                                        if any(
                                            k in c.name.lower()
                                            for k in [
                                                "id",
                                                "type",
                                                "category",
                                                "city",
                                                "name",
                                                "tier",
                                                "region",
                                                "country",
                                                "status",
                                                "segment",
                                            ]
                                        )
                                        else "text"
                                    )
                                )
                            ),
                        }
                        for c in table.columns
                    ],
                    primary_keys=[c.name for c in table.primary_key],
                    foreign_keys=[
                        {
                            "column": fk.parent.name,
                            "references": f"{fk.column.table.name}.{fk.column.name}",
                        }
                        for fk in table.foreign_keys
                    ],
                )
            )

        return schema

    def generate_sql_query(
        self,
        blueprint: Annotated[
            SQLBlueprint, Doc("AI-generated SQL query generation blueprint.")
        ],
    ) -> Select[tuple[Any, ...]]:  # pyright: ignore[reportExplicitAny]
        """
        Generate a SQLAlchemy Select statement from the `SQLBlueprint`.
        Supports multi-table queries with automatic FK-based joins.
        """

        if not blueprint.tables:
            raise ValueError("At least one table must be provided")

        tables = {
            name: Table(name, self.metadata, autoload_with=self.engine)
            for name in blueprint.tables
        }

        table_list = list(tables.values())

        def _resolve_column(
            col_name: str,
        ) -> ColumnElement[Any]:  # pyright: ignore[reportExplicitAny]
            nonlocal table_list
            for t in table_list:
                if col_name in t.c:
                    return t.c[col_name]
            raise ValueError(f"Column '{col_name}' not found in any table")

        joined = table_list[0]
        used_tables: set[str] = set([joined.name])

        for t in table_list[1:]:
            joined_flag = False

            for base in table_list:
                for fk in base.foreign_keys:
                    ref_table = fk.column.table

                    if base.name in used_tables and ref_table.name == t.name:
                        joined = joined.join(t, fk.parent == fk.column)
                        used_tables.add(t.name)
                        joined_flag = True
                        break

                    if ref_table.name in used_tables and base.name == t.name:
                        joined = joined.join(t, fk.parent == fk.column)
                        used_tables.add(base.name)
                        joined_flag = True
                        break

                if joined_flag:
                    break

            if not joined_flag:
                raise ValueError(f"No FK relationship found to join table {t.name}")

        hue_cols: list[ColumnElement[Any]] = [  # pyright: ignore[reportExplicitAny]
            _resolve_column(c) for c in blueprint.group_by
        ]

        metrics: list[ColumnElement[Any]] = []  # pyright: ignore[reportExplicitAny]
        metric_map = dict[
            str, ColumnElement[Any]  # pyright: ignore[reportExplicitAny]
        ]()

        for m in blueprint.metrics:
            col = _resolve_column(m.column)
            label = m.alias if m.alias else m.column

            match m.aggregation:
                case "sum":
                    agg = func.sum(col).label(label)
                case "avg":
                    agg = func.avg(col).label(label)
                case "count":
                    agg = func.count(col).label(label)
                case "count_distinct":
                    agg = func.count(func.distinct(col)).label(label)
                case "min":
                    agg = func.min(col).label(label)
                case "max":
                    agg = func.max(col).label(label)

            metrics.append(agg)
            metric_map[label] = agg

        if not hue_cols and not metrics:
            raise ValueError("Query must include at least one column or metric")

        stmt: Select[tuple[Any, ...]] = select(  # pyright: ignore[reportExplicitAny]
            *(hue_cols + metrics)
        ).select_from(joined)

        filters: list[ColumnElement[bool]] = []
        for f in blueprint.filters:
            col = _resolve_column(f.column)

            match f.operator:
                case "=":
                    filter_expr = col == f.value
                case "!=":
                    filter_expr = col != f.value
                case ">=":
                    filter_expr = col >= f.value
                case "<=":
                    filter_expr = col <= f.value
                case ">":
                    filter_expr = col > f.value
                case "<":
                    filter_expr = col < f.value
                case "LIKE":
                    value = str(f.value)
                    if "%" not in value:
                        value = f"%{value}%"
                    filter_expr = col.like(value)
                case "IN":
                    if not isinstance(f.value, list):
                        raise ValueError("IN operator requires list value")
                    filter_expr = col.in_(f.value)

            filters.append(filter_expr)

        if filters:
            stmt = stmt.where(and_(*filters))

        if hue_cols:
            stmt = stmt.group_by(*hue_cols)

        # Order by
        for o in blueprint.order_by:
            if o.column in metric_map:
                col = metric_map[o.column]
            else:
                col = _resolve_column(o.column)

            stmt = stmt.order_by(col.desc() if o.sort_type == "desc" else col.asc())

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

    def compile_sql_query(
        self,
        stmt: Annotated[
            Select[tuple[Any, ...]],  # pyright: ignore[reportExplicitAny]
            Doc("The SQL statement to compile."),
        ],
    ) -> str:
        """Compiles the SQLAlchemy statement to a raw SQL string."""
        return stmt.compile(self.engine, compile_kwargs={"literal_binds": True}).string

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
            Any,  # pyright: ignore[reportExplicitAny]
            Doc("Additional query parameters."),
        ],
    ) -> URL:
        """
        Build a SQLAlchemy connection URL from credentials using SQLAlchemy URL parsing. Supports PostgreSQL, MySQL, and SQLite.
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
                query=query_kwargs,
            )
