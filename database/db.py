"""Handle database operations for the application."""

import re
from typing import Annotated, Any

from sqlalchemy import types
from sqlalchemy.engine import URL, Engine, create_engine
from sqlalchemy.schema import MetaData, Table
from sqlalchemy.sql.elements import ColumnElement, Label
from sqlalchemy.sql.expression import Join, Select, select, text
from sqlalchemy.sql.functions import func
from sqlalchemy.sql.operators import and_, or_
from typing_extensions import Doc

from database.schemas import DatabaseCredential, SQLBlueprint, TableSchema


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
        db_creds: Annotated[
            DatabaseCredential, Doc("The credentials to connect to the database.")
        ],
    ):
        """
        Generates a database handler from given credentials.
        """
        return cls(
            db_url=URL.create(
                drivername=db_creds.db_type,
                username=db_creds.username,
                password=(
                    db_creds.password.get_secret_value() if db_creds.password else None
                ),
                host=db_creds.host,
                port=db_creds.port,
                database=db_creds.database,
                query=db_creds.to_query_dict(),
            )
        )

    def get_schema(self) -> list[TableSchema]:
        self.metadata.reflect(bind=self.engine)
        schema: list[TableSchema] = []

        keywords = [
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
        pattern_string = rf"(?:^|_| )({'|'.join(keywords)})(?:_|$| |(?=[A-Z]))"
        cat_pattern = re.compile(pattern_string, re.IGNORECASE | re.ASCII)

        for name, table in self.metadata.tables.items():
            column_info: list[dict[str, str]] = []

            for c in table.columns:
                match (c.type, cat_pattern.search(c.name)):
                    case (_, match) if match:
                        col_type = "category"
                    case (
                        types.Integer()
                        | types.BigInteger()
                        | types.SmallInteger()
                        | types.Float()
                        | types.Numeric()
                        | types.DECIMAL(),
                        _,
                    ):
                        col_type = "numeric"
                    case (types.Date() | types.DateTime() | types.Time(), _):
                        col_type = "time"
                    case (types.Enum(), _):
                        col_type = "category"
                    case _:
                        col_type = "text"

                column_info.append({"name": c.name, "type": col_type})

            schema.append(
                TableSchema(
                    name=name,
                    columns=[c.name for c in table.columns],
                    column_types=column_info,
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
    ) -> Annotated[
        Select[Any],  # pyright: ignore[reportExplicitAny]
        Doc("The SQL statement generated from blueprint"),
    ]:
        """Generates a SQLAlchemy SQL statement from given blueprint."""
        if not blueprint.dimensions and not blueprint.metrics:
            raise ValueError("At least one column or metric must be provided")

        tables = self._load_tables(blueprint)
        joined = self._build_joins(tables)

        hue_cols = [
            self._resolve_column(tables, c.table, c.column)
            for c in blueprint.dimensions
        ]
        metrics, metric_map = self._build_metrics(blueprint, tables)

        if not blueprint.dimensions and not metrics:
            raise ValueError("Query must include at least one column or metric")

        stmt = select(*(hue_cols + metrics)).select_from(joined)

        stmt = self._apply_filters(stmt, blueprint, tables)

        if blueprint.metrics and hue_cols:
            stmt = stmt.group_by(*hue_cols)

        stmt = self._apply_order_by(stmt, blueprint, tables, metric_map)

        return stmt.limit(blueprint.limit)

    def _load_tables(
        self,
        blueprint: Annotated[
            SQLBlueprint, Doc("AI-generated SQL query generation blueprint.")
        ],
    ) -> Annotated[dict[str, Table], Doc("The mapping of tables to their names.")]:
        """Loads tables from SQL blueprint."""
        table_names = (
            {d.table for d in blueprint.dimensions}
            | {f.column.table for f in blueprint.filters}
            | {
                o.column.table
                for o in blueprint.order_by
                if not isinstance(o.column, str)
            }
            | {m.column.table for m in blueprint.metrics}
        )
        return {
            name: Table(name, self.metadata, autoload_with=self.engine)
            for name in table_names
        }

    def _resolve_column(
        self,
        tables: Annotated[
            dict[str, Table], Doc("The mapping of tables to their names.")
        ],
        table: Annotated[str, Doc("Name of table")],
        column: Annotated[str, Doc("Name of column belonging to table")],
    ) -> Annotated[
        ColumnElement[Any],  # pyright: ignore[reportExplicitAny]
        Doc("The requested column."),
    ]:
        """Tries to get the column from the given table."""
        if table not in tables:
            raise ValueError(f"Table '{table}' not loaded")
        if column not in tables[table].c:
            raise ValueError(f"Column '{column}' not found in table '{table}'")
        return tables[table].c[column]

    def _build_joins(
        self,
        tables: Annotated[
            dict[str, Table], Doc("The mapping of tables to their names.")
        ],
    ) -> Annotated[Table | Join, Doc("The final joined tables.")]:
        """Builds joins on the tables given."""
        table_list = list(tables.values())
        joined = table_list[0]
        used = {joined.name}

        for t in table_list[1:]:
            for base in table_list:
                for fk in base.foreign_keys:
                    ref = fk.column.table
                    if base.name in used and ref.name == t.name:
                        joined = joined.join(t, fk.parent == fk.column)
                        used.add(t.name)
                        break
                    if ref.name in used and base.name == t.name:
                        joined = joined.join(t, fk.column == fk.parent)
                        used.add(base.name)
                        break
                else:
                    continue
                break
            else:
                raise ValueError(f"No FK relationship found to join table {t.name}")

        return joined

    def _build_metrics(
        self,
        blueprint: Annotated[
            SQLBlueprint, Doc("AI-generated SQL query generation blueprint.")
        ],
        tables: Annotated[
            dict[str, Table], Doc("The mapping of tables to their names.")
        ],
    ) -> tuple[
        Annotated[list[Label[int]], Doc("The list of metrics inferred.")],
        Annotated[
            dict[str, Label[int]], Doc("The mapping of metrics to its name/alias.")
        ],
    ]:
        """Does mathematical aggregation on the columns based on blueprint."""
        metrics: list[Label[int]] = []
        metric_map: dict[str, Label[int]] = {}

        for m in blueprint.metrics:
            col = self._resolve_column(
                tables,
                m.column.table,
                m.column.column,
            )

            label = m.alias or m.column.column

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

        return metrics, metric_map

    def _apply_filters(
        self,
        stmt: Annotated[
            Select[Any],  # pyright: ignore[reportExplicitAny]
            Doc("The SQL statement being generated."),
        ],
        blueprint: Annotated[
            SQLBlueprint, Doc("AI-generated SQL query generation blueprint.")
        ],
        tables: Annotated[
            dict[str, Table], Doc("The mapping of tables to their names.")
        ],
    ) -> Annotated[
        Select[Any],  # pyright: ignore[reportExplicitAny]
        Doc("The SQL statement with added filters."),
    ]:
        """Updates SQL statement to include filters based on given blueprint."""
        filters: list[ColumnElement[bool]] = []

        for f in blueprint.filters:
            col = self._resolve_column(
                tables,
                f.column.table,
                f.column.column,
            )

            match f.operator:
                case "=":
                    filters.append(col == f.value)
                case "!=":
                    filters.append(col != f.value)
                case ">=":
                    filters.append(col >= f.value)
                case "<=":
                    filters.append(col <= f.value)
                case ">":
                    filters.append(col > f.value)
                case "<":
                    filters.append(col < f.value)
                case "LIKE":
                    if isinstance(f.value, list):
                        filters.append(or_(*[col.like(f"%{v}%") for v in f.value]))
                    else:
                        filters.append(col.like(f"%{f.value}%"))
                case "IN":
                    if not isinstance(f.value, list):
                        raise ValueError("IN operator requires list value")
                    filters.append(col.in_(f.value))

        if not filters:
            return stmt
        if len(filters) == 1:
            return stmt.where(filters[0])
        return stmt.where(and_(*filters))

    def _apply_order_by(
        self,
        stmt: Annotated[
            Select[Any],  # pyright: ignore[reportExplicitAny]
            Doc("The SQL statement being generated."),
        ],
        blueprint: Annotated[
            SQLBlueprint, Doc("AI-generated SQL query generation blueprint.")
        ],
        tables: Annotated[
            dict[str, Table], Doc("The mapping of tables to their names.")
        ],
        metric_map: dict[str, Label[int]],
    ) -> Annotated[
        Select[Any],  # pyright: ignore[reportExplicitAny]
        Doc("The SQL statement with added ordering."),
    ]:
        """Updates SQL statement to include result ordering based on given blueprint."""
        for o in blueprint.order_by:
            if isinstance(o.column, str) and o.column in metric_map:
                col = metric_map[o.column]
            else:
                col = self._resolve_column(
                    tables,
                    o.column.table,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
                    o.column.column,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
                )

            stmt = stmt.order_by(col.desc() if o.sort_type == "desc" else col.asc())

        return stmt

    def execute_query(
        self,
        stmt: Annotated[
            Select[Any] | str,  # pyright: ignore[reportExplicitAny]
            Doc("The SQL statement to execute."),
        ],
    ) -> Annotated[
        list[dict[str, Any]],  # pyright: ignore[reportExplicitAny]
        Doc("The results of SQL query as a parsed dictionary."),
    ]:
        """Executes the statement and returns a list of dicts for the GraphState."""

        with self.engine.connect() as conn:
            result = conn.execute(text(stmt) if isinstance(stmt, str) else stmt)
            return [dict(row) for row in result.mappings()]

    def compile_sql_query(
        self,
        stmt: Annotated[
            Select[Any],  # pyright: ignore[reportExplicitAny]
            Doc("The SQL statement to compile."),
        ],
    ) -> Annotated[str, Doc("The SQL query in text-form.")]:
        """Compiles the SQLAlchemy statement to a raw SQL string."""
        return stmt.compile(self.engine, compile_kwargs={"literal_binds": True}).string
