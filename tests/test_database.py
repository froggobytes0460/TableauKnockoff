"""Tests for database operations."""

from collections.abc import Callable, Generator
from typing import Annotated, Any

from annotated_doc import Doc
import pytest
from sqlalchemy import types
from sqlalchemy.schema import Column, ForeignKey, Table
from sqlalchemy.sql import func

from database import DatabaseCredential, DatabaseHandler
from database.schemas import (
    ColumnRef,
    Filter,
    Metric,
    OrderBy,
    QueryParam,
    SQLBlueprint,
)

QueryResult = Annotated[
    list[dict[str, str | int]], Doc("Preview of database query result.")
]


def _helper(n: int) -> Callable[[QueryResult], bool]:
    def _check_customer_count_in_result(result: QueryResult) -> bool:
        return "customer_count" in result[0] if result else False

    def _check_alice_in_result(result: QueryResult) -> bool:
        return result[0]["name"] == "Alice" if result else False

    def _check_bob_in_result(result: QueryResult) -> bool:
        return result[0]["name"] == "Bob" if result else False

    def _check_alice_total_amount(result: QueryResult) -> bool:
        return any(
            r.get("name") == "Alice" and r.get("total_amount") == 300.0 for r in result
        )

    return {
        2: _check_customer_count_in_result,
        3: _check_alice_in_result,
        4: _check_bob_in_result,
        5: _check_alice_total_amount,
    }[n]


class TestDatabase:
    """Tests for database operations."""

    @pytest.fixture(scope="function")
    def test_handler(self) -> Generator[DatabaseHandler]:
        """Create an in-memory SQLite DatabaseHandler with test tables.

        The fixture returns a ``DatabaseHandler`` ready for use in tests.
        """
        handler = DatabaseHandler.from_credentials(
            db_creds=DatabaseCredential(
                db_type="sqlite",
                database="file:test",
                query=(
                    QueryParam(key="cache", value="shared"),
                    QueryParam(key="mode", value="memory"),
                    QueryParam(key="uri", value="true"),
                ),
            )
        )

        customers = Table(
            "customers",
            handler.metadata,
            Column("id", types.Integer(), primary_key=True, autoincrement=True),
            Column("name", types.String(50), nullable=False, unique=True),
            Column("city", types.String(100), index=True),
            Column("status", types.String(20), server_default="active"),
            Column("created_at", types.Date(), server_default=func.now()),
        )

        _ = Table(
            "orders",
            handler.metadata,
            Column("id", types.Integer(), primary_key=True),
            Column(
                "customer_id",
                types.Integer(),
                ForeignKey(customers.c.id, ondelete="CASCADE"),
                nullable=False,
            ),
            Column("amount", types.Numeric(10, 2)),
            Column(
                "order_date", types.DateTime(timezone=True), server_default=func.now()
            ),
        )

        handler.metadata.create_all(handler.engine)
        yield handler

        handler.engine.dispose()

    @pytest.fixture(scope="function")
    def populate_test_data(self, test_handler: DatabaseHandler) -> None:
        """Populate in-memory database with sample customers and orders.

        This fixture inserts two customers (Alice and Bob) and two orders for Alice.
        It is used by tests that need data in the tables.
        """
        customers_table = test_handler.metadata.tables["customers"]
        orders_table = test_handler.metadata.tables["orders"]
        with test_handler.engine.connect() as conn:
            _ = conn.execute(
                customers_table.insert().values(
                    name="Alice", city="New York", status="active"
                )
            )
            _ = conn.execute(
                customers_table.insert().values(
                    name="Bob", city="London", status="inactive"
                )
            )
            _ = conn.execute(orders_table.insert().values(customer_id=1, amount=100.0))
            _ = conn.execute(orders_table.insert().values(customer_id=1, amount=200.0))
            conn.commit()

    def test_get_schema(self, test_handler: DatabaseHandler):
        """Tests `get_schema` of DatabaseHandler"""
        # Main function
        schema = test_handler.get_schema()

        # Assertions
        assert isinstance(schema, list)
        assert len(schema) == 2

        customers_schema = next(s for s in schema if s.name == "customers")
        orders_schema = next(s for s in schema if s.name == "orders")

        assert customers_schema.name == "customers"
        assert set(customers_schema.columns) == {
            "id",
            "name",
            "city",
            "status",
            "created_at",
        }
        assert customers_schema.primary_keys == ["id"]
        assert customers_schema.foreign_keys == []

        col_types = {c["name"]: c["type"] for c in customers_schema.column_types}
        assert col_types["id"] == "category"
        assert col_types["name"] == "category"
        assert col_types["city"] == "category"
        assert col_types["status"] == "category"
        assert col_types["created_at"] == "time"

        assert orders_schema.name == "orders"
        assert set(orders_schema.columns) == {
            "id",
            "customer_id",
            "amount",
            "order_date",
        }
        assert orders_schema.primary_keys == ["id"]

        assert len(orders_schema.foreign_keys) == 1
        assert orders_schema.foreign_keys[0]["column"] == "customer_id"
        assert orders_schema.foreign_keys[0]["references"] == "customers.id"

        order_col_types = {c["name"]: c["type"] for c in orders_schema.column_types}
        assert order_col_types["id"] == "category"
        assert order_col_types["amount"] == "numeric"
        assert order_col_types["order_date"] == "time"

    @pytest.mark.parametrize(
        "sql_blueprint, expected_count, check_func",
        (
            # Test 1: Simple dimension select (no metrics)
            (
                SQLBlueprint(
                    dimensions=[
                        ColumnRef(table="customers", column="city"),
                        ColumnRef(table="customers", column="status"),
                    ],
                    metrics=[],
                    filters=[],
                    order_by=[],
                    limit=10,
                ),
                2,
                None,
            ),
            # Test 2: Dimension with count metric (group by)
            (
                SQLBlueprint(
                    dimensions=[ColumnRef(table="customers", column="city")],
                    metrics=[
                        Metric(
                            column=ColumnRef(table="customers", column="id"),
                            aggregation="count",
                            alias="customer_count",
                        )
                    ],
                    filters=[],
                    order_by=[],
                    limit=10,
                ),
                2,
                _helper(2),
            ),
            # Test 3: With filter
            (
                SQLBlueprint(
                    dimensions=[ColumnRef(table="customers", column="name")],
                    metrics=[],
                    filters=[
                        Filter(
                            column=ColumnRef(table="customers", column="city"),
                            operator="=",
                            value="New York",
                        )
                    ],
                    order_by=[],
                    limit=10,
                ),
                1,
                _helper(3),
            ),
            # Test 4: With order_by and limit
            (
                SQLBlueprint(
                    dimensions=[ColumnRef(table="customers", column="name")],
                    metrics=[],
                    filters=[],
                    order_by=[
                        OrderBy(
                            column=ColumnRef(table="customers", column="name"),
                            sort_type="desc",
                        )
                    ],
                    limit=1,
                ),
                1,
                _helper(4),
            ),
            # Test 5: Join across tables with aggregation
            (
                SQLBlueprint(
                    dimensions=[ColumnRef(table="customers", column="name")],
                    metrics=[
                        Metric(
                            column=ColumnRef(table="orders", column="amount"),
                            aggregation="sum",
                            alias="total_amount",
                        )
                    ],
                    filters=[],
                    order_by=[],
                    limit=10,
                ),
                1,
                _helper(5),
            ),
            # Test 6: IN operator filter
            (
                SQLBlueprint(
                    dimensions=[ColumnRef(table="customers", column="name")],
                    metrics=[],
                    filters=[
                        Filter(
                            column=ColumnRef(table="customers", column="city"),
                            operator="IN",
                            value=["New York", "London"],
                        )
                    ],
                    order_by=[],
                    limit=10,
                ),
                2,
                None,
            ),
        ),
        ids=[
            "Simple dimension select",
            "Dimension with count metric",
            "With filter",
            "With order_by and limit",
            "Join across tables with aggregation",
            "IN operator filter",
        ],
    )
    @pytest.mark.usefixtures("populate_test_data")
    def test_generate_sql_query(
        self,
        test_handler: DatabaseHandler,
        sql_blueprint: SQLBlueprint,
        expected_count: Annotated[
            int, Doc("Expected number of results from SQL query.")
        ],
        check_func: Annotated[
            Callable[[QueryResult], bool] | None,
            Doc(
                "Optional function which checks additional properties in query result."
            ),
        ],
    ):
        """Tests `generate_sql_query` of DatabaseHandler."""
        # Main function
        stmt = test_handler.generate_sql_query(sql_blueprint)
        sql = test_handler.compile_sql_query(stmt)

        # Assertions
        assert "SELECT" in sql
        assert "customers" in sql
        result = test_handler.execute_query(stmt)
        assert len(result) == expected_count
        if check_func:
            assert check_func(result)
