"""Tests for Pydantic models."""

import pytest

from agent.states import (
    AgentOutput,
    ChartConfig,
    ValidationResult,
)
from database import DatabaseCredential
from database.schemas import (
    ColumnRef,
    Filter,
    Metric,
    SQLBlueprint,
)


class TestModels:
    """Tests for Pydantic models."""

    def test_validation_result_model(self):
        """Tests creation of `ValidationResult` model"""
        # Valid case
        result = ValidationResult(is_valid=True, confidence_score=0.95)
        assert result.is_valid is True
        assert result.confidence_score == 0.95

        # Invalid case with error message
        result = ValidationResult(
            is_valid=False, confidence_score=0.3, error_message="Invalid SQL"
        )
        assert result.is_valid is False
        assert result.error_message == "Invalid SQL"

        # Should raise error if invalid without error message
        with pytest.raises(ValueError):
            _ = ValidationResult(is_valid=False, confidence_score=0.3)

    def test_chart_config_model(self):
        """Tests creation of `ChartConfig` model"""
        # Valid bar chart
        config = ChartConfig(
            chart_type="bar",
            x="category",
            y="value",
            title="Sales by Category",
        )
        assert config.chart_type == "bar"

        # Valid pie chart
        config = ChartConfig(
            chart_type="pie",
            x="category",
            y="value",
            title="Distribution",
        )
        assert config.chart_type == "pie"

        # With color
        config = ChartConfig(
            chart_type="bar",
            x="category",
            y="value",
            color="region",
            title="Sales by Category and Region",
        )
        assert config.color == "region"

        # Should fail without x or y
        with pytest.raises(ValueError):
            _ = ChartConfig(  # pyright: ignore[reportCallIssue]
                chart_type="bar", x="category", title="Invalid"
            )

    def test_agent_output_model(self):
        """Tests creation of `AgentOutput` model"""
        # Success case
        output = AgentOutput(
            success=True,
            sql="SELECT * FROM sales",
            chart_config=ChartConfig(
                chart_type="bar",
                x="month",
                y="sales",
                title="Sales",
            ),
        )
        assert output.success is True

        # Failure case
        output = AgentOutput(success=False, error="Query failed")
        assert output.success is False
        assert output.error == "Query failed"

        # Should fail: success without sql/chart_config
        with pytest.raises(ValueError):
            _ = AgentOutput(success=True)

        # Should fail: success with error
        with pytest.raises(ValueError):
            _ = AgentOutput(
                success=True,
                sql="SELECT 1",
                chart_config=ChartConfig(chart_type="bar", x="a", y="b", title="t"),
                error="This should not be here",
            )

    def test_sql_blueprint_model(self):
        """Tests creation of `SQLBlueprint` model"""
        # Valid blueprint with dimensions and metrics
        blueprint = SQLBlueprint(
            dimensions=[ColumnRef(table="sales", column="month")],
            metrics=[
                Metric(
                    column=ColumnRef(table="sales", column="amount"),
                    aggregation="sum",
                    alias="total",
                )
            ],
        )
        assert len(blueprint.dimensions) == 1
        assert len(blueprint.metrics) == 1

        # Should fail without dimensions and metrics
        with pytest.raises(ValueError):
            _ = SQLBlueprint()

        # With filters
        blueprint = SQLBlueprint(
            dimensions=[ColumnRef(table="sales", column="region")],
            filters=[
                Filter(
                    column=ColumnRef(table="sales", column="amount"),
                    operator=">",
                    value=100,
                )
            ],
        )
        assert len(blueprint.filters) == 1

    def test_database_credential_model(self):
        """Tests creation of `DatabaseCredential` model"""
        # SQLite (no network required)
        cred = DatabaseCredential(db_type="sqlite", database=":memory:")
        assert cred.requires_network is False

        # PostgreSQL (network required)
        cred = DatabaseCredential(
            db_type="postgresql",
            database="test_db",
            username="user",
            password="pass",  # pyright: ignore[reportArgumentType]
            host="localhost",
            port=5432,
        )
        assert cred.requires_network is True

        # Should fail: PostgreSQL without host/port
        with pytest.raises(ValueError):
            _ = DatabaseCredential(
                db_type="postgresql",
                database="test_db",
            )

        # Should fail: SQLite with network params
        with pytest.raises(ValueError):
            _ = DatabaseCredential(
                db_type="sqlite",
                database=":memory:",
                host="localhost",
            )
