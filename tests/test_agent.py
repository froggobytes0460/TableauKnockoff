"""Tests for the agent functionality."""

from typing import Annotated, Any, cast
from unittest.mock import AsyncMock

from langchain_core.runnables import ensure_config
from langgraph.constants import END
from langgraph.types import GraphOutput
import pytest
from pytest_mock import MockFixture
from typing_extensions import Doc

from agent import AgentInput, AgentOutput, run_agent
from agent.deps import DependencyFactory
from agent.llm import LLMChart, LLMPlanner, LLMValidator
from agent.nodes import chart_node, plan_node, sql_validation_node
from agent.states import ChartConfig, GraphState, ValidationResult
from database import DatabaseCredential, DatabaseHandler
from database.schemas import ColumnRef, Metric, OrderBy, SQLBlueprint

ResultPreview = Annotated[
    list[dict[str, str | int]], Doc("Preview of database query result.")
]


class TestAgent:
    """Tests for agent functionality."""

    @pytest.fixture
    def helper_result_preview(
        self,
    ) -> ResultPreview:
        """Create sample preview data for testing."""
        return [
            {"store_id": "OUT0020", "total_quantity": 635},
            {"store_id": "OUT0050", "total_quantity": 579},
            {"store_id": "OUT0004", "total_quantity": 574},
            {"store_id": "OUT0012", "total_quantity": 572},
            {"store_id": "OUT0025", "total_quantity": 568},
            {"store_id": "OUT0003", "total_quantity": 542},
            {"store_id": "OUT0045", "total_quantity": 532},
            {"store_id": "OUT0022", "total_quantity": 513},
            {"store_id": "OUT0030", "total_quantity": 499},
            {"store_id": "OUT0042", "total_quantity": 489},
        ]

    @pytest.fixture
    def helper_sql_blueprint(self) -> SQLBlueprint:
        return SQLBlueprint(
            dimensions=[ColumnRef(table="sales", column="store_id")],
            metrics=[
                Metric(
                    column=ColumnRef(table="sales", column="quantity"),
                    aggregation="sum",
                    alias="total_quantity",
                )
            ],
            order_by=[OrderBy(column="total_quantity", sort_type="desc")],
            limit=10,
        )

    @pytest.fixture
    def helper_sample_state(
        self, helper_result_preview: ResultPreview, helper_sql_blueprint: SQLBlueprint
    ) -> GraphState:
        """Create a sample GraphState for testing."""
        return GraphState(
            question="Top 10 stores by the total quantity of sales.",
            sql_blueprint=helper_sql_blueprint,
            preview=helper_result_preview,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mock_success, mock_sql, mock_chart_config, mock_error",
        [
            # Success case
            (
                True,
                "SELECT 1",
                ChartConfig(
                    chart_type="bar", x="Mock X", y="Mock Y", title="Mock Chart"
                ),
                None,
            ),
            # Failure case
            (False, None, None, "Something went wrong"),
        ],
        ids=["Success Case", "Failiure Case"],
    )
    async def test_run_agent(
        self,
        mocker: MockFixture,
        mock_success: bool,
        mock_sql: str,
        mock_chart_config: ChartConfig,
        mock_error: str,
    ):
        """Tests run_agent for both success and failure scenarios."""
        # Mock
        _ = mocker.patch(
            "agent.graph.sql_chart_builder.ainvoke",
            new_callable=AsyncMock,
            return_value=GraphOutput(
                value=AgentOutput(
                    success=mock_success,
                    sql=mock_sql,
                    chart_config=mock_chart_config,
                    error=mock_error,
                )
            ),
        )

        # Main function
        result = await run_agent(
            input=AgentInput(question="test"),
            db_config=mocker.Mock(spec=DatabaseCredential),
            thread_id="test_thread",
        )

        # Assertions
        assert isinstance(result, AgentOutput)
        assert result.success is mock_success
        assert result.sql == mock_sql
        assert result.error == mock_error

    def test_plan_node_success(
        self,
        helper_sample_state: GraphState,
        helper_sql_blueprint: SQLBlueprint,
        mocker: Annotated[
            MockFixture, Doc("Pytest plugin fixure for mocking abilites.")
        ],
    ):
        """Tests for success of `plan_node`"""
        # Mock
        mock_planner = mocker.Mock(spec=LLMPlanner)
        mock_deps = mocker.Mock(spec=DependencyFactory)
        mock_db = mocker.Mock(spec=DatabaseHandler)

        mock_planner.return_value = helper_sql_blueprint
        _ = mocker.patch("agent.nodes.LLMPlanner", return_value=mock_planner)
        mock_db.get_schema.return_value = []  # pyright: ignore[reportAny]
        mock_deps.get_db.return_value = mock_db  # pyright: ignore[reportAny]

        # Main function
        result = plan_node(
            state=helper_sample_state,
            config=ensure_config(
                {
                    "configurable": {
                        "deps": mock_deps,
                        "db_config": mocker.Mock(spec=DatabaseCredential),
                    }
                }
            ),
        )

        # Assertions
        assert result.goto == "sql_validation"
        mock_planner.assert_called_once()

        result_update = cast(
            dict[str, Any], result.update  # pyright: ignore[reportExplicitAny]
        )
        assert result_update["sql_blueprint"] == helper_sql_blueprint
        assert result_update["db_schema"] == []
        assert result_update["error"] is None

    def test_validation_node_success(
        self,
        helper_sample_state: GraphState,
        helper_result_preview: ResultPreview,
        mocker: Annotated[
            MockFixture, Doc("Pytest plugin fixure for mocking abilites.")
        ],
    ):
        """Tests for success of `validation_node`"""
        # Mock
        mock_validator = mocker.Mock(spec=LLMValidator)
        mock_deps = mocker.Mock(spec=DependencyFactory)
        mock_db = mocker.Mock(spec=DatabaseHandler)

        mock_sql_query = mocker.Mock()
        mock_sql_query.limit.return_value = mock_sql_query  # pyright: ignore[reportAny]
        mock_db.generate_sql_query.return_value = (  # pyright: ignore[reportAny]
            mock_sql_query
        )
        mock_db.compile_sql_query.return_value = "SELECT sales.store_id, sum(sales.quantity) AS total_quantity FROM sales GROUP BY sales.store_id ORDER BY sales.store_id DESC LIMIT 100"  # pyright: ignore[reportAny]
        mock_db.execute_query.return_value = (  # pyright: ignore[reportAny]
            helper_result_preview
        )
        mock_db.engine = mocker.Mock()
        mock_db.engine.dialect = mocker.Mock()
        mock_db.engine.dialect.name = "postgresql"
        mock_deps.get_db.return_value = mock_db  # pyright: ignore[reportAny]

        mock_validator.return_value = ValidationResult(
            is_valid=True,
            confidence_score=0.95,
            error_message=None,
        )
        _ = mocker.patch("agent.nodes.LLMValidator", return_value=mock_validator)

        # Main function
        result = sql_validation_node(
            state=helper_sample_state,
            config=ensure_config(
                {
                    "configurable": {
                        "deps": mock_deps,
                        "db_config": mocker.Mock(spec=DatabaseCredential),
                    }
                }
            ),
        )

        # Assertions
        assert result.goto == "chart"
        mock_validator.assert_called_once()

        result_update = cast(
            dict[str, Any], result.update  # pyright: ignore[reportExplicitAny]
        )
        assert result_update["preview"] == helper_result_preview
        assert (
            result_update["sql"]
            == "SELECT sales.store_id, sum(sales.quantity) AS total_quantity FROM sales GROUP BY sales.store_id ORDER BY sales.store_id DESC LIMIT 100"
        )
        assert result_update["error"] is None

    def test_chart_node_success(
        self,
        helper_sample_state: GraphState,
        mocker: Annotated[
            MockFixture, Doc("Pytest plugin fixure for mocking abilites.")
        ],
    ):
        """Tests for success of `chart_node` when falling back to LLM"""
        # Mock
        mock_chart_llm = mocker.Mock(spec=LLMChart)
        mock_return_chart = ChartConfig(
            chart_type="bar", x="col1", y="col2", title="Test Chart"
        )
        mock_chart_llm.return_value = mock_return_chart
        _ = mocker.patch("agent.nodes.LLMChart", return_value=mock_chart_llm)

        # Main function
        helper_sample_state.preview = [{"col1": 1, "col2": 2, "col3": 3}]
        result = chart_node(
            state=helper_sample_state,
            config=ensure_config({}),
        )

        # Assertions
        assert result.goto == END
        mock_chart_llm.assert_called_once()
        assert (
            cast(dict[str, ChartConfig], result.update)["chart_config"]
            == mock_return_chart
        )

    @pytest.mark.parametrize(
        "preview_data, expected_chart_type, expected_x, expected_y, expected_title_part",
        [
            (
                [
                    {"store_id": "OUT0020", "total_quantity": 635},
                    {"store_id": "OUT0050", "total_quantity": 579},
                    {"store_id": "OUT0004", "total_quantity": 574},
                ],
                "bar",
                "store_id",
                "total_quantity",
                "Total Quantity",
            ),
            (
                [
                    {"month": "2023-08", "revenue": 0},
                    {"month": "2023-09", "revenue": 5000},
                    {"month": "2023-10", "revenue": 3000},
                    {"month": "2023-11", "revenue": 2000},
                    {"month": "2023-12", "revenue": 4000},
                ],
                "area",
                "month",
                "revenue",
                "Revenue",
            ),
            (
                [
                    {"height": 160, "weight": 55},
                    {"height": 170, "weight": 65},
                    {"height": 180, "weight": 75},
                ],
                "scatter",
                "height",
                "weight",
                "Height vs Weight",
            ),
            (
                [
                    {"product": "A", "sales": 100},
                    {"product": "B", "sales": 200},
                    {"product": "C", "sales": 150},
                ],
                "pie",
                "product",
                "sales",
                "Sales",
            ),
        ],
        ids=["bar_chart", "line_chart", "scatter_chart", "pie_chart"],
    )
    def test_chart_node_rule_based_success(
        self,
        helper_sample_state: GraphState,
        preview_data: ResultPreview,
        expected_chart_type: str,
        expected_x: str,
        expected_y: str,
        expected_title_part: str,
        mocker: Annotated[
            MockFixture, Doc("Pytest plugin fixure for mocking abilites.")
        ],
    ):
        """Tests for success of heuristic rules version of `chart_node`"""
        from agent.nodes import chart_node
        from langgraph.constants import END

        # Mock
        mock_chart_llm = mocker.Mock(spec=LLMChart)
        _ = mocker.patch("agent.nodes.LLMChart", return_value=mock_chart_llm)

        helper_sample_state.preview = preview_data

        # Main function
        result = chart_node(
            state=helper_sample_state,
            config=ensure_config({}),
        )

        # Assertions
        mock_chart_llm.assert_not_called()
        assert result.goto == END

        chart_config = cast(dict[str, ChartConfig], result.update)["chart_config"]
        assert chart_config.chart_type == expected_chart_type
        assert chart_config.x == expected_x
        assert chart_config.y == expected_y
        assert expected_title_part in chart_config.title
