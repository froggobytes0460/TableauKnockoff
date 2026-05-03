"""Tests for API endpoints using FastAPI TestClient."""

from collections.abc import Generator
from typing import Annotated

from fastapi.testclient import TestClient
import pytest
from pytest_mock import MockFixture
from typing_extensions import Doc

from agent import AgentInput, AgentOutput, run_agent
from agent.states import ChartConfig
from config import settings
from database import DatabaseCredential, DatabaseHandler
from database.schemas import QueryParam
from main import app

APIHeader = Annotated[dict[str, str], Doc("API Header for query request in API.")]


class TestAPI:
    @pytest.fixture
    def test_client(self) -> Generator[TestClient]:
        """Generates TestClient for all tests."""
        with TestClient(app) as client:
            yield client

    @pytest.fixture
    def helper_api_headers(self) -> APIHeader:
        """Generate API headers with test API key."""
        return {"X-API-KEY": "test_api_key"}

    @pytest.fixture(autouse=True)
    def helper_set_env(
        self,
        mocker: MockFixture,
    ):
        """Set environment variables for testing."""
        mocker_api_key = mocker.Mock(spec=settings.internal_api_key)
        mocker_api_key.get_secret_value.return_value = (  # pyright: ignore[reportAny]
            "test_api_key"
        )
        _ = mocker.patch("web.api.settings.internal_api_key", mocker_api_key)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "should_succeed",
        [True, False],
        ids=["Success Case", "Failure Case"],
    )
    async def test_submit_question(
        self,
        test_client: TestClient,
        helper_api_headers: APIHeader,
        mocker: MockFixture,
        should_succeed: bool,
    ):
        """Test submit_question endpoint."""
        # Mock
        mock_agent = mocker.AsyncMock(spec=run_agent)
        if should_succeed:
            mock_agent.return_value = AgentOutput(
                success=True,
                sql="SELECT 1",
                chart_config=ChartConfig(
                    chart_type="bar", x="category", y="value", title="Test Chart"
                ),
            )
        else:
            mock_agent.side_effect = Exception("Mock error")

        _ = mocker.patch("web.api.run_agent", mock_agent)

        # Main endpoint request
        response = test_client.post(
            "/api/submit_question",
            json={
                "agent_input": AgentInput(
                    question="What is the total sales?"
                ).model_dump(),
                "db_config": DatabaseCredential(
                    db_type="sqlite",
                    database="file:test",
                    query=(
                        QueryParam(key="cache", value="shared"),
                        QueryParam(key="mode", value="memory"),
                        QueryParam(key="uri", value="true"),
                    ),
                ).model_dump(),
            },
            headers=helper_api_headers,
        )

        # Assertions
        if should_succeed:
            assert response.status_code == 200
            json_data = response.json()  # pyright: ignore[reportAny]
            assert json_data["success"] is True
            assert json_data["sql"] == "SELECT 1"
            assert json_data["chart_config"]["chart_type"] == "bar"
        else:
            assert response.status_code == 500
            json_data = response.json()  # pyright: ignore[reportAny]
            assert "Agent processing failed" in json_data["detail"]

        mock_agent.assert_awaited_once()

    @pytest.mark.parametrize(
        "should_succeed",
        [True, False],
        ids=["Success Case", "Failure Case"],
    )
    def test_get_chart_data(
        self,
        test_client: TestClient,
        helper_api_headers: APIHeader,
        mocker: MockFixture,
        should_succeed: bool,
    ):
        """Test get_chart_data endpoint."""
        # Mock
        mock_db = mocker.Mock(spec=DatabaseHandler)
        if should_succeed:
            mock_db.execute_query.return_value = [  # pyright: ignore[reportAny]
                {"col": "value"}
            ]
        else:
            mock_db.execute_query.side_effect = Exception(  # pyright: ignore[reportAny]
                "Database error"
            )

        _ = mocker.patch(
            "database.db.DatabaseHandler.from_credentials", return_value=mock_db
        )

        # Main endpoint request
        response = test_client.post(
            "/api/get_chart_data",
            json={
                "sql_query": (
                    "SELECT 1" if should_succeed else "SELECT * FROM nonexistent"
                ),
                "db_config": DatabaseCredential(
                    db_type="sqlite",
                    database="file:test",
                    query=(
                        QueryParam(key="cache", value="shared"),
                        QueryParam(key="mode", value="memory"),
                        QueryParam(key="uri", value="true"),
                    ),
                ).model_dump(),
            },
            headers=helper_api_headers,
        )

        # Assertions
        if should_succeed:
            assert response.status_code == 200
            assert response.json() == [{"col": "value"}]
        else:
            assert response.status_code == 500

        mock_db.execute_query.assert_called_once()  # pyright: ignore[reportAny]
