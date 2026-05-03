"""Tests for API endpoints using FastAPI TestClient."""

from collections.abc import Generator
from typing import Annotated

from fastapi.testclient import TestClient
import pytest
from pytest_mock import MockFixture
from typing_extensions import Doc

from agent import AgentInput, AgentOutput, run_agent
from agent.states import ChartConfig
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
        monkeypatch: Annotated[
            pytest.MonkeyPatch,
            Doc(
                "Utility for setting up environmental variables and other things for the duration of the test."
            ),
        ],
    ):
        """Set environment variables for testing."""
        monkeypatch.setenv("INTERNAL_API_KEY", "test_api_key")

    @pytest.mark.asyncio
    async def test_submit_question_success(
        self,
        test_client: TestClient,
        helper_api_headers: APIHeader,
        mocker: MockFixture,
    ):
        """Test successful submit_question endpoint."""
        # Mock
        mock_agent = mocker.AsyncMock(spec=run_agent)
        mock_agent.return_value = AgentOutput(
            success=True,
            sql="SELECT 1",
            chart_config=ChartConfig(
                chart_type="bar", x="category", y="value", title="Test Chart"
            ),
        )
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

        # Assertion
        assert response.status_code == 200
        mock_agent.assert_awaited_once()

        json_data = response.json()  # pyright: ignore[reportAny]
        assert json_data["success"] is True
        assert json_data["sql"] == "SELECT 1"
        assert json_data["chart_config"]["chart_type"] == "bar"

    @pytest.mark.asyncio
    async def test_submit_question_failure(
        self,
        test_client: TestClient,
        helper_api_headers: APIHeader,
        mocker: MockFixture,
    ):
        """Test submit_question endpoint failure."""
        # Mock
        mock_agent = mocker.AsyncMock(spec=run_agent)
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
        assert response.status_code == 500
        json_data = response.json()  # pyright: ignore[reportAny]
        assert "Agent processing failed" in json_data["detail"]
        mock_agent.assert_awaited_once()

    def test_get_chart_data_success(
        self,
        test_client: TestClient,
        helper_api_headers: APIHeader,
        mocker: MockFixture,
    ):
        """Test successful get_chart_data endpoint."""
        # Mock
        mock_db = mocker.Mock(spec=DatabaseHandler)
        mock_db.execute_query.return_value = [  # pyright: ignore[reportAny]
            {"col": "value"}
        ]
        _ = mocker.patch(
            "database.db.DatabaseHandler.from_credentials", return_value=mock_db
        )

        # Main endpoint request
        response = test_client.post(
            "/api/get_chart_data",
            json={
                "sql_query": "SELECT 1",
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
        assert response.status_code == 200
        assert response.json() == [{"col": "value"}]
        mock_db.execute_query.assert_called_once()  # pyright: ignore[reportAny]

    def test_get_chart_data_failure(
        self,
        test_client: TestClient,
        helper_api_headers: APIHeader,
        mocker: MockFixture,
    ):
        """Test get_chart_data endpoint failure."""
        # Mock
        mock_db = mocker.Mock(spec=DatabaseHandler)
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
                "sql_query": "SELECT * FROM nonexistent",
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

        # Assertion
        assert response.status_code == 500
        mock_db.execute_query.assert_called_once()  # pyright: ignore[reportAny]
