"""API calls the UI may make."""

from fastapi import status
from fastapi.exceptions import HTTPException
from fastapi.params import Security
from fastapi.routing import APIRouter
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing_extensions import Annotated, Doc

from agent import AgentInput, AgentOutput, run_agent
from config import settings
from database import DatabaseCredential, DatabaseHandler
from web.tags import APITag

api_key_header = APIKeyHeader(
    name="X-API-KEY", description="API Key Header", auto_error=True
)


class ChartDataRequest(BaseModel):
    """Request body for chart data endpoint."""

    sql_query: str
    db_config: DatabaseCredential


def verify_api_key(
    api_key: Annotated[str, Security(api_key_header)],
):
    """Verify the API key from the Authorization header."""
    if not settings.internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal API key not configured",
        )
    if api_key != settings.internal_api_key.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )
    return api_key


api_router = APIRouter(
    prefix="/api",
    tags=[APITag.API],
    dependencies=[
        Security(verify_api_key),
    ],
)


@api_router.post(
    "/submit_question",
    tags=[APITag.USER],
)
async def submit_question(
    agent_input: Annotated[AgentInput, Doc("The input pass to Agentic AI.")],
    db_config: Annotated[DatabaseCredential, Doc("Credentials to database.")],
) -> AgentOutput:
    try:
        return await run_agent(
            input=agent_input,
            db_config=db_config,
            thread_id="default_thread",  # TODO: Once UI has been implemented, replace this with a session id passed by UI.
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent processing failed: {str(e)}",
        )


@api_router.post("/get_chart_data", tags=[APITag.DB])
def get_chart_data(
    request: Annotated[ChartDataRequest, Doc("The request body for this endpoint.")],
):
    try:
        db = DatabaseHandler.from_credentials(request.db_config)
        return db.execute_query(request.sql_query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query failed: {str(e)}",
        )
