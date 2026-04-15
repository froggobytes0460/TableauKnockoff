"""
Node definitions for the agent graph.
"""

from typing import Annotated, Literal, TypedDict
from langgraph.graph import END  # pyright: ignore[reportMissingTypeStubs]
from langgraph.types import Command
from langgraph.runtime import Runtime
from typing_extensions import Doc

from agent.llm import LLMPlanner
from agent.schemas import GraphState
from database import DatabaseHandler


class DatabaseContext(TypedDict):
    db: DatabaseHandler


def init_node(
    state: Annotated[  # pyright: ignore[reportUnusedParameter]
        GraphState, Doc("The graph state for langgraph agent.")
    ],
    runtime: Annotated[
        Runtime[DatabaseContext], Doc("The runtime context for langgraph agent.")
    ],
) -> Command[Literal["plan"]]:
    """Initializes the graph state with the original question and database schema."""
    return Command(
        goto="plan", update={"db_schema": runtime.context["db"].get_schema()}
    )


def plan_node(
    state: Annotated[GraphState, Doc("The graph state for langgraph agent.")],
) -> Command[str]:
    """Generates a SQL query plan based on the question and database schema."""
    llm_planner = LLMPlanner()
    try:
        return Command(
            goto="validation",
            update={
                "sql_blueprint": llm_planner(
                    state.question,
                    state.db_schema,
                    retry=bool(state.error),
                    previous_blueprint=state.sql_blueprint if state.error else None,
                    error_message=state.error,
                )
            },
        )
    except Exception as e:
        if state.retry_count <= 3:
            return Command(
                goto="plan",
                update={"error": str(e), "retry_count": state.retry_count + 1},
            )

    return Command(goto=END)
