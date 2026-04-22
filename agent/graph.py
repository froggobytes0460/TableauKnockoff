"""
Compilation of agent graph from nodes.
"""

import os
from pathlib import Path
from typing import Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from typing_extensions import Doc

from agent.deps import DependencyFactory
from agent.nodes import chart_node, plan_node, sql_validation_node
from agent.states import AgentInput, AgentOutput, GraphState
from database import DatabaseCredential

if not (path := Path(".env").resolve()).exists():
    raise FileNotFoundError(f"Missing .env file at path: {path}")

if not load_dotenv(path):
    raise EnvironmentError(f"Failed to load .env file at path: {path}") from None

graph = StateGraph(GraphState, input_schema=AgentInput, output_schema=AgentOutput)

_ = graph.add_node("plan", plan_node)  # pyright: ignore[reportUnknownMemberType]
_ = graph.add_node(  # pyright: ignore[reportUnknownMemberType]
    "sql_validation", sql_validation_node
)
_ = graph.add_node("chart", chart_node)  # pyright: ignore[reportUnknownMemberType]

_ = graph.set_entry_point("plan")

sql_chart_builder = graph.compile(  # pyright: ignore[reportUnknownMemberType]
    name="SQLandChartQueryGeneratorAgent", checkpointer=MemorySaver()
)


def run_agent(
    input: Annotated[AgentInput, Doc("The input question for the agent.")],
    db_config: Annotated[
        DatabaseCredential, Doc("The credentials to connect to the database.")
    ],
    thread_id: Annotated[str, Doc("The thread ID for the agent execution.")],
) -> Annotated[AgentOutput, Doc("The output from the agent.")]:
    config = RunnableConfig(
        tags=["sql-chart-agent"],
        metadata={
            "use_case": "sql_generation",
            "env": "dev",
        },
        configurable={
            "thread_id": thread_id,
            "deps": DependencyFactory(),
            "db_config": db_config,
        },
        run_name="sql_chart_pipeline",
    )

    return sql_chart_builder.invoke(  # pyright: ignore[reportUnknownMemberType, reportReturnType]
        input=input,
        config=config,
    )


# NOTE: The following main block is for demonstration and testing purposes. In a production environment, this module would typically be imported and the `run_agent` function would be called from an API endpoint or another part of the application.
if __name__ == "__main__":
    from langchain_core.globals import set_debug

    _db_config = DatabaseCredential(
        db_type="postgresql",
        database=os.getenv("DB_NAME"),  # pyright: ignore[reportArgumentType]
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),  # pyright: ignore[reportArgumentType]
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),  # pyright: ignore[reportArgumentType]
    )
    input = AgentInput(
        question="Total quantity of sales per store ID.",
    )
    set_debug(True)
    output = run_agent(input=input, db_config=_db_config, thread_id="test_thread")
    print(output)
