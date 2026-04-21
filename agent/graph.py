"""
Compilation of agent graph from nodes.
"""

import os
from typing import Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langchain_core.runnables import RunnableConfig
from typing_extensions import Doc

from agent.deps import DependencyFactory
from agent.nodes import chart_node, plan_node, sql_validation_node
from agent.states import AgentInput, AgentOutput, GraphState
from database.schemas import DatabaseCredential

_ = load_dotenv()

graph = StateGraph(GraphState, input_schema=AgentInput, output_schema=AgentOutput)

_ = graph.add_node("plan", plan_node)  # pyright: ignore[reportUnknownMemberType]
_ = graph.add_node(  # pyright: ignore[reportUnknownMemberType]
    "sql_validation", sql_validation_node
)
_ = graph.add_node("chart", chart_node)  # pyright: ignore[reportUnknownMemberType]

_ = graph.set_entry_point("plan")

sql_chart_builder = graph.compile(  # pyright: ignore[reportUnknownMemberType]
    name="SQLandChartQueryGeneratorAgent"
)


# NOTE: This is just for testing purposes. In production, the database credentials should be passed by application.
_db_config = DatabaseCredential(
    db_type="postgresql",
    database=os.getenv("DB_NAME"),  # pyright: ignore[reportArgumentType]
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT")),  # pyright: ignore[reportArgumentType]
)


def run_agent(
    input: Annotated[AgentInput, Doc("The input question for the agent.")],
) -> Annotated[AgentOutput, Doc("The output from the agent.")]:
    config = RunnableConfig(
        tags=["sql-chart-agent"],
        metadata={
            "use_case": "sql_generation",
            "env": "dev",
        },
        configurable={
            "deps": DependencyFactory(),
            "db_config": _db_config,
        },
        run_name="sql_chart_pipeline",
    )

    return sql_chart_builder.invoke(  # pyright: ignore[reportUnknownMemberType, reportReturnType]
        input=input,
        config=config,
    )
