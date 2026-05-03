"""
Compilation of agent graph from nodes.
"""

import asyncio
from typing import Annotated
from langgraph.graph import StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from typing_extensions import Doc

from config import settings

from .deps import DependencyFactory
from database import DatabaseCredential

from .nodes import (
    chart_node,
    plan_node,
    sql_validation_node,
)

from .states import AgentInput, AgentOutput, GraphState

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


async def run_agent(
    input: Annotated[AgentInput, Doc("The input question for the agent.")],
    db_config: Annotated[
        DatabaseCredential, Doc("The credentials to connect to the database.")
    ],
    thread_id: Annotated[str, Doc("The thread ID for the agent execution.")],
) -> Annotated[AgentOutput, Doc("The output from the agent.")]:
    config = RunnableConfig(
        configurable={
            "thread_id": thread_id,
            "deps": DependencyFactory(),
            "db_config": db_config,
        }
    )

    raw_output = (
        await sql_chart_builder.ainvoke(  # pyright: ignore[reportUnknownMemberType]
            input=input, config=config, version="v2"
        )
    )
    return raw_output.value


# NOTE: The following main block is for demonstration and testing purposes. In a production environment, this module would typically be imported and the `run_agent` function would be called from an API endpoint or another part of the application.
if __name__ == "__main__":
    from langchain_core.globals import set_debug

    set_debug(True)
    print(
        asyncio.run(
            run_agent(
                input=AgentInput(
                    question="Total quantity of sales per store ID.",
                ),
                db_config=DatabaseCredential(
                    db_type=settings.db.type,
                    database=settings.db.name,
                    username=settings.db.username,
                    password=settings.db.password,
                    host=settings.db.host,
                    port=settings.db.port,
                ),
                thread_id="test_thread",
            )
        ).model_dump_json(indent=4)
    )
