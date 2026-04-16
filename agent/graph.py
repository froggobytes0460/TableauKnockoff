"""
Compilation of agent graph from nodes.
"""

from langgraph.graph import StateGraph  # pyright: ignore[reportMissingTypeStubs]

from agent.nodes import DatabaseContext, init_node, plan_node, validation_node
from agent.schemas import GraphState

graph = StateGraph(GraphState, DatabaseContext)

_ = graph.add_node("init", init_node)  # pyright: ignore[reportUnknownMemberType]
_ = graph.add_node("plan", plan_node)  # pyright: ignore[reportUnknownMemberType]
_ = graph.add_node(  # pyright: ignore[reportUnknownMemberType]
    "validation", validation_node
)

agent = graph.compile(  # pyright: ignore[reportUnknownMemberType]
    name="SQLQueryGeneratorAgent"
)
