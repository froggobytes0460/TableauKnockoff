"""
Compilation of agent graph from nodes.
"""

from langgraph.graph import StateGraph  # pyright: ignore[reportMissingTypeStubs]

from agent.nodes import DatabaseContext, init_node, plan_node
from agent.schemas import GraphState

graph = StateGraph(GraphState, DatabaseContext)

_ = graph.add_node("init", init_node)
_ = graph.add_node("plan", plan_node)

agent = graph.compile(  # pyright: ignore[reportUnknownMemberType]
    name="SQLQueryGeneratorAgent"
)
