"""
Node definitions for the agent graph.
"""

from collections.abc import Callable
import re
from typing import Annotated, cast

from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.constants import END
from langgraph.types import Command
from typing_extensions import Doc

from agent.deps import DependencyFactory
from agent.llm import LLMChart, LLMPlanner, LLMValidator
from agent.states import ChartConfig, GraphState
from database import DatabaseCredential

RE_TIME = re.compile(r"date|time|year|timestamp|(_dt|^dt$)|month|day", re.I)
RE_ID = re.compile(r"id|_id|uuid|pk|key", re.I)


def get_deps(
    config: RunnableConfig,
) -> tuple[DependencyFactory, DatabaseCredential]:
    config = ensure_config(config)
    configurable = config.get("configurable") or {}

    deps = configurable.get("deps")
    db_config = configurable.get("db_config")

    if deps is None:
        raise ValueError("DependencyFactory ('deps') missing in config.configurable")

    if db_config is None:
        raise ValueError("Database config ('db_config') missing in config.configurable")

    return cast(DependencyFactory, deps), cast(DatabaseCredential, db_config)


def plan_node(
    state: Annotated[GraphState, Doc("The graph state for langgraph agent.")],
    config: RunnableConfig,
) -> Command[str]:
    """Generates a SQL query plan based on the question and database schema."""
    llm_planner = LLMPlanner()

    deps, db_config = get_deps(config)
    db_schema = deps.get_db(db_config).get_schema()

    try:
        node_config = config.copy()
        node_config["tags"] = node_config.get("tags", []) + ["plan_node"]
        return Command(
            goto="sql_validation",
            update={
                "sql_blueprint": llm_planner(
                    state.question,
                    db_schema=db_schema,
                    retry=bool(state.error),
                    previous_blueprint=state.sql_blueprint if state.error else None,
                    error_message=state.error,
                    config=node_config,
                ),
                "db_schema": db_schema,
                "error": None,
            },
        )
    except Exception as e:
        if state.retry_count <= 3:
            return Command(
                goto="plan",
                update={
                    "error": str(e),
                    "retry_count": state.retry_count + 1,
                    "success": False,
                },
            )

    return Command(
        goto=END,
        update={"error": f"Planning failed after multiple attempts", "success": False},
    )


def sql_validation_node(
    state: Annotated[GraphState, Doc("The graph state for langgraph agent.")],
    config: RunnableConfig,
) -> Command[str]:
    llm_validator = LLMValidator()

    deps, db_config = get_deps(config)
    db = deps.get_db(db_config)

    if not state.sql_blueprint:
        return Command(goto="plan", update={"success": False})

    if state.retry_count <= 3:
        try:
            sql_query = db.generate_sql_query(state.sql_blueprint)
            text_sql_query = db.compile_sql_query(sql_query)
            preview = db.execute_query(sql_query.limit(10))
            node_config = config.copy()
            node_config["tags"] = node_config.get("tags", []) + ["sql_validation_node"]

            validation_output = llm_validator(
                question=state.question,
                sql_dialect=db.engine.dialect.name,
                sql_query=text_sql_query,
                preview=preview,
                db_schema=state.db_schema,
                config=node_config,
            )
            if not validation_output.is_valid:
                return Command(
                    goto="plan",
                    update={
                        "error": validation_output.error_message,
                        "retry_count": state.retry_count + 1,
                        "success": False,
                    },
                )

            if validation_output.confidence_score < 0.85:
                return Command(
                    goto="plan",
                    update={
                        "error": f"Low confidence SQL validation: {validation_output.error_message or 'uncertain correctness'}",
                        "retry_count": state.retry_count + 1,
                        "success": False,
                    },
                )

            return Command(
                goto="chart",
                update={
                    "preview": preview,
                    "sql": text_sql_query,
                    "error": None,
                },
            )

        except Exception as e:
            return Command(
                goto="plan",
                update={
                    "error": f"Validation failed: {str(e)}",
                    "retry_count": state.retry_count + 1,
                    "success": False,
                },
            )
    return Command(
        goto=END,
        update={"error": f"Planning failed after multiple attempts", "success": False},
    )


def _rule_based_chart(state: GraphState) -> ChartConfig | None:
    if not (data := state.preview):
        return None

    sample = data[0]
    cols = list(sample.keys())
    total = len(data)

    uniques = {c: len({r[c] for r in data if r[c] is not None}) for c in cols}

    time_cols: list[str] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for c in cols:
        if RE_TIME.search(c):
            time_cols.append(c)
            continue

        val = sample[c]  # pyright: ignore[reportAny]
        if isinstance(val, (int, float)) and not RE_ID.search(c):
            if uniques[c] > 0:
                numeric_cols.append(c)

        elif isinstance(val, str) and 1 < uniques[c] <= (total * 0.6):
            categorical_cols.append(c)

    fmt_title: Callable[[str], str] = lambda s: s.replace("_", " ").title()

    match (time_cols, categorical_cols, numeric_cols):
        # Simple Time Series
        case ([t, *_], _, [n]) if all(r[n] >= 0 for r in data if r[n] is not None):
            return ChartConfig(
                chart_type="area",
                x=t,
                y=n,
                title=f"Total {fmt_title(n)} over {fmt_title(t)}",
            )

        case ([t, *_], _, [n]):
            return ChartConfig(
                chart_type="line",
                x=t,
                y=n,
                title=f"{fmt_title(n)} Trends over {fmt_title(t)}",
            )

        # Simple Grouped Bar
        case (_, [cx, cg], [n]) if uniques[cg] <= 10:
            return ChartConfig(
                chart_type="bar",
                x=cx,
                y=n,
                color=cg,
                title=f"{fmt_title(n)} by {fmt_title(cx)} and {fmt_title(cg)}",
            )

        case ([], [c], [n]) if 2 <= uniques[c] <= 5:
            return ChartConfig(
                chart_type="pie",
                x=c,
                y=n,
                title=f"{fmt_title(n)} Distribution by {fmt_title(c)}",
            )

        case ([], [c], [n]):
            return ChartConfig(
                chart_type="bar", x=c, y=n, title=f"{fmt_title(n)} by {fmt_title(c)}"
            )

        # Scatter: Correlation between exactly two numbers
        case ([], [], [n1, n2]):
            return ChartConfig(
                chart_type="scatter",
                x=n1,
                y=n2,
                title=f"Correlation: {fmt_title(n1)} vs {fmt_title(n2)}",
            )

        case _:
            return None


def chart_node(
    state: Annotated[GraphState, Doc("The graph state for langgraph agent.")],
    config: RunnableConfig,
) -> Command[str]:
    """Generates chart configuration using rules first, then LLM fallback."""
    if chart := _rule_based_chart(state):
        return Command(
            goto=END,
            update={
                "chart_config": chart,
            },
        )

    node_config = config.copy()
    node_config["tags"] = node_config.get("tags", []) + ["chart_node"]

    llm_chart = LLMChart()
    try:
        chart = llm_chart(
            question=state.question,
            preview=state.preview,
            config=node_config,
        )

        return Command(goto=END, update={"chart_config": chart, "success": True})

    except Exception as e:
        return Command(
            goto=END,
            update={"error": f"Chart generation failed: {str(e)}", "success": False},
        )
