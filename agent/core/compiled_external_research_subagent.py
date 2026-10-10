from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.agents import create_agent

from agent.core.settings import Settings
from agent.logging.execution_log import ExecutionLog
from agent.logging.subagent_trace import SubAgentTraceMiddleware
from agent.tools.external_research import build_external_research_tools


def build_compiled_external_research_subagent(
    *,
    model: Any,
    settings: Settings,
    run_root: Path,
    agent_name: str,
    system_prompt: str,
    execution_log: ExecutionLog | None = None,
) -> Any:
    tools = build_external_research_tools(settings=settings, run_root=run_root, execution_log=execution_log)
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            SubAgentTraceMiddleware(
                run_root=run_root,
                subagent_name="external_researcher",
                system_prompt=system_prompt,
                trace_root=run_root / "debug" / "tool_traces",
            )
        ],
        name=f"{agent_name}_external_researcher",
    )
