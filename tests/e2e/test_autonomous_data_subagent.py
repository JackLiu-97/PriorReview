from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field

from agent.core.compiled_data_subagent import build_compiled_data_subagent, build_data_subagent_tools
from agent.core.deep_agents import (
    MAIN_DEEP_AGENT_EXCLUDED_TOOLS,
    _MainToolFilterMiddleware,
    _deny_all_filesystem_permissions,
    _make_local_backend,
)
from agent.core.prompts import build_data_subagent_prompt
from agent.core.settings import Settings


class ScriptedToolModel(FakeMessagesListChatModel):
    bound_tool_names: set[str] = Field(default_factory=set)

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names.update(tool.name for tool in tools)
        return self


class FakeSQLRepository:
    def __init__(self, settings):
        pass

    def execute_query_plan(self, plan, *, max_rows):
        rows = [{"rating": index, "note": "historical evidence " * 4} for index in range(150)]
        return {
            "available": True, "mode": "test", "error": None,
            "executed_queries": [{"name": "query_01", "row_count": len(rows), "error": None}],
            "results": {"query_01": rows},
        }


def call(name: str, arguments: dict, identifier: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": identifier, "type": "tool_call"}])


class AutonomousDataSubagentTest(unittest.TestCase):
    def test_parent_delegation_runs_query_code_repair_full_result_analysis_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "artifacts" / "data" / "reviewer_round_01"
            script = (
                "import json\nfrom pathlib import Path\n"
                "rows = json.loads(Path('query_01.result.json').read_text(encoding='utf-8'))['results']['query_01']\n"
                "result = {'count': len(rows), 'mean': sum(row['rating'] for row in rows) / len(rows), 'last': rows[-1]['rating']}\n"
                "Path('analysis.json').write_text(json.dumps(result), encoding='utf-8')\n"
                "print(json.dumps(result))\n"
            )
            child_model = ScriptedToolModel(responses=[
                call("write_sql_artifact", {"filename": "query_01.sql", "sql": "SELECT rating_numeric FROM reviews LIMIT 150"}, "sql-write"),
                call("run_sql_artifact", {"sql_filename": "query_01.sql", "output_filename": "query_01.result.json", "max_rows": 150}, "sql-run"),
                call("write_file", {"file_path": "/analysis.py", "content": "print(undefined_variable)\n"}, "code-write"),
                call("execute", {"command": "python analysis.py"}, "code-error"),
                call("edit_file", {"file_path": "/analysis.py", "old_string": "print(undefined_variable)\n", "new_string": script}, "code-repair"),
                call("execute", {"command": "python analysis.py"}, "code-run"),
                call("read_file", {"file_path": "/analysis.json"}, "result-read"),
                call("write_data_summary", {"content": "Analyzed all 150 records. Mean rating=74.5; last=149. Evidence: query_01.result.json, analysis.py, analysis.json. Synthetic fixture, not a scientific conclusion."}, "summary"),
                AIMessage(content="Analyzed 150 records; mean=74.5. See analysis.json and data_summary.md."),
            ])
            settings = Settings(data_agent_max_steps=100)
            real_builder = build_data_subagent_tools
            with patch(
                "agent.core.compiled_data_subagent.build_data_subagent_tools",
                side_effect=lambda **kwargs: real_builder(**kwargs, sql_repository_factory=FakeSQLRepository),
            ):
                child = build_compiled_data_subagent(
                    model=child_model, settings=settings, run_root=root, agent_name="reviewer",
                    system_prompt=build_data_subagent_prompt("reviewer", sql_schema_context="", sql_planner_rules=""),
                )
            parent_model = ScriptedToolModel(responses=[
                    call("task", {"subagent_type": "data_specialist", "description": "Analyze all retrieved ratings; save code and summarize."}, "delegate"),
                    AIMessage(content="Evidence received."),
            ])
            # Exercise the real main-agent filesystem deny rules as well: they
            # must not remove the compiled child's independent coding tools.
            parent = create_deep_agent(
                model=parent_model,
                backend=_make_local_backend(run_root=root),
                permissions=_deny_all_filesystem_permissions(),
                middleware=[_MainToolFilterMiddleware(MAIN_DEEP_AGENT_EXCLUDED_TOOLS)],
                subagents=[
                    {"name": "data_specialist", "description": "Autonomous data analysis", "runnable": child},
                ],
            )
            state = parent.invoke(
                {"messages": [HumanMessage(content="Collect historical evidence.")]},
                {"configurable": {"assigned_data_workspace_abs": str(workspace), "timeline_role": "reviewer", "timeline_round": 1}},
            )
            result = json.loads((workspace / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(result, {"count": 150, "mean": 74.5, "last": 149})
            self.assertIn("150", (workspace / "data_summary.md").read_text(encoding="utf-8"))
            delegated = next(message for message in state["messages"] if isinstance(message, ToolMessage))
            self.assertIn("mean=74.5", delegated.content)
            self.assertTrue({"execute", "read_file", "write_file", "edit_file", "ls", "glob", "grep"} <= child_model.bound_tool_names)
            self.assertNotIn("task", child_model.bound_tool_names)
            self.assertFalse(MAIN_DEEP_AGENT_EXCLUDED_TOOLS.intersection(parent_model.bound_tool_names))
            self.assertEqual(len(list((workspace / "_execution").glob("*.stdout.log"))), 2)
            traces = list((root / "debug" / "tool_traces" / "reviewer_round_01").rglob("events.jsonl"))
            self.assertTrue(traces)
            trace_text = "\n".join(path.read_text(encoding="utf-8") for path in traces)
            self.assertIn("NameError", trace_text)
            self.assertIn("results_preview_truncated", trace_text)
            summary_path = next((root / "debug" / "tool_traces").rglob("trace_summary.json"))
            trace_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(Path(trace_summary["parent_workspace"]), workspace.resolve())

    def test_execution_switch_removes_code_tools_and_updates_the_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(data_agent_code_execution_enabled=False)
            prompt = build_data_subagent_prompt(
                "reviewer", sql_schema_context="", sql_planner_rules="", code_execution_enabled=False,
            )
            model = ScriptedToolModel(responses=[AIMessage(content="No query needed.")])
            child = build_compiled_data_subagent(
                model=model, settings=settings, run_root=Path(tmp), agent_name="reviewer", system_prompt=prompt,
            )
            child.invoke({"messages": [HumanMessage(content="Report the limitation.")]})
            self.assertEqual(model.bound_tool_names, {"write_sql_artifact", "run_sql_artifact", "run_rag_query", "write_data_summary"})
            self.assertIn("disabled general code execution", prompt)
            self.assertNotIn("Autonomous data analysis workflow", prompt)

    def test_all_roles_receive_coding_tools_with_finalizer_rag_scope_preserved(self) -> None:
        for role in ("reviewer", "author", "sc", "finalizer"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model = ScriptedToolModel(responses=[AIMessage(content="Ready to analyze evidence.")])
                child = build_compiled_data_subagent(
                    model=model, settings=Settings(), run_root=root, agent_name=role,
                    system_prompt=build_data_subagent_prompt(role, sql_schema_context="", sql_planner_rules=""),
                )
                child.invoke(
                    {"messages": [HumanMessage(content="Report your task.")]},
                    {"configurable": {"assigned_data_workspace_abs": str(root / "artifacts" / "data" / f"{role}_round_01")}},
                )
                self.assertTrue({"execute", "read_file", "write_file", "edit_file"} <= model.bound_tool_names)
                self.assertEqual("run_rag_query" in model.bound_tool_names, role != "finalizer")


if __name__ == "__main__":
    unittest.main()
