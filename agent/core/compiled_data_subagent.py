from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any, Callable

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, tool

from agent.core.rag_context import RAGRepository, RAG_EMBEDDING_FIELDS, RAG_TABLE_NAMES
from agent.core.settings import Settings
from agent.core.data_agent_workspace import build_data_filesystem_middleware, resolve_data_workspace
from agent.core.sql_context import SQLRepository
from agent.logging.execution_log import ExecutionLog
from agent.logging.subagent_trace import SubAgentTraceMiddleware
from agent.schemas.common import SQLQueryCandidate, SQLQueryPlan


_JSON_WORKSPACE_RE = re.compile(r'"assigned_data_workspace_abs"\s*:\s*"(?P<path>(?:\\.|[^"\\])*)"')
_QUOTED_WORKSPACE_RE = re.compile(
    r"assigned_data_workspace_abs[`'\"]?\s*[:=]\s*[`'\"](?P<path>[^\n\r`'\"]+)[`'\"]",
    re.IGNORECASE,
)
_UNQUOTED_WORKSPACE_RE = re.compile(
    r"assigned_data_workspace_abs[`'\"]?\s*[:=]\s*(?P<path>[^\n\r]+)",
    re.IGNORECASE,
)
_PROSE_WORKSPACE_RE = re.compile(
    r"assigned data workspace:\s*[`'\"]?(?P<path>[^\n\r`'\"]+)[`'\"]?",
    re.IGNORECASE,
)
_ASSIGNED_WORKSPACE_RE = re.compile(
    r"assigned workspace:\s*[`'\"]?(?P<path>[^\n\r`'\"]+)[`'\"]?",
    re.IGNORECASE,
)
SQL_RESULTS_PREVIEW_MAX_CHARS = 5000


class ControlledDataSubagentTools:
    """Controlled SQL/RAG tools for the compiled data_specialist subagent."""

    def __init__(
        self,
        *,
        settings: Settings,
        run_root: Path | None = None,
        execution_log: ExecutionLog | None = None,
        sql_repository_factory: Callable[[Settings], SQLRepository] = SQLRepository,
        rag_repository_factory: Callable[[Settings], RAGRepository] = RAGRepository,
        enable_rag: bool = True,
        allowed_sql_tables: set[str] | frozenset[str] | None = None,
        agent_name: str | None = None,
    ) -> None:
        self.settings = settings
        self.run_root = run_root
        self.execution_log = execution_log
        self.sql_repository_factory = sql_repository_factory
        self.rag_repository_factory = rag_repository_factory
        self.enable_rag = enable_rag
        self.allowed_sql_tables = frozenset(allowed_sql_tables) if allowed_sql_tables is not None else None
        self.agent_name = agent_name

    def to_tools(self) -> list[BaseTool]:
        write_sql_artifact_impl = self.write_sql_artifact
        run_sql_artifact_impl = self.run_sql_artifact
        run_rag_query_impl = self.run_rag_query
        write_data_summary_impl = self.write_data_summary

        @tool("write_sql_artifact")
        def write_sql_artifact_tool(
            filename: Annotated[str, "Relative .sql artifact filename inside assigned_data_workspace_abs."],
            sql: Annotated[str, "Read-only SQL query text to save."],
            runtime: ToolRuntime,
        ) -> str:
            """Write a read-only SQL query artifact inside assigned_data_workspace_abs."""
            return write_sql_artifact_impl(state=runtime.state, config=runtime.config, filename=filename, sql=sql)

        @tool("run_sql_artifact")
        def run_sql_artifact_tool(
            sql_filename: Annotated[str, "Relative .sql filename inside assigned_data_workspace_abs."],
            output_filename: Annotated[str, "Relative .json output filename inside assigned_data_workspace_abs."],
            runtime: ToolRuntime,
            name: Annotated[str, "Logical query name."] = "query_01",
            reason: Annotated[str, "Reason for the query."] = "Historical prior query requested by data_specialist.",
            max_rows: Annotated[int, "Maximum result rows to return."] = 20,
        ) -> str:
            """Execute a saved SQL artifact through the controlled SQL repository and write JSON output."""
            return run_sql_artifact_impl(
                state=runtime.state,
                config=runtime.config,
                sql_filename=sql_filename,
                output_filename=output_filename,
                name=name,
                reason=reason,
                max_rows=max_rows,
            )

        @tool("run_rag_query")
        def run_rag_query_tool(
            query: Annotated[str, "Semantic retrieval query."],
            output_filename: Annotated[str, "Relative .json output filename inside assigned_data_workspace_abs."],
            runtime: ToolRuntime,
            top_k: Annotated[int, "Maximum merged documents to return."] = 5,
            per_source_limit: Annotated[int | None, "Maximum results per source before merging."] = None,
            conference: Annotated[str | None, "Optional conference filter, for example ICLR."] = None,
            year: Annotated[int | None, "Optional year filter."] = None,
            accepted: Annotated[bool | None, "Optional accepted-paper filter."] = None,
            tables: Annotated[list[str] | None, "Optional RAG tables to search."] = None,
            embedding_fields: Annotated[list[str] | None, "Optional embedding fields: label and/or keywords."] = None,
            max_snippet_chars: Annotated[int, "Maximum snippet length saved in the JSON artifact."] = 800,
        ) -> str:
            """Run controlled pgvector RAG retrieval and write JSON output."""
            return run_rag_query_impl(
                state=runtime.state,
                config=runtime.config,
                query=query,
                output_filename=output_filename,
                top_k=top_k,
                per_source_limit=per_source_limit,
                conference=conference,
                year=year,
                accepted=accepted,
                tables=tables,
                embedding_fields=embedding_fields,
                max_snippet_chars=max_snippet_chars,
            )

        @tool("write_data_summary")
        def write_data_summary_tool(
            content: Annotated[str, "Concise markdown summary of the data findings and artifact names."],
            runtime: ToolRuntime,
        ) -> str:
            """Write the final data_summary.md artifact inside assigned_data_workspace_abs."""
            return write_data_summary_impl(state=runtime.state, config=runtime.config, content=content)

        tools = [
            write_sql_artifact_tool,
            run_sql_artifact_tool,
            write_data_summary_tool,
        ]
        if self.enable_rag:
            tools.insert(2, run_rag_query_tool)
        return tools

    def write_sql_artifact(self, *, state: Any, filename: str, sql: str, config: Any | None = None) -> str:
        workspace = _extract_data_workspace(state=state, config=config, run_root=self.run_root, agent_name=self.agent_name)
        ctx = self._timeline_context(config=config, workspace=workspace)
        self._record_tool_call(ctx, "write_sql_artifact", {"filename": filename, "sql_preview": _truncate_text(sql, 240)})
        path_or_error = _resolve_workspace_file(workspace, filename, required_suffix=".sql")
        if isinstance(path_or_error, str):
            self._record_tool_result(ctx, "write_sql_artifact", {"error": path_or_error})
            return _json_error(path_or_error)
        path_or_error.parent.mkdir(parents=True, exist_ok=True)
        path_or_error.write_text(sql.strip() + "\n", encoding="utf-8")
        output = {
            "sql_file": self._rel(path_or_error),
            "bytes": path_or_error.stat().st_size,
            "error": None,
        }
        self._record_tool_result(ctx, "write_sql_artifact", output)
        return _json_success(
            action="write_sql_artifact",
            path=str(path_or_error),
            bytes=path_or_error.stat().st_size,
        )

    def run_sql_artifact(
        self,
        *,
        state: Any,
        sql_filename: str,
        output_filename: str,
        name: str = "query_01",
        reason: str = "Historical prior query requested by data_specialist.",
        max_rows: int = 20,
        config: Any | None = None,
    ) -> str:
        workspace = _extract_data_workspace(state=state, config=config, run_root=self.run_root, agent_name=self.agent_name)
        ctx = self._timeline_context(config=config, workspace=workspace)
        self._record_tool_call(
            ctx,
            "run_sql_artifact",
            {"sql_file": self._rel_workspace_value(workspace, sql_filename), "result_file": self._rel_workspace_value(workspace, output_filename), "name": name, "max_rows": max_rows},
        )
        sql_path_or_error = _resolve_workspace_file(workspace, sql_filename, required_suffix=".sql")
        if isinstance(sql_path_or_error, str):
            self._record_tool_result(ctx, "run_sql_artifact", {"error": sql_path_or_error})
            return _json_error(sql_path_or_error)
        output_path_or_error = _resolve_workspace_file(workspace, output_filename, required_suffix=".json")
        if isinstance(output_path_or_error, str):
            self._record_tool_result(ctx, "run_sql_artifact", {"error": output_path_or_error})
            return _json_error(output_path_or_error)
        if not sql_path_or_error.exists():
            error = f"SQL artifact does not exist: {sql_path_or_error}"
            self._record_tool_result(ctx, "run_sql_artifact", {"error": error})
            return _json_error(error)

        sql = sql_path_or_error.read_text(encoding="utf-8").strip()
        if self.allowed_sql_tables is not None:
            scoped_validation_error = self._scoped_sql_validation_error(sql=sql, max_rows=max_rows)
            if scoped_validation_error is not None:
                result = {
                    "available": True,
                    "mode": "validation_error",
                    "planned_queries": [
                        SQLQueryCandidate(name=name, sql=sql, reason=reason).model_dump(mode="json"),
                    ],
                    "executed_queries": [
                        {
                            "name": name,
                            "reason": reason,
                            "sql": sql,
                            "row_count": 0,
                            "error": scoped_validation_error,
                        }
                    ],
                    "results": {name: []},
                    "error": None,
                }
                return self._write_sql_result(
                    ctx=ctx,
                    output_path=output_path_or_error,
                    result=result,
                )

        repository = self.sql_repository_factory(self.settings)
        result = repository.execute_query_plan(
            SQLQueryPlan(
                queries=[
                    SQLQueryCandidate(
                        name=name,
                        sql=sql,
                        reason=reason,
                    )
                ]
            ),
            max_rows=max(1, int(max_rows)),
        )
        return self._write_sql_result(ctx=ctx, output_path=output_path_or_error, result=result)

    def _scoped_sql_validation_error(self, *, sql: str, max_rows: int) -> str | None:
        try:
            SQLRepository(self.settings, allowed_tables=self.allowed_sql_tables).validate_read_only_sql(
                sql,
                max_rows=max(1, int(max_rows)),
            )
            return None
        except Exception as exc:
            return str(exc)

    def _write_sql_result(self, *, ctx: dict[str, Any], output_path: Path, result: dict[str, Any]) -> str:
        compact_result = _compact_sql_result(result)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(compact_result, ensure_ascii=False, indent=2), encoding="utf-8")
        query_error_count = sum(1 for item in result.get("executed_queries", []) if item.get("error"))
        row_count = sum(int(item.get("row_count") or 0) for item in result.get("executed_queries", []) if isinstance(item, dict))
        results_preview, results_preview_truncated = _json_preview(
            result.get("results") or {},
            max_chars=SQL_RESULTS_PREVIEW_MAX_CHARS,
        )
        self._record_tool_result(
            ctx,
            "run_sql_artifact",
            {
                "result_file": self._rel(output_path),
                "row_count": row_count,
                "has_query_errors": query_error_count > 0,
                "results_preview_chars": len(results_preview),
                "results_preview_max_chars": SQL_RESULTS_PREVIEW_MAX_CHARS,
                "results_preview_truncated": results_preview_truncated,
                "error": result.get("error"),
            },
        )
        return _json_success(
            action="run_sql_artifact",
            output_path=str(output_path),
            available=bool(result.get("available")),
            mode=result.get("mode"),
            has_query_errors=query_error_count > 0,
            query_error_count=query_error_count,
            executed_queries=[
                {
                    "name": item.get("name"),
                    "row_count": item.get("row_count"),
                    "error": item.get("error"),
                }
                for item in result.get("executed_queries", [])
            ],
            results_preview=results_preview,
            results_preview_chars=len(results_preview),
            results_preview_max_chars=SQL_RESULTS_PREVIEW_MAX_CHARS,
            results_preview_truncated=results_preview_truncated,
            error=result.get("error"),
        )

    def run_rag_query(
        self,
        *,
        state: Any,
        query: str,
        output_filename: str,
        top_k: int = 5,
        per_source_limit: int | None = None,
        conference: str | None = None,
        year: int | None = None,
        accepted: bool | None = None,
        tables: list[str] | None = None,
        embedding_fields: list[str] | None = None,
        max_snippet_chars: int = 800,
        config: Any | None = None,
    ) -> str:
        if not self.enable_rag:
            return _json_error("RAG retrieval is not available for this data_specialist role.")
        workspace = _extract_data_workspace(state=state, config=config, run_root=self.run_root, agent_name=self.agent_name)
        ctx = self._timeline_context(config=config, workspace=workspace)
        self._record_tool_call(
            ctx,
            "run_rag_query",
            {
                "query": _truncate_text(query, 240),
                "result_file": self._rel_workspace_value(workspace, output_filename),
                "top_k": top_k,
                "conference": conference,
                "year": year,
                "accepted": accepted,
            },
        )
        output_path_or_error = _resolve_workspace_file(workspace, output_filename, required_suffix=".json")
        if isinstance(output_path_or_error, str):
            self._record_tool_result(ctx, "run_rag_query", {"error": output_path_or_error})
            return _json_error(output_path_or_error)

        invalid_tables = sorted(set(tables or []) - set(RAG_TABLE_NAMES))
        if invalid_tables:
            error = f"Unknown RAG table(s): {', '.join(invalid_tables)}"
            self._record_tool_result(ctx, "run_rag_query", {"error": error})
            return _json_error(error)
        invalid_fields = sorted(set(embedding_fields or []) - set(RAG_EMBEDDING_FIELDS))
        if invalid_fields:
            error = f"Unknown RAG embedding field(s): {', '.join(invalid_fields)}"
            self._record_tool_result(ctx, "run_rag_query", {"error": error})
            return _json_error(error)

        repository = self.rag_repository_factory(self.settings)
        result = repository.query(
            query,
            top_k=max(1, int(top_k)),
            per_source_limit=per_source_limit,
            conference=conference,
            year=year,
            accepted=accepted,
            tables=tables,
            embedding_fields=embedding_fields,
            max_snippet_chars=max(1, int(max_snippet_chars)),
        )
        compact_result = _compact_rag_result(result)
        output_path_or_error.parent.mkdir(parents=True, exist_ok=True)
        output_path_or_error.write_text(json.dumps(compact_result, ensure_ascii=False, indent=2), encoding="utf-8")
        documents = compact_result.get("documents") or []
        self._record_tool_result(
            ctx,
            "run_rag_query",
            {
                "result_file": self._rel(output_path_or_error),
                "document_count": len(documents),
                "error": compact_result.get("error"),
            },
        )
        return _json_success(
            action="run_rag_query",
            output_path=str(output_path_or_error),
            available=bool(compact_result.get("available")),
            mode=compact_result.get("mode"),
            query=compact_result.get("query", query),
            document_count=len(documents),
            top_documents=[
                {
                    "rank": doc.get("rank"),
                    "title": doc.get("title"),
                    "source_table": doc.get("source_table"),
                    "decision": doc.get("decision"),
                    "similarity": doc.get("similarity"),
                    "snippet": _truncate_text(str(doc.get("snippet") or ""), 300),
                }
                for doc in documents[:3]
                if isinstance(doc, dict)
            ],
            error=compact_result.get("error"),
        )

    def write_data_summary(self, *, state: Any, content: str, config: Any | None = None) -> str:
        workspace = _extract_data_workspace(state=state, config=config, run_root=self.run_root, agent_name=self.agent_name)
        ctx = self._timeline_context(config=config, workspace=workspace)
        self._record_tool_call(ctx, "write_data_summary", {"filename": "data_summary.md", "chars": len(content)})
        if workspace is None:
            error = "assigned_data_workspace_abs is missing from the subagent task."
            self._record_tool_result(ctx, "write_data_summary", {"error": error})
            return _json_error(error)
        path = _resolve_workspace_file(workspace, "data_summary.md", required_suffix=".md")
        if isinstance(path, str):
            self._record_tool_result(ctx, "write_data_summary", {"error": path})
            return _json_error(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.strip() + "\n", encoding="utf-8")
        self._record_tool_result(
            ctx,
            "write_data_summary",
            {"summary_file": self._rel(path), "bytes": path.stat().st_size, "error": None},
        )
        return _json_success(action="write_data_summary", path=str(path), bytes=path.stat().st_size)

    def _timeline_context(self, *, config: Any | None, workspace: Path | None) -> tuple[str | None, int | None]:
        configurable = _configurable(config)
        role = configurable.get("timeline_role")
        round_id = configurable.get("timeline_round")
        if not isinstance(role, str) or not role:
            role = self._role_from_workspace(workspace)
        if not isinstance(round_id, int):
            try:
                round_id = int(round_id) if round_id is not None else self._round_from_workspace(workspace)
            except (TypeError, ValueError):
                round_id = self._round_from_workspace(workspace)
        return role, round_id

    def _record_tool_call(self, ctx: tuple[str | None, int | None], name: str, input_payload: dict[str, Any]) -> None:
        if self.execution_log is None:
            return
        role, round_id = ctx
        self.execution_log.record(event="tool_call", role=role, round_id=round_id, name=name, input=input_payload)

    def _record_tool_result(self, ctx: tuple[str | None, int | None], name: str, output_payload: dict[str, Any]) -> None:
        if self.execution_log is None:
            return
        role, round_id = ctx
        self.execution_log.record(event="tool_result", role=role, round_id=round_id, name=name, output=output_payload)

    def _rel(self, path: Path) -> str:
        if self.execution_log is not None:
            return self.execution_log.rel(path)
        return str(path)

    def _rel_workspace_value(self, workspace: Path | None, value: str) -> str:
        if workspace is None:
            return value
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = workspace / candidate
        return self._rel(candidate)

    @staticmethod
    def _role_from_workspace(workspace: Path | None) -> str | None:
        if workspace is None:
            return None
        name = workspace.name
        return name.rsplit("_round_", 1)[0] if "_round_" in name else None

    @staticmethod
    def _round_from_workspace(workspace: Path | None) -> int | None:
        if workspace is None:
            return None
        name = workspace.name
        if "_round_" not in name:
            return None
        try:
            return int(name.rsplit("_round_", 1)[1])
        except ValueError:
            return None


def build_compiled_data_subagent(
    *,
    model: Any,
    settings: Settings,
    run_root: Path,
    agent_name: str,
    system_prompt: str,
    execution_log: ExecutionLog | None = None,
) -> Any:
    tools = build_data_subagent_tools(
        settings=settings,
        run_root=run_root,
        execution_log=execution_log,
        agent_name=agent_name,
    )

    def invoke(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        # Bind the actual role/round directory when the parent delegates. Each
        # invocation gets a concrete backend, avoiding shared mutable cwd and
        # deprecated backend factories in DeepAgents.
        middleware = [
            SubAgentTraceMiddleware(
                run_root=run_root,
                subagent_name="data_specialist",
                system_prompt=system_prompt,
                trace_root=run_root / "debug" / "tool_traces",
            )
        ]
        if settings.data_agent_code_execution_enabled:
            middleware.append(build_data_filesystem_middleware(
                settings=settings, run_root=run_root, agent_name=agent_name, config=config,
            ))
        agent = create_agent(
            model=model, tools=tools, system_prompt=system_prompt,
            middleware=middleware, name=f"{agent_name}_data_specialist",
        )
        return agent.invoke(state, {**config, "recursion_limit": settings.data_agent_max_steps})

    return RunnableLambda(invoke, name=f"{agent_name}_data_specialist")


def build_data_subagent_tools(
    *,
    settings: Settings,
    run_root: Path | None = None,
    execution_log: ExecutionLog | None = None,
    agent_name: str | None = None,
    sql_repository_factory: Callable[[Settings], SQLRepository] = SQLRepository,
    rag_repository_factory: Callable[[Settings], RAGRepository] = RAGRepository,
) -> list[BaseTool]:
    is_finalizer = agent_name == "finalizer"
    return ControlledDataSubagentTools(
        settings=settings,
        run_root=run_root,
        execution_log=execution_log,
        enable_rag=not is_finalizer,
        allowed_sql_tables=frozenset({"papers", "reviews"}) if is_finalizer else None,
        agent_name=agent_name,
        sql_repository_factory=sql_repository_factory,
        rag_repository_factory=rag_repository_factory,
    ).to_tools()


def _resolve_workspace_file(workspace: Path | None, value: str, *, required_suffix: str) -> Path | str:
    if workspace is None:
        return "assigned_data_workspace_abs is missing from the subagent task."
    if not value or not value.strip():
        return "Artifact filename is empty."
    candidate = Path(value.strip())
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved_workspace = workspace.expanduser().resolve(strict=False)
    resolved_candidate = candidate.expanduser().resolve(strict=False)
    if resolved_candidate != resolved_workspace and resolved_workspace not in resolved_candidate.parents:
        return f"Artifact path must stay inside assigned_data_workspace_abs: {value}"
    try:
        relative_parts = resolved_candidate.relative_to(resolved_workspace).parts
    except ValueError:
        return f"Artifact path must stay inside assigned_data_workspace_abs: {value}"
    if "subagent_traces" in relative_parts:
        return "Artifacts may not be written under subagent_traces."
    if resolved_candidate.suffix.lower() != required_suffix:
        return f"Artifact must use {required_suffix} suffix: {value}"
    return resolved_candidate


def _extract_data_workspace(
    *, state: Any, config: Any | None = None, run_root: Path | None = None, agent_name: str | None = None,
) -> Path | None:
    if run_root is not None:
        try:
            return resolve_data_workspace(config=config, run_root=run_root, agent_name=agent_name)
        except ValueError:
            return None
    # Compatibility for standalone helper callers without a workflow run. Live
    # compiled agents always bind run_root and require trusted runtime config.
    workspace = _extract_data_workspace_from_config(config, run_root=run_root)
    if workspace is not None:
        return workspace.expanduser().resolve(strict=False)
    return _extract_data_workspace_from_state(state)


def _extract_data_workspace_from_config(config: Any | None, *, run_root: Path | None) -> Path | None:
    configurable = _configurable(config)
    abs_workspace = configurable.get("assigned_data_workspace_abs")
    if isinstance(abs_workspace, str) and abs_workspace.strip():
        candidate = Path(abs_workspace.strip()).expanduser()
        if candidate.is_absolute():
            return candidate.resolve(strict=False)

    workspace = configurable.get("assigned_data_workspace")
    if not isinstance(workspace, str) or not workspace.strip():
        return None

    candidate = Path(workspace.strip()).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
        if run_root is None or _is_relative_to(resolved, run_root.expanduser().resolve(strict=False)):
            return resolved
        return None

    if run_root is None:
        return None
    resolved_run_root = run_root.expanduser().resolve(strict=False)
    resolved = (resolved_run_root / candidate).resolve(strict=False)
    if _is_relative_to(resolved, resolved_run_root):
        return resolved
    return None


def _configurable(config: Any | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, dict) else {}


def _is_relative_to(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _extract_data_workspace_from_state(state: Any) -> Path | None:
    for message in _state_messages(state):
        workspace = _extract_data_workspace_from_text(_message_text(message))
        if workspace is not None:
            return workspace.expanduser().resolve(strict=False)
    return None


def _extract_data_workspace_from_text(text: str) -> Path | None:
    json_match = _JSON_WORKSPACE_RE.search(text)
    if json_match:
        try:
            return Path(json.loads(f'"{json_match.group("path")}"'))
        except json.JSONDecodeError:
            return Path(_clean_workspace_path(json_match.group("path")))

    for pattern in (_QUOTED_WORKSPACE_RE, _PROSE_WORKSPACE_RE, _ASSIGNED_WORKSPACE_RE, _UNQUOTED_WORKSPACE_RE):
        match = pattern.search(text)
        if match:
            path = _clean_workspace_path(match.group("path"))
            candidate = Path(path)
            if candidate.is_absolute():
                return candidate
    return None


def _state_messages(state: Any) -> list[Any]:
    if isinstance(state, dict):
        messages = state.get("messages", [])
    else:
        messages = getattr(state, "messages", [])
    return list(messages or [])


def _message_text(message: Any) -> str:
    if isinstance(message, BaseMessage):
        return _content_to_text(message.content)
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content") or message.get("data", {}).get("content")
    return _content_to_text(content)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_to_text(item) for item in content)
    if isinstance(content, dict):
        if content.get("type") == "text" and "text" in content:
            return str(content["text"])
        if isinstance(content.get("content"), str):
            return str(content["content"])
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _clean_workspace_path(value: str) -> str:
    return value.strip().strip("`'\" ,")


def _json_success(**payload: Any) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False)


def _json_error(error: str) -> str:
    return json.dumps({"ok": False, "error": error}, ensure_ascii=False)


def _json_preview(value: Any, *, max_chars: int) -> tuple[str, bool]:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text, False
    if max_chars <= 3:
        return "." * max(0, max_chars), True
    return text[: max_chars - 3] + "...", True


def _compact_sql_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": result.get("available"),
        "mode": result.get("mode"),
        "planned_queries": result.get("planned_queries") or [],
        "executed_queries": result.get("executed_queries") or [],
        "results": result.get("results") or {},
        "error": result.get("error"),
    }


def _compact_rag_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": result.get("available"),
        "mode": result.get("mode"),
        "query": result.get("query"),
        "filters": result.get("filters") or {},
        "searched_sources": result.get("searched_sources") or [],
        "documents": [_compact_rag_document(doc) for doc in result.get("documents") or [] if isinstance(doc, dict)],
        "error": result.get("error"),
    }


def _compact_rag_document(doc: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "rank",
        "id",
        "source_table",
        "source_kind",
        "conference",
        "year",
        "paper_id",
        "review_id",
        "source_file",
        "title",
        "decision",
        "accepted",
        "score_numeric",
        "severity",
        "importance",
        "reviewer_satisfaction",
        "similarity",
        "distance",
        "matched_embedding_fields",
        "snippet",
    )
    return {key: doc.get(key) for key in keys if doc.get(key) is not None}


def _truncate_text(text: str, max_chars: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."
