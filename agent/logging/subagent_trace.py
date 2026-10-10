from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import BaseMessage, message_to_dict
from langgraph.config import get_config


_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|access[_-]?token|auth[_-]?token|bearer|secret|password|credential)",
    re.IGNORECASE,
)


def _runtime_config(runtime: Any | None) -> dict[str, Any]:
    config = getattr(runtime, "config", None)
    if isinstance(config, dict):
        return config
    # LangGraph's before/model hooks receive a bare Runtime; ToolRuntime is the
    # variant that carries .config. Both must resolve the same trusted assignment.
    try:
        return get_config()
    except RuntimeError:
        return {}


@dataclass(slots=True)
class _TraceContext:
    trace_id: str
    subagent_name: str
    run_root: Path
    parent_workspace: Path
    trace_dir: Path
    state_key: str
    started_at: str
    started_perf: float
    before_files: set[Path] = field(default_factory=set)
    event_count: int = 0


class SubAgentTraceMiddleware(AgentMiddleware):
    """Persist subagent model/tool events for later audit.

    DeepAgent's built-in `task` tool returns only a final subagent message to the
    parent agent. This middleware is attached to the subagent itself so its full
    internal trajectory is saved in the parent workspace.
    """

    def __init__(self, *, run_root: Path, subagent_name: str, system_prompt: str, trace_root: Path | None = None) -> None:
        super().__init__()
        self.run_root = run_root
        self.subagent_name = subagent_name
        self.system_prompt = system_prompt
        self.trace_root = trace_root
        self._lock = threading.RLock()
        self._active: dict[str, _TraceContext] = {}

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        key = self._state_key(state)
        initial_messages = self._state_messages(state)
        task_description = self._message_text(initial_messages[0]) if initial_messages else ""
        parent_workspace = self._resolve_parent_workspace(task_description, runtime)
        trace_id = self._new_trace_id()
        trace_dir = self._trace_dir(parent_workspace, trace_id, runtime)
        trace_dir.mkdir(parents=True, exist_ok=True)

        ctx = _TraceContext(
            trace_id=trace_id,
            subagent_name=self.subagent_name,
            run_root=self.run_root,
            parent_workspace=parent_workspace,
            trace_dir=trace_dir,
            state_key=key,
            started_at=self._now(),
            started_perf=perf_counter(),
            before_files=self._list_files(parent_workspace, trace_dir),
        )
        with self._lock:
            self._active[key] = ctx

        self._write_json(
            trace_dir / "subagent_input.json",
            {
                "trace_id": trace_id,
                "subagent_name": self.subagent_name,
                "parent_workspace": str(parent_workspace),
                "system_prompt": self.system_prompt,
                "messages": self._serialize_messages(initial_messages),
                "task_description": task_description,
            },
        )
        self._write_summary(ctx, status="running")
        self._write_event(
            ctx,
            {
                "event_type": "subagent_started",
                "subagent_name": self.subagent_name,
                "parent_workspace": str(parent_workspace),
                "message_count": len(initial_messages),
            },
        )
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        ctx = self._context_for_state(request.state)
        model_name = self._model_name(request.model)
        if ctx is not None:
            self._write_event(
                ctx,
                {
                    "event_type": "model_call_started",
                    "model": model_name,
                    "message_count": len(request.messages),
                    "system_message": self._message_summary(request.system_message),
                    "messages": [self._message_summary(message) for message in request.messages],
                    "tool_names": [self._tool_name(tool) for tool in request.tools],
                },
            )
        try:
            response = handler(request)
        except Exception as exc:
            if ctx is not None:
                self._write_event(
                    ctx,
                    {
                        "event_type": "model_call_failed",
                        "model": model_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            raise

        if ctx is not None:
            self._write_event(
                ctx,
                {
                    "event_type": "model_call_finished",
                    "model": model_name,
                    "messages": self._serialize_messages(response.result),
                    "structured_response": self._safe_data(response.structured_response),
                    "token_usage": self._extract_token_usage(response.result),
                    "finish_reasons": self._extract_finish_reasons(response.result),
                    "tool_calls": self._extract_tool_calls(response.result),
                },
            )
        return response

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]) -> Any:
        ctx = self._context_for_state(request.state)
        tool_call = request.tool_call
        tool_name = str(tool_call.get("name", "unknown"))
        tool_call_id = str(tool_call.get("id", ""))
        if ctx is not None:
            self._write_event(
                ctx,
                {
                    "event_type": "tool_call_started",
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "args": self._safe_data(tool_call.get("args", {})),
                },
            )
        try:
            result = handler(request)
        except Exception as exc:
            if ctx is not None:
                self._write_event(
                    ctx,
                    {
                        "event_type": "tool_call_failed",
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            raise

        if ctx is not None:
            self._write_event(
                ctx,
                {
                    "event_type": "tool_call_finished",
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "result": self._safe_data(result),
                },
            )
        return result

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        ctx = self._pop_context_for_state(state)
        if ctx is None:
            return None

        messages = self._state_messages(state)
        after_files = self._list_files(ctx.parent_workspace, ctx.trace_dir)
        created_files = sorted(str(path) for path in after_files - ctx.before_files)
        final_answer = self._final_ai_text(messages)
        ended_at = self._now()
        self._write_event(
            ctx,
            {
                "event_type": "subagent_finished",
                "created_file_count": len(created_files),
                "final_answer_preview": final_answer[:2000],
            },
        )
        self._write_json(
            ctx.trace_dir / "subagent_state.json",
            {
                "trace_id": ctx.trace_id,
                "subagent_name": ctx.subagent_name,
                "parent_workspace": str(ctx.parent_workspace),
                "created_files": created_files,
                "final_answer": final_answer,
                "messages": self._serialize_messages(messages),
            },
        )
        self._write_summary(ctx, status="completed", ended_at=ended_at, created_files=created_files)
        return None

    def _write_event(self, ctx: _TraceContext, event: dict[str, Any]) -> None:
        with self._lock:
            ctx.event_count += 1
            payload = {
                "ts": self._now(),
                "trace_id": ctx.trace_id,
                "event_index": ctx.event_count,
                **event,
            }
            with (ctx.trace_dir / "events.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(self._safe_data(payload), ensure_ascii=False) + "\n")

    def _write_summary(
        self,
        ctx: _TraceContext,
        *,
        status: str,
        ended_at: str | None = None,
        created_files: list[str] | None = None,
    ) -> None:
        summary = {
            "trace_id": ctx.trace_id,
            "subagent_name": ctx.subagent_name,
            "status": status,
            "parent_workspace": str(ctx.parent_workspace),
            "trace_dir": str(ctx.trace_dir),
            "started_at": ctx.started_at,
            "ended_at": ended_at,
            "duration_seconds": round(perf_counter() - ctx.started_perf, 3) if ended_at else None,
            "event_count": ctx.event_count,
            "created_files": created_files or [],
            "artifacts": {
                "subagent_input": str(ctx.trace_dir / "subagent_input.json"),
                "events": str(ctx.trace_dir / "events.jsonl"),
                "subagent_state": str(ctx.trace_dir / "subagent_state.json"),
                "trace_summary": str(ctx.trace_dir / "trace_summary.json"),
            },
        }
        self._write_json(ctx.trace_dir / "trace_summary.json", summary)

    def _new_trace_id(self) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{self.subagent_name}_{timestamp}_{uuid.uuid4().hex[:8]}"

    def _trace_dir(self, parent_workspace: Path, trace_id: str, runtime: Any | None = None) -> Path:
        if self.trace_root is not None:
            label = self._runtime_label(runtime) or self._workspace_label(parent_workspace) or "_unassigned"
            return self.trace_root / label / trace_id
        fallback_base = (self.run_root / "debug" / "tool_traces").resolve()
        label = self._runtime_label(runtime) or self._workspace_label(parent_workspace) or "_unassigned"
        return fallback_base / label / trace_id

    @staticmethod
    def _workspace_label(parent_workspace: Path) -> str | None:
        name = parent_workspace.name
        return name if name else None

    @staticmethod
    def _absolute_workspace_match(text: str) -> str | None:
        posix_match = re.search(
            r"(/[^\s\"'`]+/(?:workspace|artifacts/(?:data|external))/[^\s\"'`]+)",
            text,
        )
        if posix_match:
            return posix_match.group(1)
        windows_match = re.search(
            r"([A-Za-z]:\\[^\s\"'`]*(?:workspace|artifacts\\(?:data|external))\\[^\s\"'`]+)",
            text,
        )
        if windows_match:
            return windows_match.group(1)
        return None

    @staticmethod
    def _runtime_label(runtime: Any | None) -> str | None:
        config = _runtime_config(runtime)
        if not isinstance(config, dict):
            return None
        configurable = config.get("configurable")
        if not isinstance(configurable, dict):
            return None
        role = configurable.get("timeline_role")
        round_id = configurable.get("timeline_round")
        if not isinstance(role, str) or not role:
            return None
        if isinstance(round_id, int):
            return f"{role}_round_{round_id:02d}"
        return role

    def _resolve_parent_workspace(self, task_description: str, runtime: Any | None = None) -> Path:
        runtime_workspace = self._resolve_runtime_workspace(runtime)
        if runtime_workspace is not None:
            return runtime_workspace

        absolute_match = self._absolute_workspace_match(task_description)
        if absolute_match:
            workspace = self._normalize_workspace_candidate(self._clean_path(absolute_match))
            if workspace is not None:
                return workspace

        relative_match = re.search(r"\b((?:workspace|artifacts/(?:data|external))/[^\s\"'`]+)", task_description)
        if relative_match:
            workspace = self._normalize_workspace_candidate(
                self.run_root / self._clean_path(relative_match.group(1))
            )
            if workspace is not None:
                return workspace

        return (self.run_root / "debug" / "tool_traces" / "_unassigned").resolve()

    def _resolve_runtime_workspace(self, runtime: Any | None) -> Path | None:
        config = _runtime_config(runtime)
        if not isinstance(config, dict):
            return None
        configurable = config.get("configurable")
        if not isinstance(configurable, dict):
            return None

        if self.subagent_name == "external_researcher":
            abs_keys = ("assigned_external_workspace_abs", "assigned_data_workspace_abs")
            rel_keys = ("assigned_external_workspace", "assigned_data_workspace")
        else:
            abs_keys = ("assigned_data_workspace_abs", "assigned_external_workspace_abs")
            rel_keys = ("assigned_data_workspace", "assigned_external_workspace")

        for key in abs_keys:
            abs_workspace = configurable.get(key)
            if isinstance(abs_workspace, str) and abs_workspace.strip():
                workspace = self._normalize_workspace_candidate(Path(abs_workspace.strip()).expanduser())
                if workspace is not None:
                    return workspace

        for key in rel_keys:
            rel_workspace = configurable.get(key)
            if isinstance(rel_workspace, str) and rel_workspace.strip():
                rel_path = Path(rel_workspace.strip())
                if not rel_path.is_absolute():
                    workspace = self._normalize_workspace_candidate(self.run_root / rel_path)
                    if workspace is not None:
                        return workspace

        return None

    def _normalize_workspace_candidate(self, candidate: Path) -> Path | None:
        workspace_root = self.run_root.resolve(strict=False)
        resolved = candidate.resolve(strict=False)

        if resolved.is_file() or resolved.suffix:
            resolved = resolved.parent

        if resolved == workspace_root or workspace_root in resolved.parents:
            return resolved
        return None

    @staticmethod
    def _clean_path(path: str) -> Path:
        return Path(path.rstrip(".,;:)，。；）"))

    def _context_for_state(self, state: Any) -> _TraceContext | None:
        key = self._state_key(state)
        with self._lock:
            ctx = self._active.get(key)
            if ctx is not None:
                return ctx
            if len(self._active) == 1:
                return next(iter(self._active.values()))
            return None

    def _pop_context_for_state(self, state: Any) -> _TraceContext | None:
        key = self._state_key(state)
        with self._lock:
            ctx = self._active.pop(key, None)
            if ctx is not None:
                return ctx
            if len(self._active) == 1:
                _, only_ctx = self._active.popitem()
                return only_ctx
            return None

    def _state_key(self, state: Any) -> str:
        messages = self._state_messages(state)
        first_text = self._message_text(messages[0]) if messages else ""
        return hashlib.sha1(first_text.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _state_messages(state: Any) -> list[Any]:
        if isinstance(state, dict):
            messages = state.get("messages", [])
        else:
            messages = getattr(state, "messages", [])
        return list(messages or [])

    @staticmethod
    def _list_files(root: Path, trace_dir: Path) -> set[Path]:
        if not root.exists() or not root.is_dir():
            return set()
        resolved_trace = trace_dir.resolve()
        files: set[Path] = set()
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved == resolved_trace or resolved_trace in resolved.parents:
                continue
            files.add(resolved)
        return files

    @staticmethod
    def _model_name(model: Any) -> str:
        for attr in ("model_name", "model", "model_id", "deployment_name"):
            value = getattr(model, attr, None)
            if value:
                return str(value)
        return model.__class__.__name__

    @staticmethod
    def _tool_name(tool: Any) -> str:
        if isinstance(tool, dict):
            return str(tool.get("name") or tool.get("function", {}).get("name") or "unknown")
        return str(getattr(tool, "name", tool.__class__.__name__))

    @classmethod
    def _serialize_messages(cls, messages: Any) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for message in messages or []:
            try:
                serialized.append(cls._safe_data(message_to_dict(message)))
            except Exception:
                serialized.append({"repr": repr(message)})
        return serialized

    @classmethod
    def _message_summary(cls, message: Any) -> dict[str, Any] | None:
        if message is None:
            return None
        text = cls._message_text(message)
        return {
            "type": getattr(message, "type", message.__class__.__name__),
            "name": getattr(message, "name", None),
            "id": getattr(message, "id", None),
            "content_preview": text[:2000],
            "content_length": len(text),
        }

    @staticmethod
    def _message_text(message: Any) -> str:
        if isinstance(message, BaseMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        if item.get("type") == "text" and "text" in item:
                            parts.append(str(item["text"]))
                        elif isinstance(item.get("content"), str):
                            parts.append(str(item["content"]))
                    else:
                        parts.append(str(item))
                return "\n".join(parts)
            return str(content)
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str):
            return content
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)

    @classmethod
    def _final_ai_text(cls, messages: list[Any]) -> str:
        for message in reversed(messages):
            if getattr(message, "type", None) == "ai":
                text = cls._message_text(message)
                if text.strip():
                    return text
        return cls._message_text(messages[-1]) if messages else ""

    @classmethod
    def _extract_token_usage(cls, messages: list[Any]) -> list[dict[str, Any]]:
        usages: list[dict[str, Any]] = []
        for message in messages or []:
            usage = getattr(message, "usage_metadata", None)
            response_metadata = getattr(message, "response_metadata", None) or {}
            token_usage = response_metadata.get("token_usage") if isinstance(response_metadata, dict) else None
            if usage or token_usage:
                usages.append(
                    {
                        "id": getattr(message, "id", None),
                        "usage_metadata": cls._safe_data(usage),
                        "token_usage": cls._safe_data(token_usage),
                    }
                )
        return usages

    @staticmethod
    def _extract_finish_reasons(messages: list[Any]) -> list[Any]:
        reasons: list[Any] = []
        for message in messages or []:
            metadata = getattr(message, "response_metadata", None) or {}
            if isinstance(metadata, dict) and metadata.get("finish_reason") is not None:
                reasons.append(metadata.get("finish_reason"))
        return reasons

    @classmethod
    def _extract_tool_calls(cls, messages: list[Any]) -> list[Any]:
        calls: list[Any] = []
        for message in messages or []:
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                calls.extend(cls._safe_data(tool_calls))
        return calls

    @classmethod
    def _safe_data(cls, value: Any) -> Any:
        if isinstance(value, dict):
            safe: dict[str, Any] = {}
            for key, item in value.items():
                key_str = str(key)
                safe[key_str] = "[REDACTED]" if _SENSITIVE_KEY_RE.search(key_str) else cls._safe_data(item)
            return safe
        if isinstance(value, (list, tuple, set)):
            return [cls._safe_data(item) for item in value]
        if isinstance(value, BaseMessage):
            try:
                return cls._safe_data(message_to_dict(value))
            except Exception:
                return {"repr": repr(value)}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            json.dumps(value)
            return value
        except TypeError:
            return repr(value)

    @classmethod
    def _write_json(cls, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(cls._safe_data(data), fh, ensure_ascii=False, indent=2)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
