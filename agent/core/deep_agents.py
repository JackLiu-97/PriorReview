from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

from deepagents import (
    FilesystemPermission,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import CompositeBackend, FilesystemBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import message_to_dict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent.core.openai_http import make_openai_http_client
from agent.core.prompts import build_data_subagent_prompt, build_external_researcher_prompt, build_main_agent_prompt
from agent.core.rag_context import RAG_QUERY_CONTEXT
from agent.core.settings import Settings
from agent.core.sql_context import SQL_PLANNER_RULES, SQL_SCHEMA_CONTEXT
from agent.core.compiled_data_subagent import build_compiled_data_subagent
from agent.core.compiled_external_research_subagent import build_compiled_external_research_subagent
from agent.logging.artifact_manager import ArtifactManager
from agent.logging.execution_log import ExecutionLog
from agent.logging.run_logger import RunLogger
from agent.schemas.raw import AgentRawResult

ModelT = TypeVar("ModelT", bound=BaseModel)

DATA_SUBAGENT_WORKSPACE_ROOT = "artifacts/data"
EXTERNAL_SUBAGENT_WORKSPACE_ROOT = "artifacts/external"
DATA_SUBAGENT_DESCRIPTIONS = {
    "reviewer": (
        "Collect reviewer-side historical evidence from SQL and RAG: similar papers' review scores, "
        "recommendations, decisions, strengths, weaknesses, reviewer questions, and semantic priors from "
        "review_weakness/review_strength/rebuttal_pair retrieval."
    ),
    "author": (
        "Collect historical rebuttal and discussion evidence from SQL and RAG: reviewer concerns, author replies, "
        "AC/reviewer follow-up comments, reviewer satisfaction patterns, and rebuttal_pair examples for similar issues."
    ),
    "sc": (
        "Collect score-calibration evidence from SQL and RAG: historical rating/recommendation/decision distributions, "
        "similar weakness/strength priors, meta-review context, reviewer score summaries, and calibration examples "
        "for checking whether the current review score matches historical patterns."
    ),
    "finalizer": (
        "Collect SQL-only historical review style guidance from papers and reviews: similar-score and similar-field "
        "review phrasing, tone, organization, and cautious expression patterns. Style guidance must not change "
        "current review opinions, scores, confidence, or recommendation."
    ),
}
EXTERNAL_RESEARCHER_DESCRIPTION = (
    "Reviewer-only external API researcher for cutoff-safe Semantic Scholar, arXiv, and GitHub evidence. "
    "Use only for related-work, prior-art, repository, or implementation-context lookups before the configured cutoff date."
)
DEEPAGENTS_INTERNAL_WORKSPACE_ROOT = "debug/deepagents_internal"
MAIN_DEEP_AGENT_EXCLUDED_TOOLS = frozenset(
    {
        "execute",
        "read_file",
        "write_file",
        "edit_file",
        "ls",
        "glob",
        "grep",
    }
)
_RESTRICTED_HARNESS_PROFILE_KEYS: set[str] = set()


class DeepAgentExecutionError(RuntimeError):
    pass


class _MainToolFilterMiddleware(AgentMiddleware[Any, Any, Any]):
    """Hide restricted built-in tools from the main DeepAgent model request."""

    def __init__(self, excluded_tools: frozenset[str]) -> None:
        self.excluded_tools = excluded_tools

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(request.override(tools=self._filter_tools(request.tools)))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return await handler(request.override(tools=self._filter_tools(request.tools)))

    def _filter_tools(self, tools: list[Any]) -> list[Any]:
        return [tool for tool in tools if self._tool_name(tool) not in self.excluded_tools]

    @staticmethod
    def _tool_name(tool: Any) -> str | None:
        if isinstance(tool, dict):
            name = tool.get("name")
            return name if isinstance(name, str) else None
        name = getattr(tool, "name", None)
        return name if isinstance(name, str) else None


def _make_local_backend(*, run_root: Path, env: dict[str, str] | None = None) -> CompositeBackend:
    """Build a non-shell backend for DeepAgents internal virtual file storage."""

    del env  # Kept for compatibility with older tests/call sites.
    default_backend = FilesystemBackend(
        root_dir=run_root,
        virtual_mode=True,
    )
    internal_root = run_root / DEEPAGENTS_INTERNAL_WORKSPACE_ROOT
    conversation_backend = FilesystemBackend(
        root_dir=internal_root / "conversation_history",
        virtual_mode=True,
    )
    large_results_backend = FilesystemBackend(
        root_dir=internal_root / "large_tool_results",
        virtual_mode=True,
    )
    return CompositeBackend(
        default=default_backend,
        routes={
            "/conversation_history/": conversation_backend,
            "/large_tool_results/": large_results_backend,
        },
    )


def _register_restricted_harness_profile(model_name: str) -> None:
    """Disable DeepAgents' auto-added general-purpose subagent for this model."""

    key = model_name if ":" in model_name else f"openai:{model_name}"
    if key in _RESTRICTED_HARNESS_PROFILE_KEYS:
        return
    register_harness_profile(
        key,
        HarnessProfile(
            excluded_middleware=frozenset({TodoListMiddleware}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    _RESTRICTED_HARNESS_PROFILE_KEYS.add(key)


def _data_subagent_workspace_rel(agent_name: str, artifact_stem: str) -> str:
    return f"{DATA_SUBAGENT_WORKSPACE_ROOT}/{agent_name}_{artifact_stem}"


def _external_subagent_workspace_rel(agent_name: str, artifact_stem: str) -> str:
    return f"{EXTERNAL_SUBAGENT_WORKSPACE_ROOT}/{agent_name}_{artifact_stem}"


def _deny_all_filesystem_permissions() -> list[FilesystemPermission]:
    return [
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="deny",
        )
    ]


@dataclass(slots=True)
class DeepAgentRunArtifacts:
    """Artifacts emitted by a single Deep Agent execution."""

    workspace: str
    prompt_artifact: str
    input_artifact: str
    state_artifact: str
    stream_artifact: str | None
    created_files: list[str]

    @property
    def output_refs(self) -> list[str]:
        """Return every useful output path for logging."""

        refs = [self.prompt_artifact, self.state_artifact, *self.created_files]
        if self.stream_artifact:
            refs.append(self.stream_artifact)
        return refs


def _content_to_text(content: Any) -> str:
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
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _token_chunk_text(token: Any) -> str:
    return _content_to_text(getattr(token, "content", ""))


def _run_agent_stream(
    *,
    agent: Any,
    graph_input: dict[str, Any],
    runtime_config: dict[str, Any] | None,
    stream_path: Path,
) -> dict[str, Any]:
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    latest_state: dict[str, Any] | None = None
    with stream_path.open("w", encoding="utf-8") as stream_file:
        for event in agent.stream(
            graph_input,
            runtime_config,
            stream_mode=["messages", "values"],
        ):
            if not isinstance(event, tuple) or len(event) != 2:
                continue
            mode, data = event
            if mode == "messages":
                token = data[0] if isinstance(data, tuple) and data else data
                text = _token_chunk_text(token)
                if text:
                    stream_file.write(text)
                    stream_file.flush()
            elif mode == "values" and isinstance(data, dict):
                latest_state = data
    if latest_state is None:
        raise DeepAgentExecutionError("Streaming agent run completed without a final state.")
    return latest_state


class StructuredDeepAgent:
    """Thin wrapper around `create_deep_agent` for structured node execution."""

    def __init__(
        self,
        *,
        settings: Settings,
        artifact_manager: ArtifactManager,
        logger: RunLogger,
        execution_log: ExecutionLog | None = None,
        agent_name: str,
        response_model: type[ModelT],
        use_data_subagent: bool = False,
        use_external_researcher: bool = False,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        openai_model: str | None = None,
        temperature: float | None = None,
        use_default_openai_config: bool = True,
    ) -> None:
        if use_default_openai_config:
            resolved_api_key = openai_api_key if openai_api_key is not None else settings.openai_api_key
            resolved_base_url = openai_base_url if openai_base_url is not None else settings.openai_base_url
            resolved_model = openai_model if openai_model is not None else settings.openai_model
            resolved_temperature = temperature if temperature is not None else settings.temperature
        else:
            resolved_api_key = openai_api_key
            resolved_base_url = openai_base_url
            resolved_model = openai_model
            resolved_temperature = temperature if temperature is not None else settings.temperature
        if not resolved_api_key:
            raise DeepAgentExecutionError(f"{agent_name} OPENAI_API_KEY is missing. Configure .env before running the workflow.")
        if not resolved_model:
            raise DeepAgentExecutionError(f"{agent_name} OPENAI_MODEL is missing. Configure .env before running the workflow.")
        if use_external_researcher and agent_name != "reviewer":
            raise DeepAgentExecutionError("external_researcher is available only to the reviewer agent.")

        _register_restricted_harness_profile(resolved_model)

        self.settings = settings
        self.artifact_manager = artifact_manager
        self.logger = logger
        self.execution_log = execution_log
        self.agent_name = agent_name
        self.response_model = response_model
        self.use_data_subagent = use_data_subagent
        self.use_external_researcher = use_external_researcher
        self.run_root = artifact_manager.run_dir

        model_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "temperature": resolved_temperature,
            "api_key": resolved_api_key,
            "base_url": resolved_base_url,
            "use_responses_api": False,
            "streaming": True,
        }
        http_client = make_openai_http_client(resolved_base_url)
        if http_client is not None:
            model_kwargs["http_client"] = http_client
        self.model = ChatOpenAI(**model_kwargs)
        self.backend = _make_local_backend(run_root=self.run_root)
        self.agent = self._build_agent()

    def run(
        self,
        *,
        payload: dict[str, Any],
        artifact_stem: str,
        node_name: str,
        agent_role: str | None = None,
    ) -> tuple[ModelT, DeepAgentRunArtifacts]:
        """Execute the Deep Agent and persist its prompt, input, and final state."""

        run_label = f"{self.agent_name}_{artifact_stem}"
        workspace_rel = run_label
        data_workspace_rel: str | None = None
        data_workspace_abs: Path | None = None
        external_workspace_rel: str | None = None
        external_workspace_abs: Path | None = None
        if self.use_data_subagent:
            data_workspace_rel = _data_subagent_workspace_rel(self.agent_name, artifact_stem)
            data_workspace_abs = self.run_root / data_workspace_rel
        if self.use_external_researcher:
            external_workspace_rel = _external_subagent_workspace_rel(self.agent_name, artifact_stem)
            external_workspace_abs = self.run_root / external_workspace_rel

        # Only expose controlled subagent workspaces to agents with those delegation capabilities.
        payload_with_workspace = dict(payload)
        if data_workspace_abs is not None and data_workspace_rel is not None:
            payload_with_workspace.update(
                {
                    "assigned_data_workspace": data_workspace_rel,
                    "assigned_data_workspace_abs": str(data_workspace_abs.resolve()),
                }
            )
        if external_workspace_abs is not None and external_workspace_rel is not None:
            payload_with_workspace.update(
                {
                    "assigned_external_workspace": external_workspace_rel,
                    "assigned_external_workspace_abs": str(external_workspace_abs.resolve()),
                }
            )

        prompt_artifact = self.artifact_manager.write_json(
            f"debug/prompts/{run_label}.json",
            self._prompt_snapshot(),
        )
        self.logger.log(
            event_type="deep_agent_started",
            message=f"Running deep agent {self.agent_name}.",
            node_name=node_name,
            agent_role=agent_role,
            input_refs=[],
            metadata={"workspace": workspace_rel, "response_model": self.response_model.__name__},
        )
        before_files: set[Path] = set()
        if data_workspace_abs is not None:
            before_files |= self._list_workspace_files(data_workspace_abs)
        if external_workspace_abs is not None:
            before_files |= self._list_workspace_files(external_workspace_abs)
        stream_artifact = str(self.run_root / f"artifacts/outputs/{run_label}.stream.md")
        raw_state = _run_agent_stream(
            agent=self.agent,
            graph_input={
                "messages": [
                    {
                        "role": "user",
                        "content": self._format_user_payload(payload_with_workspace),
                    }
                ]
            },
            runtime_config=self._runtime_config(
                data_workspace_rel=data_workspace_rel,
                data_workspace_abs=data_workspace_abs,
                external_workspace_rel=external_workspace_rel,
                external_workspace_abs=external_workspace_abs,
                role=agent_role or self.agent_name,
                round_id=None,
                stage=artifact_stem,
            ),
            stream_path=Path(stream_artifact),
        )
        structured = raw_state.get("structured_response")
        structured_source = "native"
        if structured is None:
            result = self._parse_structured_response_from_messages(raw_state)
            structured_source = "message_json"
        else:
            result = structured if isinstance(structured, self.response_model) else self.response_model.model_validate(structured)
        after_files: set[Path] = set()
        if data_workspace_abs is not None:
            after_files |= self._list_workspace_files(data_workspace_abs)
        if external_workspace_abs is not None:
            after_files |= self._list_workspace_files(external_workspace_abs)
        created_files = sorted(str(path) for path in after_files - before_files)
        state_artifact = self.artifact_manager.write_json(
            f"debug/raw_messages/{run_label}.json",
            {
                "structured_response": result.model_dump(mode="json"),
                "structured_response_source": structured_source,
                "stream_artifact": stream_artifact,
                "created_files": created_files,
                "messages": self._serialize_messages(raw_state.get("messages", [])),
            },
        )
        self.logger.log(
            event_type="deep_agent_finished",
            message=f"Deep agent {self.agent_name} completed.",
            node_name=node_name,
            agent_role=agent_role,
            output_refs=[prompt_artifact, state_artifact, *created_files],
            metadata={"workspace": workspace_rel, "created_file_count": len(created_files)},
        )
        # DeepAgentRunArtifacts，这次执行的产物记录对象
        return result, DeepAgentRunArtifacts(
            workspace=workspace_rel,
            prompt_artifact=prompt_artifact,
            input_artifact="",
            state_artifact=state_artifact,
            stream_artifact=stream_artifact,
            created_files=created_files,
        )

    def _build_agent(self):
        """Create a role-specific Deep Agent graph once at runtime startup."""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "system_prompt": self._build_main_system_prompt(),
            "backend": self.backend,
            "permissions": _deny_all_filesystem_permissions(),
            "middleware": [_MainToolFilterMiddleware(MAIN_DEEP_AGENT_EXCLUDED_TOOLS)],
            "name": f"{self.agent_name}_deep_agent",
        }
        if self.agent_name != "finalizer":
            kwargs["response_format"] = self.response_model
        subagents: list[dict[str, Any]] = []
        if self.use_data_subagent:
            data_subagent_prompt = self._build_data_subagent_prompt()
            subagents.append(
                {
                    "name": "data_specialist",
                    "description": DATA_SUBAGENT_DESCRIPTIONS[self.agent_name],
                    "runnable": build_compiled_data_subagent(
                        model=self.model,
                        settings=self.settings,
                        run_root=self.run_root,
                        agent_name=self.agent_name,
                        system_prompt=data_subagent_prompt,
                        execution_log=self.execution_log,
                    ),
                }
            )
        if self.use_external_researcher:
            external_subagent_prompt = self._build_external_researcher_prompt()
            subagents.append(
                {
                    "name": "external_researcher",
                    "description": EXTERNAL_RESEARCHER_DESCRIPTION,
                    "runnable": build_compiled_external_research_subagent(
                        model=self.model,
                        settings=self.settings,
                        run_root=self.run_root,
                        agent_name=self.agent_name,
                        system_prompt=external_subagent_prompt,
                        execution_log=self.execution_log,
                    ),
                }
            )
        if subagents:
            kwargs["subagents"] = subagents
        return create_deep_agent(**kwargs)

    def _parse_structured_response_from_messages(self, raw_state: dict[str, Any]) -> ModelT:
        text = RawDeepAgent._extract_final_ai_text(raw_state.get("messages", []))
        parsed = self._extract_json_object(text)
        if parsed is None:
            raise DeepAgentExecutionError(
                f"Deep agent {self.agent_name} returned no structured_response and no JSON object "
                f"could be extracted from the final message. Available keys: {sorted(raw_state.keys())}"
            )
        parsed = self._strip_extra_model_fields(parsed, self.response_model)
        try:
            return self.response_model.model_validate(parsed)
        except Exception as exc:
            raise DeepAgentExecutionError(
                f"Deep agent {self.agent_name} returned no structured_response, and extracted JSON "
                f"did not validate as {self.response_model.__name__}: {exc}"
            ) from exc

    @staticmethod
    def _extract_json_object(text: str) -> Any | None:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    @classmethod
    def _strip_extra_model_fields(cls, value: Any, model_type: type[BaseModel]) -> Any:
        if not isinstance(value, dict):
            return value
        cleaned: dict[str, Any] = {}
        for field_name, field_info in model_type.model_fields.items():
            if field_name not in value:
                continue
            cleaned[field_name] = cls._strip_extra_value(value[field_name], field_info.annotation)
        return cleaned

    @classmethod
    def _strip_extra_value(cls, value: Any, annotation: Any) -> Any:
        origin = get_origin(annotation)
        if origin is list:
            args = get_args(annotation)
            item_type = args[0] if args else Any
            if isinstance(value, list):
                return [cls._strip_extra_value(item, item_type) for item in value]
            return value
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return cls._strip_extra_model_fields(value, annotation)
        return value

    def _build_main_system_prompt(self) -> str:
        """Return the main system prompt for this agent role."""

        return build_main_agent_prompt(
            self.agent_name,
            use_data_subagent=self.use_data_subagent,
            use_external_researcher=self.use_external_researcher,
        )

    def _build_data_subagent_prompt(self) -> str:
        """Return the role-specific prompt for the delegated data specialist."""

        return build_data_subagent_prompt(
            self.agent_name,
            sql_schema_context=SQL_SCHEMA_CONTEXT,
            sql_planner_rules=SQL_PLANNER_RULES,
            rag_query_context=RAG_QUERY_CONTEXT,
            code_execution_enabled=self.settings.data_agent_code_execution_enabled,
        )

    def _build_external_researcher_prompt(self) -> str:
        return build_external_researcher_prompt()

    def _prompt_snapshot(self) -> dict[str, Any]:
        """Persist the exact prompt text used for reproducible debugging."""

        snapshot = {"main_system_prompt": self._build_main_system_prompt()}
        if self.use_data_subagent:
            snapshot["data_subagent_system_prompt"] = self._build_data_subagent_prompt()
            snapshot["data_subagent_tracing_enabled"] = True
            snapshot["data_subagent_code_execution_enabled"] = self.settings.data_agent_code_execution_enabled
        if self.use_external_researcher:
            snapshot["external_researcher_system_prompt"] = self._build_external_researcher_prompt()
            snapshot["external_subagent_tracing_enabled"] = True
        return snapshot

    @staticmethod
    def _format_user_payload(payload: dict[str, Any]) -> str:
        return "Task payload:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _runtime_config(
        *,
        data_workspace_rel: str | None,
        data_workspace_abs: Path | None,
        external_workspace_rel: str | None,
        external_workspace_abs: Path | None,
        role: str,
        round_id: int | None,
        stage: str,
    ) -> dict[str, Any] | None:
        if data_workspace_rel is None and external_workspace_rel is None:
            return None
        configurable: dict[str, Any] = {
            "timeline_role": role,
            "timeline_round": round_id,
            "timeline_stage": stage,
        }
        if data_workspace_rel is not None:
            configurable["assigned_data_workspace"] = data_workspace_rel
        if data_workspace_abs is not None:
            configurable["assigned_data_workspace_abs"] = str(data_workspace_abs.resolve())
        if external_workspace_rel is not None:
            configurable["assigned_external_workspace"] = external_workspace_rel
        if external_workspace_abs is not None:
            configurable["assigned_external_workspace_abs"] = str(external_workspace_abs.resolve())
        return {"configurable": configurable}

    @staticmethod
    def _list_workspace_files(workspace_abs: Path) -> set[Path]:
        if not workspace_abs.exists():
            return set()
        return {path.resolve() for path in workspace_abs.rglob("*") if path.is_file()}

    @staticmethod
    def _serialize_messages(messages: Any) -> list[dict[str, Any]]:
        """Convert LangChain messages into JSON-serializable records."""

        serialized: list[dict[str, Any]] = []
        for message in messages or []:
            try:
                serialized.append(message_to_dict(message))
            except Exception:
                serialized.append({"repr": repr(message)})
        return serialized


class RawDeepAgent:
    """Deep Agent wrapper for loose intermediate outputs."""

    def __init__(
        self,
        *,
        settings: Settings,
        artifact_manager: ArtifactManager,
        logger: RunLogger,
        execution_log: ExecutionLog | None = None,
        agent_name: str,
        use_data_subagent: bool = False,
        use_external_researcher: bool = False,
    ) -> None:
        if not settings.openai_api_key:
            raise DeepAgentExecutionError("OPENAI_API_KEY is missing. Configure .env before running the workflow.")
        if use_external_researcher and agent_name != "reviewer":
            raise DeepAgentExecutionError("external_researcher is available only to the reviewer agent.")

        _register_restricted_harness_profile(settings.openai_model)

        self.settings = settings
        self.artifact_manager = artifact_manager
        self.logger = logger
        self.execution_log = execution_log
        self.agent_name = agent_name
        self.use_data_subagent = use_data_subagent
        self.use_external_researcher = use_external_researcher
        self.run_root = artifact_manager.run_dir

        model_kwargs: dict[str, Any] = {
            "model": settings.openai_model,
            "temperature": settings.temperature,
            "api_key": settings.openai_api_key,
            "base_url": settings.openai_base_url,
            "use_responses_api": False,
            "streaming": True,
        }
        http_client = make_openai_http_client(settings.openai_base_url)
        if http_client is not None:
            model_kwargs["http_client"] = http_client
        self.model = ChatOpenAI(**model_kwargs)
        self.backend = _make_local_backend(run_root=self.run_root)
        self.agent = self._build_agent()

    def run(
        self,
        *,
        payload: dict[str, Any],
        artifact_stem: str,
        node_name: str,
        agent_role: str,
        round_id: int,
        stage: str,
    ) -> tuple[AgentRawResult, DeepAgentRunArtifacts]:
        """Execute a Deep Agent and persist its raw final text."""

        run_label = f"{self.agent_name}_{artifact_stem}"
        workspace_rel = run_label
        data_workspace_rel: str | None = None
        data_workspace_abs: Path | None = None
        external_workspace_rel: str | None = None
        external_workspace_abs: Path | None = None
        if self.use_data_subagent:
            data_workspace_rel = _data_subagent_workspace_rel(self.agent_name, artifact_stem)
            data_workspace_abs = self.run_root / data_workspace_rel
        if self.use_external_researcher:
            external_workspace_rel = _external_subagent_workspace_rel(self.agent_name, artifact_stem)
            external_workspace_abs = self.run_root / external_workspace_rel
        # Only expose controlled subagent workspaces to agents with those delegation capabilities.
        payload_with_workspace = dict(payload)
        if data_workspace_abs is not None and data_workspace_rel is not None:
            payload_with_workspace.update(
                {
                    "assigned_data_workspace": data_workspace_rel,
                    "assigned_data_workspace_abs": str(data_workspace_abs.resolve()),
                }
            )
        if external_workspace_abs is not None and external_workspace_rel is not None:
            payload_with_workspace.update(
                {
                    "assigned_external_workspace": external_workspace_rel,
                    "assigned_external_workspace_abs": str(external_workspace_abs.resolve()),
                }
            )

        prompt_artifact = self.artifact_manager.write_json(
            f"debug/prompts/{run_label}.json",
            self._prompt_snapshot(),
        )
        self.logger.log(
            event_type="deep_agent_started",
            message=f"Running raw deep agent {self.agent_name}.",
            node_name=node_name,
            agent_role=agent_role,
            input_refs=[],
            metadata={"workspace": workspace_rel, "response_model": "raw"},
        )

        before_files: set[Path] = set()
        if data_workspace_abs is not None:
            before_files |= self._list_workspace_files(data_workspace_abs)
        if external_workspace_abs is not None:
            before_files |= self._list_workspace_files(external_workspace_abs)
        stream_artifact = str(self.run_root / f"artifacts/outputs/{run_label}.stream.md")
        raw_state = _run_agent_stream(
            agent=self.agent,
            graph_input={
                "messages": [
                    {
                        "role": "user",
                        "content": self._format_user_payload(payload_with_workspace),
                    }
                ]
            },
            runtime_config=self._runtime_config(
                data_workspace_rel=data_workspace_rel,
                data_workspace_abs=data_workspace_abs,
                external_workspace_rel=external_workspace_rel,
                external_workspace_abs=external_workspace_abs,
                role=agent_role,
                round_id=round_id,
                stage=stage,
            ),
            stream_path=Path(stream_artifact),
        )
        messages = raw_state.get("messages", [])
        content = self._extract_final_ai_text(messages)
        after_files: set[Path] = set()
        if data_workspace_abs is not None:
            after_files |= self._list_workspace_files(data_workspace_abs)
        if external_workspace_abs is not None:
            after_files |= self._list_workspace_files(external_workspace_abs)
        created_files = sorted(str(path) for path in after_files - before_files)
        state_artifact = self.artifact_manager.write_json(
            f"debug/raw_messages/{run_label}.json",
            {
                "raw_content": content,
                "stream_artifact": stream_artifact,
                "created_files": created_files,
                "messages": self._serialize_messages(messages),
            },
        )
        result_artifact_refs = [
            self.execution_log.rel(path) if self.execution_log is not None else path
            for path in created_files
        ]
        result_state_ref = self.execution_log.rel(state_artifact) if self.execution_log is not None else state_artifact
        result = AgentRawResult(
            role=agent_role,
            round_id=round_id,
            stage=stage,
            content=content,
            artifact_refs=result_artifact_refs,
            state_artifact=result_state_ref,
        )
        self.logger.log(
            event_type="deep_agent_finished",
            message=f"Raw deep agent {self.agent_name} completed.",
            node_name=node_name,
            agent_role=agent_role,
            output_refs=[prompt_artifact, state_artifact, *created_files],
            metadata={"workspace": workspace_rel, "created_file_count": len(created_files)},
        )
        return result, DeepAgentRunArtifacts(
            workspace=workspace_rel,
            prompt_artifact=prompt_artifact,
            input_artifact="",
            state_artifact=state_artifact,
            stream_artifact=stream_artifact,
            created_files=created_files,
        )

    def _build_agent(self):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system_prompt": self._build_main_system_prompt(),
            "backend": self.backend,
            "permissions": _deny_all_filesystem_permissions(),
            "middleware": [_MainToolFilterMiddleware(MAIN_DEEP_AGENT_EXCLUDED_TOOLS)],
            "name": f"{self.agent_name}_deep_agent",
        }
        subagents: list[dict[str, Any]] = []
        if self.use_data_subagent:
            data_subagent_prompt = self._build_data_subagent_prompt()
            subagents.append(
                {
                    "name": "data_specialist",
                    "description": DATA_SUBAGENT_DESCRIPTIONS[self.agent_name],
                    "runnable": build_compiled_data_subagent(
                        model=self.model,
                        settings=self.settings,
                        run_root=self.run_root,
                        agent_name=self.agent_name,
                        system_prompt=data_subagent_prompt,
                        execution_log=self.execution_log,
                    ),
                }
            )
        if self.use_external_researcher:
            external_subagent_prompt = self._build_external_researcher_prompt()
            subagents.append(
                {
                    "name": "external_researcher",
                    "description": EXTERNAL_RESEARCHER_DESCRIPTION,
                    "runnable": build_compiled_external_research_subagent(
                        model=self.model,
                        settings=self.settings,
                        run_root=self.run_root,
                        agent_name=self.agent_name,
                        system_prompt=external_subagent_prompt,
                        execution_log=self.execution_log,
                    ),
                }
            )
        if subagents:
            kwargs["subagents"] = subagents
        return create_deep_agent(**kwargs)

    def _build_main_system_prompt(self) -> str:
        return build_main_agent_prompt(
            self.agent_name,
            use_data_subagent=self.use_data_subagent,
            use_external_researcher=self.use_external_researcher,
        )

    def _build_data_subagent_prompt(self) -> str:
        return build_data_subagent_prompt(
            self.agent_name,
            sql_schema_context=SQL_SCHEMA_CONTEXT,
            sql_planner_rules=SQL_PLANNER_RULES,
            rag_query_context=RAG_QUERY_CONTEXT,
            code_execution_enabled=self.settings.data_agent_code_execution_enabled,
        )

    def _build_external_researcher_prompt(self) -> str:
        return build_external_researcher_prompt()

    def _prompt_snapshot(self) -> dict[str, Any]:
        snapshot = {"main_system_prompt": self._build_main_system_prompt()}
        if self.use_data_subagent:
            snapshot["data_subagent_system_prompt"] = self._build_data_subagent_prompt()
            snapshot["data_subagent_tracing_enabled"] = True
            snapshot["data_subagent_code_execution_enabled"] = self.settings.data_agent_code_execution_enabled
        if self.use_external_researcher:
            snapshot["external_researcher_system_prompt"] = self._build_external_researcher_prompt()
            snapshot["external_subagent_tracing_enabled"] = True
        return snapshot

    @staticmethod
    def _extract_final_ai_text(messages: Any) -> str:
        for message in reversed(messages or []):
            if getattr(message, "type", None) != "ai":
                continue
            text = RawDeepAgent._content_to_text(getattr(message, "content", ""))
            if text.strip():
                return text
        if messages:
            return RawDeepAgent._content_to_text(getattr(messages[-1], "content", ""))
        return ""

    @staticmethod
    def _content_to_text(content: Any) -> str:
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
            return "\n".join(parts)
        return str(content) if content is not None else ""

    @staticmethod
    def _format_user_payload(payload: dict[str, Any]) -> str:
        return "Task payload:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _runtime_config(
        *,
        data_workspace_rel: str | None,
        data_workspace_abs: Path | None,
        external_workspace_rel: str | None,
        external_workspace_abs: Path | None,
        role: str,
        round_id: int | None,
        stage: str,
    ) -> dict[str, Any] | None:
        if data_workspace_rel is None and external_workspace_rel is None:
            return None
        configurable: dict[str, Any] = {
            "timeline_role": role,
            "timeline_round": round_id,
            "timeline_stage": stage,
        }
        if data_workspace_rel is not None:
            configurable["assigned_data_workspace"] = data_workspace_rel
        if data_workspace_abs is not None:
            configurable["assigned_data_workspace_abs"] = str(data_workspace_abs.resolve())
        if external_workspace_rel is not None:
            configurable["assigned_external_workspace"] = external_workspace_rel
        if external_workspace_abs is not None:
            configurable["assigned_external_workspace_abs"] = str(external_workspace_abs.resolve())
        return {"configurable": configurable}

    @staticmethod
    def _list_workspace_files(workspace_abs: Path) -> set[Path]:
        if not workspace_abs.exists():
            return set()
        return {path.resolve() for path in workspace_abs.rglob("*") if path.is_file()}

    @staticmethod
    def _serialize_messages(messages: Any) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for message in messages or []:
            try:
                serialized.append(message_to_dict(message))
            except Exception:
                serialized.append({"repr": repr(message)})
        return serialized
