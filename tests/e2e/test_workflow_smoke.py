from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from openai import InternalServerError

from agent.core.deep_agents import (
    DEEPAGENTS_INTERNAL_WORKSPACE_ROOT,
    MAIN_DEEP_AGENT_EXCLUDED_TOOLS,
    DeepAgentRunArtifacts,
    RawDeepAgent,
    StructuredDeepAgent,
    _make_local_backend,
    _register_restricted_harness_profile,
)
from agent.core.compiled_data_subagent import (
    SQL_RESULTS_PREVIEW_MAX_CHARS,
    ControlledDataSubagentTools,
    build_data_subagent_tools,
)
from agent.core.prompts import build_data_subagent_prompt, build_external_researcher_prompt, build_main_agent_prompt
from agent.core.runtime import WorkflowRuntime
from agent.core.settings import Settings
from agent.logging.artifact_manager import ArtifactManager
from agent.logging.run_logger import RunLogger
from agent.logging.subagent_trace import SubAgentTraceMiddleware
from agent.schemas.common import SQLQueryPlan
from agent.schemas.final_report import FinalReviewReport
from agent.schemas.paper import ParsedPaper
from agent.schemas.raw import AgentRawResult
from agent.tools.external_research import build_external_research_tools
from agent.workflows.auto_review_graph import run_review_workflow


SEEN_DISCUSSION_HISTORIES: list[dict] = []
SEEN_PAYLOADS: list[dict] = []
SEEN_AGENT_RUNS: list[dict] = []
SC_STATUS_QUEUE: list[str | None] = []
RAW_AGENT_FAILURE_QUEUE: list[Exception] = []
RAW_AGENT_ATTEMPTS: list[dict] = []
FINALIZER_FAILURE_QUEUE: list[Exception] = []
FINALIZER_ATTEMPTS: list[dict] = []


def _openai_internal_server_error(
    *,
    status_code: int = 502,
    body: dict | None = None,
    headers: dict | None = None,
) -> InternalServerError:
    response = httpx.Response(
        status_code,
        json=body or {},
        headers=headers or {},
        request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
    )
    return InternalServerError(
        f"Error code: {status_code}",
        response=response,
        body=body,
    )


def _has_forbidden_history_key(value, forbidden_keys: set[str]) -> bool:  # noqa: ANN001
    if isinstance(value, dict):
        return bool(forbidden_keys.intersection(value)) or any(
            _has_forbidden_history_key(item, forbidden_keys) for item in value.values()
        )
    if isinstance(value, list):
        return any(_has_forbidden_history_key(item, forbidden_keys) for item in value)
    return False


def _fake_raw_deep_agent_run(
    self,
    *,
    payload,
    artifact_stem,
    node_name,
    agent_role,
    round_id,
    stage,
):  # noqa: ANN001
    SEEN_PAYLOADS.append(json.loads(json.dumps(payload)))
    SEEN_AGENT_RUNS.append({"agent_name": self.agent_name, "stage": stage})
    SEEN_DISCUSSION_HISTORIES.append(
        {
            "agent_name": self.agent_name,
            "history": json.loads(json.dumps(payload.get("discussion_history", []))),
        }
    )
    workspace_rel = f"{self.agent_name}_{artifact_stem}"
    state_artifact = self.artifact_manager.write_json(
        f"debug/raw_messages/{workspace_rel}.json",
        {"agent_name": self.agent_name, "node_name": node_name},
    )
    summary_artifact = self.artifact_manager.write_text(
        f"artifacts/data/{workspace_rel}/data_summary.md",
        f"Synthetic artifacts for {self.agent_name}.",
    )

    content_md = payload["paper_markdown"]
    first_heading = next(
        (
            line.lstrip("#").strip()
            for line in content_md.splitlines()
            if line.strip().startswith("#")
        ),
        "paper",
    )
    paper_title = first_heading or "paper"
    if self.agent_name == "sc":
        sc_status = SC_STATUS_QUEUE.pop(0) if SC_STATUS_QUEUE else "NORMAL"
        if sc_status is None:
            raw_content = (
                f"## {stage}\n\n"
                "Calibration notes intentionally omit the required status marker."
            )
        else:
            raw_content = (
                f"SC_STATUS: {sc_status}\n\n"
                f"## {stage}\n\n"
                f"{paper_title} receives score calibration notes. "
                "The SC agent does not rewrite reviewer scores directly."
            )
    else:
        raw_content = (
            f"## {stage}\n\n"
            f"{paper_title} receives loose Markdown notes from {self.agent_name}. "
            "This intentionally is not strict JSON."
        )
    result = AgentRawResult(
        role=agent_role,
        round_id=round_id,
        stage=stage,
        content=raw_content,
        artifact_refs=[summary_artifact],
        state_artifact=state_artifact,
    )

    return result, DeepAgentRunArtifacts(
        workspace=workspace_rel,
        prompt_artifact=state_artifact,
        input_artifact=state_artifact,
        state_artifact=state_artifact,
        stream_artifact=None,
        created_files=[summary_artifact],
    )


def _flaky_raw_deep_agent_run(
    self,
    *,
    payload,
    artifact_stem,
    node_name,
    agent_role,
    round_id,
    stage,
):  # noqa: ANN001
    RAW_AGENT_ATTEMPTS.append({"agent_name": self.agent_name, "stage": stage})
    if RAW_AGENT_FAILURE_QUEUE:
        raise RAW_AGENT_FAILURE_QUEUE.pop(0)
    return _fake_raw_deep_agent_run(
        self,
        payload=payload,
        artifact_stem=artifact_stem,
        node_name=node_name,
        agent_role=agent_role,
        round_id=round_id,
        stage=stage,
    )


def _fake_structured_deep_agent_run(self, *, payload, artifact_stem, node_name, agent_role=None):  # noqa: ANN001
    SEEN_PAYLOADS.append(json.loads(json.dumps(payload)))
    SEEN_AGENT_RUNS.append({"agent_name": self.agent_name, "stage": "finalizer"})
    workspace_rel = f"{self.agent_name}_{artifact_stem}"
    state_artifact = self.artifact_manager.write_json(
        f"debug/raw_messages/{workspace_rel}.json",
        {"agent_name": self.agent_name, "node_name": node_name},
    )
    report = self.response_model.model_validate(
        {
            "review_type": "official_review",
            "conference_style": "ICLR_2026",
            "paper_id": payload["paper_id"],
            "title": payload["title"],
            "summary": "Sample Retrieval Paper presents a workflow for retrieval-enhanced peer review.",
            "strengths": ["Clear problem statement.", "Coherent workflow."],
            "weaknesses": ["Evaluation details remain limited."],
            "questions": ["Which ablation best isolates the contribution?"],
            "required_clarifications": ["Clarify baseline choice."],
            "recommendation": "Borderline Accept",
            "recommendation_rationale": "The finalizer synthesized loose intermediate notes into a valid report.",
            "supporting_arguments": ["The raw reviewer notes support the main conclusion."],
            "additional_feedback": ["The author response partially addresses the concerns."],
            "code_of_ethics": {"potential_violation": False, "explanation": None},
            "scores": {
                "rating": 6,
                "confidence": 3,
            },
            "discussion_history_summary": "Formatted from source reviewer anchor and marked discussion history.",
            "sc_meta_summary": "",
            "evidence_refs": [
                {
                    "source": "source_review_anchor",
                    "snippet": payload["source_review_anchor"]["content"][:120],
                    "reason": "Grounds the formatted final review.",
                }
            ],
            "output_path": None,
        }
    )
    return report, DeepAgentRunArtifacts(
        workspace=workspace_rel,
        prompt_artifact=state_artifact,
        input_artifact=state_artifact,
        state_artifact=state_artifact,
        stream_artifact=None,
        created_files=[],
    )


def _flaky_structured_deep_agent_run(self, *, payload, artifact_stem, node_name, agent_role=None):  # noqa: ANN001
    FINALIZER_ATTEMPTS.append({"agent_name": self.agent_name, "stage": "finalizer"})
    if FINALIZER_FAILURE_QUEUE:
        raise FINALIZER_FAILURE_QUEUE.pop(0)
    return _fake_structured_deep_agent_run(
        self,
        payload=payload,
        artifact_stem=artifact_stem,
        node_name=node_name,
        agent_role=agent_role,
    )


def _capture_create_deep_agent(**kwargs):  # noqa: ANN003
    _capture_create_deep_agent.calls.append(kwargs)
    return object()


_capture_create_deep_agent.calls = []


class _CapturingStream:
    """Expose the streaming interface used by the current agent wrappers."""

    def stream(self, state, config=None, *, stream_mode):  # noqa: ANN001
        assert stream_mode == ["messages", "values"]
        final_state = self.invoke(state, config)
        for message in final_state.get("messages", []):
            yield "messages", (message, {})
        yield "values", final_state


class _CapturingRawAgent(_CapturingStream):
    def __init__(self) -> None:
        self.state: dict | None = None
        self.config: dict | None = None

    def invoke(self, state, config=None):  # noqa: ANN001
        self.state = state
        self.config = config
        return {"messages": [AIMessage(content="Captured raw result.")]}


class _CapturingStructuredAgent(_CapturingStream):
    def __init__(self, structured_response: AgentRawResult) -> None:
        self.state: dict | None = None
        self.config: dict | None = None
        self.structured_response = structured_response

    def invoke(self, state, config=None):  # noqa: ANN001
        self.state = state
        self.config = config
        return {
            "structured_response": self.structured_response,
            "messages": [AIMessage(content="Captured structured result.")],
        }


class _CapturingMessageOnlyAgent(_CapturingStream):
    def __init__(self, content: str) -> None:
        self.state: dict | None = None
        self.config: dict | None = None
        self.content = content

    def invoke(self, state, config=None):  # noqa: ANN001
        self.state = state
        self.config = config
        return {"messages": [AIMessage(content=self.content)], "files": {}}


def _data_state(data_workspace: Path) -> dict:
    return {
        "messages": [
            HumanMessage(
                content=(
                    "Data query assignment.\n"
                    f'assigned_data_workspace_abs: "{data_workspace}"\n'
                    "Only use controlled SQL/RAG tools."
                )
            )
        ]
    }


class FakeExecutionLog:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, **event) -> None:  # noqa: ANN003
        self.events.append(event)

    def rel(self, path: Path) -> str:
        return str(path)


class FakeSQLRepository:
    def __init__(self, settings):  # noqa: ANN001
        self.settings = settings

    def execute_query_plan(self, plan: SQLQueryPlan, *, max_rows: int = 20) -> dict:
        query = plan.queries[0]
        return {
            "available": True,
            "mode": "fake",
            "planned_queries": [query.model_dump(mode="json")],
            "executed_queries": [
                {
                    "name": query.name,
                    "reason": query.reason,
                    "sql": query.sql,
                    "row_count": 1,
                    "error": None,
                }
            ],
            "results": {query.name: [{"answer": 1, "note": "历史评审"}]},
            "error": None,
        }


class FakeLargeSQLRepository:
    def __init__(self, settings):  # noqa: ANN001
        self.settings = settings

    def execute_query_plan(self, plan: SQLQueryPlan, *, max_rows: int = 20) -> dict:
        query = plan.queries[0]
        return {
            "available": True,
            "mode": "fake",
            "planned_queries": [query.model_dump(mode="json")],
            "executed_queries": [
                {
                    "name": query.name,
                    "reason": query.reason,
                    "sql": query.sql,
                    "row_count": 1,
                    "error": None,
                }
            ],
            "results": {query.name: [{"review_text": "历史评审片段" * 1000}]},
            "error": None,
        }


class FakeFailingSQLRepository:
    def __init__(self, settings):  # noqa: ANN001
        self.settings = settings

    def execute_query_plan(self, plan: SQLQueryPlan, *, max_rows: int = 20) -> dict:
        query = plan.queries[0]
        return {
            "available": True,
            "mode": "fake",
            "planned_queries": [query.model_dump(mode="json")],
            "executed_queries": [
                {
                    "name": query.name,
                    "reason": query.reason,
                    "sql": query.sql,
                    "row_count": 0,
                    "error": 'column "p.conference" must appear in the GROUP BY clause',
                }
            ],
            "results": {query.name: []},
            "error": None,
        }


class FakeRAGRepository:
    def __init__(self, settings):  # noqa: ANN001
        self.settings = settings

    def query(self, query: str, **kwargs) -> dict:  # noqa: ANN003
        return {
            "available": True,
            "mode": "fake",
            "query": query,
            "documents": [
                {
                    "rank": 1,
                    "title": "Memory in RL Prior",
                    "source_table": "review_weakness",
                    "decision": "Accept",
                    "similarity": 0.91,
                    "snippet": "Relevant prior evidence about memory, POMDPs, and recurrent policy evaluation.",
                }
            ],
            "error": None,
        }


class WorkflowSmokeTest(unittest.TestCase):
    def _run_sample_workflow(self, tmpdir: str, **settings_overrides):
        sample_paper = Path(tmpdir) / "sample_retrieval_paper.md"
        sample_paper.write_text(
            "# Sample Retrieval Paper\n\nThis paper studies retrieval-enhanced peer review.",
            encoding="utf-8",
        )
        settings = Settings.from_env(
            openai_api_key="test-key",
            openai_base_url="https://example.com/v1",
            openai_model="test-model",
            finalizer_openai_api_key="finalizer-test-key",
            finalizer_openai_base_url="https://finalizer.example.com/v1",
            finalizer_openai_model="finalizer-test-model",
            results_dir=tmpdir,
            **settings_overrides,
        )
        report = run_review_workflow(str(sample_paper), settings=settings)
        return report, Path(report.output_path or "")

    def test_finalizer_agent_uses_finalizer_openai_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_paper = Path(tmpdir) / "sample_retrieval_paper.md"
            sample_paper.write_text("# Sample Retrieval Paper\n\nBody.", encoding="utf-8")
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                finalizer_openai_api_key="finalizer-test-key",
                finalizer_openai_base_url="https://finalizer.example.com/v1",
                finalizer_openai_model="finalizer-test-model",
                finalizer_temperature=0.1,
                results_dir=tmpdir,
                console_log_enabled=False,
            )
            _capture_create_deep_agent.calls.clear()
            with patch("agent.core.deep_agents.create_deep_agent", new=_capture_create_deep_agent):
                runtime = WorkflowRuntime(settings, sample_paper)

            finalizer_model = runtime.finalizer_agent.model
            self.assertEqual(finalizer_model.model_name, "finalizer-test-model")
            self.assertEqual(finalizer_model.openai_api_base, "https://finalizer.example.com/v1")
            self.assertEqual(finalizer_model.temperature, 0.1)
            self.assertEqual(finalizer_model.openai_api_key.get_secret_value(), "finalizer-test-key")
            self.assertEqual(
                runtime.execution_log.models,
                {"main": "test-model", "finalizer": "finalizer-test-model"},
            )
            self.assertFalse(runtime.reviewer_agent.use_external_researcher)
            self.assertFalse(runtime.author_agent.use_external_researcher)
            self.assertFalse(runtime.sc_agent.use_external_researcher)
            reviewer_call = _capture_create_deep_agent.calls[0]
            self.assertEqual([spec["name"] for spec in reviewer_call["subagents"]], ["data_specialist"])
            author_call = _capture_create_deep_agent.calls[1]
            self.assertEqual([spec["name"] for spec in author_call["subagents"]], ["data_specialist"])
            sc_call = _capture_create_deep_agent.calls[2]
            self.assertEqual([spec["name"] for spec in sc_call["subagents"]], ["data_specialist"])
            self.assertNotIn("external_researcher", reviewer_call["system_prompt"])
            self.assertFalse(runtime.finalizer_agent.use_data_subagent)
            finalizer_call = _capture_create_deep_agent.calls[3]
            self.assertNotIn("subagents", finalizer_call)
            self.assertNotIn("Data specialist usage guide", finalizer_call["system_prompt"])
            self.assertNotIn("assigned_data_workspace_abs", finalizer_call["system_prompt"])

    def test_finalizer_writes_report_without_data_delegation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_paper = Path(tmpdir) / "sample.md"
            sample_paper.write_text("# Sample\n\nBody.", encoding="utf-8")
            settings = Settings(
                openai_api_key="test-key",
                openai_model="test-model",
                finalizer_openai_api_key="finalizer-test-key",
                finalizer_openai_model="finalizer-test-model",
                results_dir=tmpdir,
                console_log_enabled=False,
                reviewer_data_agent_enabled=False,
                author_data_agent_enabled=False,
                sc_data_agent_enabled=False,
            )
            with (
                patch("agent.core.deep_agents.create_deep_agent", new=_capture_create_deep_agent),
                patch(
                    "agent.core.deep_agents.build_compiled_data_subagent",
                    side_effect=AssertionError("Finalization must not build a data specialist"),
                ),
            ):
                runtime = WorkflowRuntime(settings, sample_paper)

            expected = FinalReviewReport(
                paper_id="sample",
                title="Sample",
                summary="A study of retrieval-enhanced review.",
                recommendation="Borderline Accept",
                recommendation_rationale="The review supports this recommendation.",
                scores={"rating": 6, "confidence": 3},
            )
            capture = _CapturingMessageOnlyAgent(expected.model_dump_json())
            runtime.finalizer_agent.agent = capture
            review = AgentRawResult(
                role="reviewer", round_id=2, stage="revision_review",
                content="Final review: borderline accept; rating 6; confidence 3.",
            )
            report = runtime.finalize_output(
                parsed_paper=ParsedPaper(paper_id="sample", raw_markdown="# Sample\n\nBody."),
                final_review=review,
                discussion_history=[],
            )

            payload = json.loads(capture.state["messages"][0]["content"].removeprefix("Task payload:\n"))
            self.assertEqual(payload["source_review_anchor"]["content"], review.content)
            self.assertIn("discussion_history", payload)
            self.assertNotIn("assigned_data_workspace", payload)
            self.assertNotIn("assigned_data_workspace_abs", payload)
            self.assertIsNone(capture.config)
            self.assertFalse((runtime.run_dir / "artifacts" / "data").exists())
            self.assertEqual(report.scores, expected.scores)
            self.assertEqual(report.recommendation, expected.recommendation)
            saved = FinalReviewReport.model_validate_json(Path(report.output_path).read_text(encoding="utf-8"))
            self.assertEqual(saved, report)
            self.assertTrue((runtime.run_dir / "artifacts" / "outputs" / "finalizer.md").exists())

    def test_role_data_agent_switches_disable_subagents_prompts_and_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_paper = Path(tmpdir) / "sample_retrieval_paper.md"
            sample_paper.write_text("# Sample Retrieval Paper\n\nBody.", encoding="utf-8")
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                finalizer_openai_api_key="finalizer-test-key",
                finalizer_openai_base_url="https://finalizer.example.com/v1",
                finalizer_openai_model="finalizer-test-model",
                results_dir=tmpdir,
                console_log_enabled=False,
                reviewer_data_agent_enabled=False,
                author_data_agent_enabled=False,
                sc_data_agent_enabled=False,
            )
            _capture_create_deep_agent.calls.clear()
            with patch("agent.core.deep_agents.create_deep_agent", new=_capture_create_deep_agent):
                runtime = WorkflowRuntime(settings, sample_paper)

            self.assertFalse(runtime.reviewer_agent.use_data_subagent)
            self.assertFalse(runtime.author_agent.use_data_subagent)
            self.assertFalse(runtime.sc_agent.use_data_subagent)
            self.assertFalse(runtime.finalizer_agent.use_data_subagent)

            for call in _capture_create_deep_agent.calls:
                self.assertNotIn("subagents", call)
                self.assertNotIn("Data Specialist Usage Guide", call["system_prompt"])
                self.assertNotIn("assigned_data_workspace_abs", call["system_prompt"])

            parsed = ParsedPaper(paper_id="sample", raw_markdown="# Sample Retrieval Paper\n\nBody.")
            review = AgentRawResult(
                role="reviewer",
                round_id=1,
                stage="initial_review",
                content="Reviewer output.",
                artifact_refs=[],
                state_artifact=None,
            )
            rebuttal = AgentRawResult(
                role="author",
                round_id=1,
                stage="author_rebuttal",
                content="Author output.",
                artifact_refs=[],
                state_artifact=None,
            )
            reviewer_payload = runtime._build_reviewer_payload(
                parsed_paper=parsed,
                round_id=1,
                review_stage="initial_review",
                latest_review=None,
                latest_rebuttal=None,
                latest_sc_decision=None,
                discussion_history=[],
            )
            author_payload = runtime._build_author_payload(
                parsed_paper=parsed,
                latest_review=review,
                round_id=1,
                discussion_history=[],
            )
            sc_payload = runtime._build_score_calibration_payload(
                parsed_paper=parsed,
                latest_review=review,
                latest_rebuttal=rebuttal,
                round_id=1,
                discussion_history=[],
            )
            for payload in (reviewer_payload, author_payload, sc_payload):
                objective = payload["round_context"]["historical_objective"]
                self.assertIn("Do not use a historical SQL/RAG data agent", objective)
                self.assertNotIn("Collect historical", objective)
                self.assertNotIn("historical priors", objective)

            for agent, role, stage in (
                (runtime.reviewer_agent, "reviewer", "initial_review"),
                (runtime.author_agent, "author", "author_rebuttal"),
                (runtime.sc_agent, "sc", "score_calibration"),
            ):
                capture = _CapturingRawAgent()
                agent.agent = capture
                _, artifacts = agent.run(
                    payload={"paper_markdown": "# Sample Retrieval Paper\n\nBody.", "discussion_history": []},
                    artifact_stem="round_01",
                    node_name=f"{role}_round",
                    agent_role=role,
                    round_id=1,
                    stage=stage,
                )
                self.assertIsNotNone(capture.state)
                raw_payload = capture.state["messages"][0]["content"].removeprefix("Task payload:\n")
                payload = json.loads(raw_payload)
                self.assertNotIn("assigned_data_workspace", payload)
                self.assertNotIn("assigned_data_workspace_abs", payload)
                self.assertIsNone(capture.config)
                prompt_snapshot = json.loads(Path(artifacts.prompt_artifact).read_text(encoding="utf-8"))
                self.assertNotIn("data_subagent_system_prompt", prompt_snapshot)
                self.assertNotIn("data_subagent_tracing_enabled", prompt_snapshot)

    @patch.object(StructuredDeepAgent, "run", new=_fake_structured_deep_agent_run)
    @patch.object(RawDeepAgent, "run", new=_fake_raw_deep_agent_run)
    def test_workflow_generates_json_and_log(self) -> None:
        SEEN_DISCUSSION_HISTORIES.clear()
        SEEN_PAYLOADS.clear()
        SEEN_AGENT_RUNS.clear()
        FINALIZER_FAILURE_QUEUE.clear()
        FINALIZER_ATTEMPTS.clear()
        SC_STATUS_QUEUE.clear()
        SC_STATUS_QUEUE.append("NORMAL")
        with tempfile.TemporaryDirectory() as tmpdir:
            report, output_path = self._run_sample_workflow(tmpdir)
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.name, "sample_retrieval_paper.json")
            self.assertEqual(output_path.parent.parent.name, "runs")
            self.assertEqual(report.title, "Sample Retrieval Paper")
            run_json_path = output_path.parent / "run.json"
            timeline_path = output_path.parent / "timeline.jsonl"
            transcript_path = output_path.parent / "transcript.md"
            self.assertTrue(run_json_path.exists())
            self.assertTrue(timeline_path.exists())
            self.assertTrue(transcript_path.exists())
            self.assertFalse((output_path.parent / "state_final.json").exists())
            self.assertFalse((output_path.parent / "sample_retrieval_paper.log.jsonl").exists())
            self.assertIn("recommendation", output_path.read_text(encoding="utf-8"))
            run_meta = json.loads(run_json_path.read_text(encoding="utf-8"))
            self.assertEqual(run_meta["status"], "completed")
            self.assertEqual(
                run_meta["models"],
                {"main": "test-model", "finalizer": "finalizer-test-model"},
            )
            self.assertEqual(run_meta["final_output"], "sample_retrieval_paper.json")
            timeline = [
                json.loads(line)
                for line in timeline_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([event["seq"] for event in timeline], list(range(1, len(timeline) + 1)))
            self.assertEqual(timeline[0]["event"], "run_start")
            self.assertEqual(timeline[-1]["event"], "run_end")
            self.assertIn("Reviewer Round 01", transcript_path.read_text(encoding="utf-8"))
            self.assertTrue((output_path.parent / "artifacts" / "inputs" / "paper_snapshot.md").exists())
            self.assertTrue((output_path.parent / "artifacts" / "outputs" / "reviewer_round_01.md").exists())
            self.assertTrue((output_path.parent / "artifacts" / "outputs" / "finalizer.md").exists())

            self.assertEqual(
                [item["agent_name"] for item in SEEN_DISCUSSION_HISTORIES],
                ["reviewer", "author", "reviewer", "sc"],
            )
            self.assertEqual(
                SEEN_AGENT_RUNS,
                [
                    {"agent_name": "reviewer", "stage": "initial_review"},
                    {"agent_name": "author", "stage": "author_rebuttal"},
                    {"agent_name": "reviewer", "stage": "revision_review"},
                    {"agent_name": "sc", "stage": "score_calibration"},
                    {"agent_name": "finalizer", "stage": "finalizer"},
                ],
            )
            self.assertEqual(
                [len(item["history"]) for item in SEEN_DISCUSSION_HISTORIES],
                [0, 1, 2, 3],
            )
            self.assertEqual(
                [entry["role"] for entry in SEEN_DISCUSSION_HISTORIES[-1]["history"]],
                ["reviewer", "author", "reviewer"],
            )
            self.assertTrue(SEEN_PAYLOADS)
            raw_payloads = [
                payload
                for payload, run in zip(SEEN_PAYLOADS, SEEN_AGENT_RUNS, strict=True)
                if run["agent_name"] != "finalizer"
            ]
            for payload in raw_payloads:
                self.assertIn("paper_markdown", payload)
                self.assertNotIn("paper_content", payload)
                self.assertNotIn("shared_context", payload)
                self.assertEqual(payload["paper_markdown"].count("Sample Retrieval Paper"), 1)
                if payload["round_context"].get("review_stage"):
                    public_policy = payload["round_context"]["public_output_policy"]
                    self.assertIn("official initial review", public_policy)
                    self.assertIn("Do not mention rebuttal", public_policy)
                    self.assertIn("author response", public_policy)
                    self.assertIn("workflow history", public_policy)
            finalizer_payload = SEEN_PAYLOADS[-1]
            self.assertEqual(finalizer_payload["paper_id"], "sample_retrieval_paper")
            self.assertEqual(finalizer_payload["title"], "Sample Retrieval Paper")
            self.assertIn("discussion_history", finalizer_payload)
            self.assertIn("source_review_anchor", finalizer_payload)
            self.assertEqual(finalizer_payload["source_review_anchor"]["role"], "reviewer")
            self.assertEqual(finalizer_payload["source_review_anchor"]["stage"], "revision_review")
            self.assertIn("revision_review", finalizer_payload["source_review_anchor"]["content"])
            self.assertEqual(
                [entry["role"] for entry in finalizer_payload["discussion_history"]],
                ["reviewer", "author", "reviewer", "sc"],
            )
            self.assertEqual(
                [entry["authority"] for entry in finalizer_payload["discussion_history"]],
                ["context_only", "context_only", "source_review", "context_only"],
            )
            self.assertEqual(
                [entry["index"] for entry in finalizer_payload["discussion_history"]],
                [1, 2, 3, 4],
            )
            self.assertNotIn("final_review_content", finalizer_payload)
            self.assertNotIn("paper_markdown", finalizer_payload)
            self.assertNotIn("author_handoff", finalizer_payload)
            self.assertNotIn("sc_handoff", finalizer_payload)

            agent_output_events = [event for event in timeline if event["event"] == "agent_output"]
            self.assertEqual(
                [(event["role"], event["name"]) for event in agent_output_events],
                [
                    ("reviewer", "initial_review"),
                    ("author", "author_rebuttal"),
                    ("reviewer", "revision_review"),
                    ("sc", "score_calibration"),
                    ("finalizer", "finalizer"),
                ],
            )
            self.assertFalse((output_path.parent / "discussion_history.json").exists())

    @patch.object(StructuredDeepAgent, "run", new=_fake_structured_deep_agent_run)
    @patch.object(RawDeepAgent, "run", new=_flaky_raw_deep_agent_run)
    def test_raw_agent_retries_until_success(self) -> None:
        SEEN_DISCUSSION_HISTORIES.clear()
        SEEN_PAYLOADS.clear()
        SEEN_AGENT_RUNS.clear()
        RAW_AGENT_FAILURE_QUEUE.clear()
        RAW_AGENT_ATTEMPTS.clear()
        FINALIZER_FAILURE_QUEUE.clear()
        FINALIZER_ATTEMPTS.clear()
        RAW_AGENT_FAILURE_QUEUE.extend(
            [
                RuntimeError("temporary reviewer failure 1"),
                RuntimeError("temporary reviewer failure 2"),
            ]
        )
        SC_STATUS_QUEUE.clear()
        SC_STATUS_QUEUE.append("NORMAL")
        with tempfile.TemporaryDirectory() as tmpdir:
            report, output_path = self._run_sample_workflow(tmpdir)
            self.assertTrue(output_path.exists())
            self.assertEqual(report.title, "Sample Retrieval Paper")
            self.assertEqual(
                RAW_AGENT_ATTEMPTS[:3],
                [
                    {"agent_name": "reviewer", "stage": "initial_review"},
                    {"agent_name": "reviewer", "stage": "initial_review"},
                    {"agent_name": "reviewer", "stage": "initial_review"},
                ],
            )

            timeline = [
                json.loads(line)
                for line in (output_path.parent / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            retry_events = [event for event in timeline if event["event"] == "agent_retry"]
            self.assertEqual([event["role"] for event in retry_events], ["reviewer", "reviewer"])
            self.assertEqual([event["name"] for event in retry_events], ["initial_review", "initial_review"])
            self.assertEqual([event["output"]["attempt"] for event in retry_events], [1, 2])
            self.assertTrue(all(event["output"]["max_attempts"] == 3 for event in retry_events))
            self.assertTrue(all(event["output"]["retryable_http_error"] is False for event in retry_events))
            self.assertEqual(json.loads((output_path.parent / "run.json").read_text(encoding="utf-8"))["status"], "completed")

    @patch.object(StructuredDeepAgent, "run", new=_fake_structured_deep_agent_run)
    @patch.object(RawDeepAgent, "run", new=_flaky_raw_deep_agent_run)
    def test_raw_agent_waits_before_retrying_retryable_http_error(self) -> None:
        SEEN_DISCUSSION_HISTORIES.clear()
        SEEN_PAYLOADS.clear()
        SEEN_AGENT_RUNS.clear()
        RAW_AGENT_FAILURE_QUEUE.clear()
        RAW_AGENT_ATTEMPTS.clear()
        FINALIZER_FAILURE_QUEUE.clear()
        FINALIZER_ATTEMPTS.clear()
        RAW_AGENT_FAILURE_QUEUE.append(
            _openai_internal_server_error(body={"retryable": True, "retry_after": 60})
        )
        SC_STATUS_QUEUE.clear()
        SC_STATUS_QUEUE.append("NORMAL")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agent.core.runtime.time.sleep") as sleep:
                report, output_path = self._run_sample_workflow(tmpdir)

            self.assertEqual(report.title, "Sample Retrieval Paper")
            sleep.assert_called_once_with(60)
            timeline = [
                json.loads(line)
                for line in (output_path.parent / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            retry_events = [event for event in timeline if event["event"] == "agent_retry"]
            self.assertEqual(len(retry_events), 1)
            self.assertEqual(retry_events[0]["role"], "reviewer")
            self.assertTrue(retry_events[0]["output"]["retryable_http_error"])
            self.assertEqual(retry_events[0]["output"]["sleep_seconds"], 60)

    @patch.object(StructuredDeepAgent, "run", new=_fake_structured_deep_agent_run)
    @patch.object(RawDeepAgent, "run", new=_flaky_raw_deep_agent_run)
    def test_raw_agent_retry_exhaustion_records_error(self) -> None:
        SEEN_DISCUSSION_HISTORIES.clear()
        SEEN_PAYLOADS.clear()
        SEEN_AGENT_RUNS.clear()
        RAW_AGENT_FAILURE_QUEUE.clear()
        RAW_AGENT_ATTEMPTS.clear()
        FINALIZER_FAILURE_QUEUE.clear()
        FINALIZER_ATTEMPTS.clear()
        RAW_AGENT_FAILURE_QUEUE.extend(
            [
                RuntimeError("reviewer failed 1"),
                RuntimeError("reviewer failed 2"),
                RuntimeError("reviewer failed 3"),
                RuntimeError("reviewer should not run 4"),
            ]
        )
        SC_STATUS_QUEUE.clear()
        SC_STATUS_QUEUE.append("NORMAL")
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "reviewer failed 3"):
                self._run_sample_workflow(tmpdir)
            self.assertEqual(len(RAW_AGENT_ATTEMPTS), 3)
            self.assertEqual(len(RAW_AGENT_FAILURE_QUEUE), 1)

            run_dir = next((Path(tmpdir) / "sample_retrieval_paper" / "runs").iterdir())
            run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_meta["status"], "failed")
            timeline = [
                json.loads(line)
                for line in (run_dir / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            retry_events = [event for event in timeline if event["event"] == "agent_retry"]
            self.assertEqual([event["role"] for event in retry_events], ["reviewer", "reviewer"])
            self.assertEqual([event["output"]["attempt"] for event in retry_events], [1, 2])
            reviewer_errors = [
                event
                for event in timeline
                if event["event"] == "agent_error" and event["role"] == "reviewer"
            ]
            self.assertEqual(len(reviewer_errors), 1)
            self.assertEqual(reviewer_errors[0]["output"]["error"], "reviewer failed 3")

    @patch.object(StructuredDeepAgent, "run", new=_flaky_structured_deep_agent_run)
    @patch.object(RawDeepAgent, "run", new=_fake_raw_deep_agent_run)
    def test_finalizer_retries_until_success(self) -> None:
        SEEN_DISCUSSION_HISTORIES.clear()
        SEEN_PAYLOADS.clear()
        SEEN_AGENT_RUNS.clear()
        FINALIZER_FAILURE_QUEUE.clear()
        FINALIZER_ATTEMPTS.clear()
        FINALIZER_FAILURE_QUEUE.extend(
            [
                RuntimeError("temporary finalizer failure 1"),
                RuntimeError("temporary finalizer failure 2"),
            ]
        )
        SC_STATUS_QUEUE.clear()
        SC_STATUS_QUEUE.append("NORMAL")
        with tempfile.TemporaryDirectory() as tmpdir:
            report, output_path = self._run_sample_workflow(tmpdir)
            self.assertTrue(output_path.exists())
            self.assertEqual(report.title, "Sample Retrieval Paper")
            self.assertEqual(len(FINALIZER_ATTEMPTS), 3)

            timeline = [
                json.loads(line)
                for line in (output_path.parent / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            retry_events = [event for event in timeline if event["event"] == "agent_retry"]
            self.assertEqual([event["output"]["attempt"] for event in retry_events], [1, 2])
            self.assertTrue(all(event["role"] == "finalizer" for event in retry_events))
            self.assertTrue(all(event["output"]["max_attempts"] == 3 for event in retry_events))
            self.assertTrue(all("profile" not in event["output"] for event in retry_events))
            self.assertTrue(all(event["output"]["retryable_http_error"] is False for event in retry_events))
            fallback_events = [event for event in timeline if event["event"] == "agent_fallback"]
            self.assertEqual(fallback_events, [])
            finalizer_outputs = [
                event
                for event in timeline
                if event["event"] == "agent_output" and event["role"] == "finalizer"
            ]
            self.assertEqual(len(finalizer_outputs), 1)

    @patch.object(StructuredDeepAgent, "run", new=_flaky_structured_deep_agent_run)
    @patch.object(RawDeepAgent, "run", new=_fake_raw_deep_agent_run)
    def test_finalizer_waits_before_retrying_retryable_http_error(self) -> None:
        SEEN_DISCUSSION_HISTORIES.clear()
        SEEN_PAYLOADS.clear()
        SEEN_AGENT_RUNS.clear()
        FINALIZER_FAILURE_QUEUE.clear()
        FINALIZER_ATTEMPTS.clear()
        FINALIZER_FAILURE_QUEUE.append(
            _openai_internal_server_error(body={"retryable": True, "retry_after": 60})
        )
        SC_STATUS_QUEUE.clear()
        SC_STATUS_QUEUE.append("NORMAL")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agent.core.runtime.time.sleep") as sleep:
                report, output_path = self._run_sample_workflow(tmpdir)

            self.assertEqual(report.title, "Sample Retrieval Paper")
            sleep.assert_called_once_with(60)
            timeline = [
                json.loads(line)
                for line in (output_path.parent / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            retry_events = [event for event in timeline if event["event"] == "agent_retry"]
            self.assertEqual(len(retry_events), 1)
            self.assertEqual(retry_events[0]["role"], "finalizer")
            self.assertTrue(retry_events[0]["output"]["retryable_http_error"])
            self.assertEqual(retry_events[0]["output"]["sleep_seconds"], 60)

    @patch.object(StructuredDeepAgent, "run", new=_flaky_structured_deep_agent_run)
    @patch.object(RawDeepAgent, "run", new=_fake_raw_deep_agent_run)
    def test_finalizer_retry_exhaustion_fails_without_fallback(self) -> None:
        SEEN_DISCUSSION_HISTORIES.clear()
        SEEN_PAYLOADS.clear()
        SEEN_AGENT_RUNS.clear()
        FINALIZER_FAILURE_QUEUE.clear()
        FINALIZER_ATTEMPTS.clear()
        FINALIZER_FAILURE_QUEUE.extend(
            [
                RuntimeError("finalizer failed 1"),
                RuntimeError("finalizer failed 2"),
                RuntimeError("finalizer failed 3"),
            ]
        )
        SC_STATUS_QUEUE.clear()
        SC_STATUS_QUEUE.append("NORMAL")
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "finalizer failed 3"):
                self._run_sample_workflow(tmpdir)
            self.assertEqual(len(FINALIZER_ATTEMPTS), 3)
            self.assertEqual(len(FINALIZER_FAILURE_QUEUE), 0)

            run_dir = next((Path(tmpdir) / "sample_retrieval_paper" / "runs").iterdir())
            run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_meta["status"], "failed")
            self.assertEqual(run_meta["error"]["message"], "finalizer failed 3")
            timeline = [
                json.loads(line)
                for line in (run_dir / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            retry_events = [event for event in timeline if event["event"] == "agent_retry"]
            self.assertEqual([event["output"]["attempt"] for event in retry_events], [1, 2])
            self.assertTrue(all("profile" not in event["output"] for event in retry_events))
            fallback_events = [event for event in timeline if event["event"] == "agent_fallback"]
            self.assertEqual(fallback_events, [])
            finalizer_errors = [
                event
                for event in timeline
                if event["event"] == "agent_error" and event["role"] == "finalizer"
            ]
            self.assertEqual(len(finalizer_errors), 1)
            self.assertEqual(finalizer_errors[0]["output"]["error"], "finalizer failed 3")

    @patch.object(StructuredDeepAgent, "run", new=_flaky_structured_deep_agent_run)
    @patch.object(RawDeepAgent, "run", new=_fake_raw_deep_agent_run)
    def test_finalizer_retry_exhaustion_records_error_once(self) -> None:
        SEEN_DISCUSSION_HISTORIES.clear()
        SEEN_PAYLOADS.clear()
        SEEN_AGENT_RUNS.clear()
        FINALIZER_FAILURE_QUEUE.clear()
        FINALIZER_ATTEMPTS.clear()
        FINALIZER_FAILURE_QUEUE.extend(
            [
                RuntimeError("finalizer failed 1"),
                RuntimeError("finalizer failed 2"),
                RuntimeError("finalizer failed 3"),
                RuntimeError("finalizer should not run 4"),
            ]
        )
        SC_STATUS_QUEUE.clear()
        SC_STATUS_QUEUE.append("NORMAL")
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "finalizer failed 3"):
                self._run_sample_workflow(tmpdir)
            self.assertEqual(len(FINALIZER_ATTEMPTS), 3)
            self.assertEqual(len(FINALIZER_FAILURE_QUEUE), 1)

            run_dir = next((Path(tmpdir) / "sample_retrieval_paper" / "runs").iterdir())
            run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_meta["status"], "failed")
            timeline = [
                json.loads(line)
                for line in (run_dir / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            retry_events = [event for event in timeline if event["event"] == "agent_retry"]
            self.assertEqual([event["output"]["attempt"] for event in retry_events], [1, 2])
            self.assertTrue(all("profile" not in event["output"] for event in retry_events))
            fallback_events = [event for event in timeline if event["event"] == "agent_fallback"]
            self.assertEqual(fallback_events, [])
            finalizer_errors = [
                event
                for event in timeline
                if event["event"] == "agent_error" and event["role"] == "finalizer"
            ]
            self.assertEqual(len(finalizer_errors), 1)
            self.assertEqual(finalizer_errors[0]["output"]["error"], "finalizer failed 3")

    @patch.object(StructuredDeepAgent, "run", new=_fake_structured_deep_agent_run)
    @patch.object(RawDeepAgent, "run", new=_fake_raw_deep_agent_run)
    def test_score_calibration_revision_goes_once_to_reviewer_then_finalizer(self) -> None:
        SEEN_DISCUSSION_HISTORIES.clear()
        SEEN_PAYLOADS.clear()
        SEEN_AGENT_RUNS.clear()
        FINALIZER_FAILURE_QUEUE.clear()
        FINALIZER_ATTEMPTS.clear()
        SC_STATUS_QUEUE.clear()
        SC_STATUS_QUEUE.append("NEEDS_REVIEWER_REVISION")
        with tempfile.TemporaryDirectory() as tmpdir:
            report, output_path = self._run_sample_workflow(tmpdir)
            self.assertTrue(output_path.exists())
            self.assertEqual(report.title, "Sample Retrieval Paper")

            self.assertEqual(
                SEEN_AGENT_RUNS,
                [
                    {"agent_name": "reviewer", "stage": "initial_review"},
                    {"agent_name": "author", "stage": "author_rebuttal"},
                    {"agent_name": "reviewer", "stage": "revision_review"},
                    {"agent_name": "sc", "stage": "score_calibration"},
                    {"agent_name": "reviewer", "stage": "verification_review"},
                    {"agent_name": "finalizer", "stage": "finalizer"},
                ],
            )
            self.assertEqual([len(item["history"]) for item in SEEN_DISCUSSION_HISTORIES], [0, 1, 2, 3, 4])
            verification_payload = SEEN_PAYLOADS[-2]
            self.assertEqual(verification_payload["round_context"]["review_stage"], "verification_review")
            self.assertEqual(verification_payload["sc_handoff"]["stage"], "score_calibration")
            self.assertIn("SC_STATUS: NEEDS_REVIEWER_REVISION", verification_payload["sc_handoff"]["content"])
            self.assertEqual(SEEN_PAYLOADS[-1]["source_review_anchor"]["stage"], "verification_review")
            self.assertIn("verification_review", SEEN_PAYLOADS[-1]["source_review_anchor"]["content"])

            timeline = [
                json.loads(line)
                for line in (output_path.parent / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            history = [event for event in timeline if event["event"] == "agent_output" and event["role"] != "finalizer"]
            self.assertEqual(
                [entry["name"] for entry in history],
                [
                    "initial_review",
                    "author_rebuttal",
                    "revision_review",
                    "score_calibration",
                    "verification_review",
                ],
            )

    @patch.object(StructuredDeepAgent, "run", new=_fake_structured_deep_agent_run)
    @patch.object(RawDeepAgent, "run", new=_fake_raw_deep_agent_run)
    def test_reviewer_author_round_limit_can_skip_rebuttal_loop(self) -> None:
        SEEN_DISCUSSION_HISTORIES.clear()
        SEEN_PAYLOADS.clear()
        SEEN_AGENT_RUNS.clear()
        SC_STATUS_QUEUE.clear()
        SC_STATUS_QUEUE.append("NORMAL")
        with tempfile.TemporaryDirectory() as tmpdir:
            report, output_path = self._run_sample_workflow(tmpdir, max_reviewer_author_rounds=1)
            self.assertTrue(output_path.exists())
            self.assertEqual(report.title, "Sample Retrieval Paper")
            self.assertEqual(
                SEEN_AGENT_RUNS,
                [
                    {"agent_name": "reviewer", "stage": "initial_review"},
                    {"agent_name": "sc", "stage": "score_calibration"},
                    {"agent_name": "finalizer", "stage": "finalizer"},
                ],
            )
            self.assertEqual(SEEN_PAYLOADS[-1]["source_review_anchor"]["stage"], "initial_review")
            self.assertIn("initial_review", SEEN_PAYLOADS[-1]["source_review_anchor"]["content"])
            self.assertNotIn("revision_review", SEEN_PAYLOADS[-1]["source_review_anchor"]["content"])

    @patch.object(StructuredDeepAgent, "run", new=_fake_structured_deep_agent_run)
    @patch.object(RawDeepAgent, "run", new=_fake_raw_deep_agent_run)
    def test_score_calibration_rereview_limit_allows_second_sc_pass(self) -> None:
        SEEN_DISCUSSION_HISTORIES.clear()
        SEEN_PAYLOADS.clear()
        SEEN_AGENT_RUNS.clear()
        SC_STATUS_QUEUE.clear()
        SC_STATUS_QUEUE.extend(["NEEDS_REVIEWER_REVISION", "NORMAL"])
        with tempfile.TemporaryDirectory() as tmpdir:
            report, output_path = self._run_sample_workflow(tmpdir, max_sc_rereview_rounds=2)
            self.assertTrue(output_path.exists())
            self.assertEqual(report.title, "Sample Retrieval Paper")
            self.assertEqual(
                SEEN_AGENT_RUNS,
                [
                    {"agent_name": "reviewer", "stage": "initial_review"},
                    {"agent_name": "author", "stage": "author_rebuttal"},
                    {"agent_name": "reviewer", "stage": "revision_review"},
                    {"agent_name": "sc", "stage": "score_calibration"},
                    {"agent_name": "reviewer", "stage": "verification_review"},
                    {"agent_name": "sc", "stage": "score_calibration"},
                    {"agent_name": "finalizer", "stage": "finalizer"},
                ],
            )
            self.assertEqual(SEEN_PAYLOADS[-1]["source_review_anchor"]["stage"], "verification_review")
            self.assertIn("verification_review", SEEN_PAYLOADS[-1]["source_review_anchor"]["content"])

            timeline = [
                json.loads(line)
                for line in (output_path.parent / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            sc_outputs = [event for event in timeline if event["event"] == "agent_output" and event["role"] == "sc"]
            self.assertEqual([event["round"] for event in sc_outputs], [1, 2])

    @patch.object(StructuredDeepAgent, "run", new=_fake_structured_deep_agent_run)
    @patch.object(RawDeepAgent, "run", new=_fake_raw_deep_agent_run)
    def test_missing_score_calibration_status_triggers_reviewer_verification(self) -> None:
        SEEN_DISCUSSION_HISTORIES.clear()
        SEEN_PAYLOADS.clear()
        SEEN_AGENT_RUNS.clear()
        SC_STATUS_QUEUE.clear()
        SC_STATUS_QUEUE.append(None)
        with tempfile.TemporaryDirectory() as tmpdir:
            report, output_path = self._run_sample_workflow(tmpdir)
            self.assertTrue(output_path.exists())
            self.assertEqual(report.title, "Sample Retrieval Paper")
            self.assertEqual(
                SEEN_AGENT_RUNS,
                [
                    {"agent_name": "reviewer", "stage": "initial_review"},
                    {"agent_name": "author", "stage": "author_rebuttal"},
                    {"agent_name": "reviewer", "stage": "revision_review"},
                    {"agent_name": "sc", "stage": "score_calibration"},
                    {"agent_name": "reviewer", "stage": "verification_review"},
                    {"agent_name": "finalizer", "stage": "finalizer"},
                ],
            )
            self.assertNotIn("SC_STATUS:", SEEN_PAYLOADS[-2]["sc_handoff"]["content"])
            self.assertEqual(SEEN_PAYLOADS[-1]["source_review_anchor"]["stage"], "verification_review")
            self.assertIn("verification_review", SEEN_PAYLOADS[-1]["source_review_anchor"]["content"])

    def test_role_prompts_delegate_historical_evidence_and_preserve_review_content(self) -> None:
        prompts = {
            agent_name: build_main_agent_prompt(agent_name, use_data_subagent=True)
            for agent_name in ("reviewer", "author", "sc")
        }
        for prompt in prompts.values():
            self.assertNotIn("Delegate exactly one focused task", prompt)
            self.assertNotIn("Do not call the subagent more than once", prompt)
            self.assertIn("data specialist usage guide", prompt.lower())
            self.assertIn("assigned_data_workspace_abs", prompt)
            self.assertNotIn("Search local available corpora / databases / artifacts", prompt)

        reviewer_prompt = prompts["reviewer"]
        self.assertIn("Data Specialist Usage Guide", reviewer_prompt)
        self.assertIn("Good delegation example", reviewer_prompt)
        self.assertIn("limited baseline comparison", reviewer_prompt)
        self.assertIn("Official-review output policy", reviewer_prompt)
        self.assertIn("formal initial official review", reviewer_prompt)
        self.assertIn("Do not mention or imply process history", reviewer_prompt)
        self.assertIn("authors responded", reviewer_prompt)
        self.assertIn("score calibration", reviewer_prompt)

        author_prompt = prompts["author"]
        self.assertIn("Data Specialist Usage Guide For Author Agent", author_prompt)
        self.assertIn("Good delegation example", author_prompt)
        self.assertIn("reviewer_satisfaction", author_prompt)

        sc_prompt = prompts["sc"]
        self.assertIn("data specialist usage guide for sc agent", sc_prompt.lower())
        self.assertIn("Example delegation", sc_prompt)
        self.assertIn("SQL score/decision statistics", sc_prompt)
        self.assertIn("pass only compact context", sc_prompt)
        self.assertIn("Do not make the final SC decision", sc_prompt)

        finalizer_prompt = build_main_agent_prompt("finalizer", use_data_subagent=False)
        self.assertIn("source_review_anchor", finalizer_prompt)
        self.assertIn("discussion_history", finalizer_prompt)
        self.assertNotIn("Data specialist usage guide", finalizer_prompt)
        self.assertNotIn("assigned_data_workspace_abs", finalizer_prompt)
        self.assertNotIn("call `data_specialist` once", finalizer_prompt)
        self.assertIn("using only the supplied payload", finalizer_prompt)
        self.assertIn("Do not retrieve historical reviews, query SQL/RAG, or delegate data tasks", finalizer_prompt)
        self.assertIn("Do not change the recommendation, rating, or confidence", finalizer_prompt)
        self.assertIn("Public review style", finalizer_prompt)
        self.assertIn("natural official conference review", finalizer_prompt)
        self.assertIn("Preserve the underlying reviewer judgment", finalizer_prompt)
        self.assertIn("Remove or rewrite process traces", finalizer_prompt)
        self.assertIn("Use `discussion_history` only for `discussion_history_summary` and `sc_meta_summary`", finalizer_prompt)

        direct_finalizer_prompt = build_main_agent_prompt("finalizer", use_data_subagent=False)
        self.assertNotIn("Data specialist usage guide", direct_finalizer_prompt)
        self.assertNotIn("assigned_data_workspace_abs", direct_finalizer_prompt)

    def test_reviewer_prompt_adds_external_research_policy_only_when_enabled(self) -> None:
        reviewer_prompt = build_main_agent_prompt(
            "reviewer",
            use_data_subagent=True,
            use_external_researcher=True,
        )
        self.assertIn("external_researcher", reviewer_prompt)
        self.assertIn("assigned_external_workspace_abs", reviewer_prompt)
        self.assertIn("PRIORREVIEW_EXTERNAL_RESEARCH_CUTOFF_DATE", reviewer_prompt)
        self.assertIn("Do not pass the complete paper markdown", reviewer_prompt)

        for agent_name in ("author", "sc"):
            prompt = build_main_agent_prompt(agent_name, use_data_subagent=True)
            self.assertNotIn("external_researcher", prompt)
            self.assertNotIn("assigned_external_workspace_abs", prompt)

    def test_external_researcher_prompt_exposes_only_external_tools(self) -> None:
        prompt = build_external_researcher_prompt()
        self.assertIn("search_semantic_scholar", prompt)
        self.assertIn("search_arxiv", prompt)
        self.assertIn("search_github", prompt)
        self.assertIn("assigned_external_workspace_abs", prompt)
        self.assertIn("PRIORREVIEW_EXTERNAL_RESEARCH_CUTOFF_DATE", prompt)
        self.assertNotIn("write_sql_artifact", prompt)
        self.assertNotIn("run_rag_query", prompt)

    def test_data_subagent_prompt_does_not_reference_parent_payload_text(self) -> None:
        forbidden_fragments = (
            "paper_markdown",
            "full text",
            "全文",
            "agent_input",
            "discussion_history",
        )
        for agent_name in ("reviewer", "author", "sc", "finalizer"):
            prompt = build_data_subagent_prompt(
                agent_name,
                sql_schema_context="",
                sql_planner_rules="",
                rag_query_context="",
            )
            lowered = prompt.lower()
            for fragment in forbidden_fragments:
                self.assertNotIn(fragment, lowered)
            if agent_name == "sc":
                # The current SC template focuses on a calibration evidence packet;
                # workspace scoping is exercised by the controlled-tool tests below.
                self.assertIn("SQL Score Boundary Statistics", prompt)
                self.assertIn("5 vs 6 Calibration Summary", prompt)
                self.assertIn("You do not make the final SC decision", prompt)
            else:
                self.assertIn("assigned_data_workspace_abs", prompt)
                self.assertIn("results_preview", prompt)
                self.assertIn("5000 characters", prompt)
                self.assertIn("results_preview_truncated=true", prompt)
            self.assertIn("Autonomous data analysis workflow", prompt)
            self.assertIn("write_sql_artifact", prompt)
            self.assertIn("run_sql_artifact", prompt)
            self.assertIn("write_data_summary", prompt)
            self.assertIn("`execute`", prompt)
            self.assertIn("`read_file`", prompt)
            self.assertIn("multiple justified repairs", prompt)
            self.assertIn("inspect the complete saved JSON", prompt)
            self.assertNotIn("You do not have shell commands", prompt)
            self.assertIn("_retry_01", prompt)
            if agent_name == "finalizer":
                self.assertNotIn("run_rag_query", prompt)
                self.assertNotIn("review_weakness", prompt)
                self.assertNotIn("review_strength", prompt)
                self.assertNotIn("rebuttal_pair", prompt)
                self.assertIn("Use only the `papers` and `reviews` tables", prompt)
                self.assertIn("style-only guidance", prompt)
                self.assertIn("preserving the original wording without paraphrasing", prompt)
                self.assertIn("If fewer than 5 usable snippets are visible, return fewer than 5 snippets", prompt)
                self.assertIn("underlying service remains unavailable", prompt)
            else:
                self.assertIn("run_rag_query", prompt)
                self.assertIn("own Python and SQL scripts", prompt)
                self.assertIn("underlying service remains unavailable", prompt)

    def test_compiled_data_subagent_tools_are_controlled_and_path_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_workspace = Path(tmpdir) / "artifacts" / "data" / "reviewer_round_01"
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                results_dir=tmpdir,
            )
            tools = ControlledDataSubagentTools(
                settings=settings,
                sql_repository_factory=FakeSQLRepository,
                rag_repository_factory=FakeRAGRepository,
            )
            state = _data_state(data_workspace)

            write_sql = json.loads(tools.write_sql_artifact(state=state, filename="query_01.sql", sql="SELECT 1"))
            self.assertTrue(write_sql["ok"])
            self.assertTrue((data_workspace / "query_01.sql").exists())

            run_sql = json.loads(
                tools.run_sql_artifact(
                    state=state,
                    sql_filename="query_01.sql",
                    output_filename="query_01.result.json",
                    name="query_01",
                    reason="test",
                )
            )
            self.assertTrue(run_sql["ok"])
            self.assertFalse(run_sql["has_query_errors"])
            self.assertEqual(run_sql["query_error_count"], 0)
            self.assertEqual(run_sql["executed_queries"][0]["row_count"], 1)
            self.assertEqual(json.loads(run_sql["results_preview"]), {"query_01": [{"answer": 1, "note": "历史评审"}]})
            self.assertEqual(run_sql["results_preview_chars"], len(run_sql["results_preview"]))
            self.assertEqual(run_sql["results_preview_max_chars"], SQL_RESULTS_PREVIEW_MAX_CHARS)
            self.assertFalse(run_sql["results_preview_truncated"])
            self.assertTrue((data_workspace / "query_01.result.json").exists())
            sql_payload = json.loads((data_workspace / "query_01.result.json").read_text(encoding="utf-8"))
            self.assertNotIn("schema_context", sql_payload)
            self.assertNotIn("planner_rules", sql_payload)
            self.assertEqual(sql_payload["results"], {"query_01": [{"answer": 1, "note": "历史评审"}]})

            run_rag = json.loads(
                tools.run_rag_query(
                    state=state,
                    query="memory in RL POMDP benchmarks",
                    output_filename="rag_01.result.json",
                    top_k=5,
                )
            )
            self.assertTrue(run_rag["ok"])
            self.assertEqual(run_rag["document_count"], 1)
            self.assertTrue((data_workspace / "rag_01.result.json").exists())
            rag_payload = json.loads((data_workspace / "rag_01.result.json").read_text(encoding="utf-8"))
            self.assertNotIn("full_text", rag_payload["documents"][0])

            summary = json.loads(tools.write_data_summary(state=state, content="## Findings\n- query_01.result.json"))
            self.assertTrue(summary["ok"])
            self.assertTrue((data_workspace / "data_summary.md").exists())

            self.assertFalse(json.loads(tools.write_sql_artifact(state=state, filename="../escape.sql", sql="SELECT 1"))["ok"])
            self.assertFalse(json.loads(tools.write_sql_artifact(state=state, filename="query.txt", sql="SELECT 1"))["ok"])
            self.assertFalse(
                json.loads(
                    tools.run_sql_artifact(
                        state=state,
                        sql_filename="query_01.sql",
                        output_filename="/tmp/escape.json",
                    )
                )["ok"]
            )
            self.assertFalse(
                json.loads(
                    tools.run_rag_query(
                        state=state,
                        query="novelty",
                        output_filename="subagent_traces/rag.json",
                    )
                )["ok"]
            )
            self.assertFalse(
                json.loads(tools.write_sql_artifact(state={"messages": []}, filename="query.sql", sql="SELECT 1"))["ok"]
            )

    def test_compiled_data_subagent_sql_tool_truncates_preview_but_persists_complete_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_workspace = Path(tmpdir) / "artifacts" / "data" / "reviewer_round_01"
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                results_dir=tmpdir,
            )
            execution_log = FakeExecutionLog()
            tools = ControlledDataSubagentTools(
                settings=settings,
                sql_repository_factory=FakeLargeSQLRepository,
                rag_repository_factory=FakeRAGRepository,
                execution_log=execution_log,
            )
            state = _data_state(data_workspace)

            self.assertTrue(json.loads(tools.write_sql_artifact(state=state, filename="query_01.sql", sql="SELECT 1"))["ok"])
            run_sql = json.loads(
                tools.run_sql_artifact(
                    state=state,
                    sql_filename="query_01.sql",
                    output_filename="query_01.result.json",
                    name="query_01",
                    reason="test",
                )
            )

            self.assertTrue(run_sql["ok"])
            self.assertTrue(run_sql["results_preview_truncated"])
            self.assertEqual(run_sql["results_preview_chars"], SQL_RESULTS_PREVIEW_MAX_CHARS)
            self.assertEqual(len(run_sql["results_preview"]), SQL_RESULTS_PREVIEW_MAX_CHARS)
            self.assertTrue(run_sql["results_preview"].endswith("..."))
            sql_payload = json.loads((data_workspace / "query_01.result.json").read_text(encoding="utf-8"))
            self.assertEqual(sql_payload["results"]["query_01"][0]["review_text"], "历史评审片段" * 1000)
            tool_result = next(
                event
                for event in execution_log.events
                if event["event"] == "tool_result" and event["name"] == "run_sql_artifact"
            )
            self.assertNotIn("results_preview", tool_result["output"])
            self.assertEqual(tool_result["output"]["results_preview_chars"], SQL_RESULTS_PREVIEW_MAX_CHARS)
            self.assertTrue(tool_result["output"]["results_preview_truncated"])

    def test_compiled_data_subagent_sql_tool_reports_query_errors_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_workspace = Path(tmpdir) / "artifacts" / "data" / "reviewer_round_01"
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                results_dir=tmpdir,
            )
            tools = ControlledDataSubagentTools(
                settings=settings,
                sql_repository_factory=FakeFailingSQLRepository,
                rag_repository_factory=FakeRAGRepository,
            )
            state = _data_state(data_workspace)

            write_sql = json.loads(tools.write_sql_artifact(state=state, filename="query_01.sql", sql="SELECT 1"))
            self.assertTrue(write_sql["ok"])
            run_sql = json.loads(
                tools.run_sql_artifact(
                    state=state,
                    sql_filename="query_01.sql",
                    output_filename="query_01.result.json",
                    name="query_01",
                    reason="test",
                )
            )

            self.assertTrue(run_sql["ok"])
            self.assertTrue(run_sql["has_query_errors"])
            self.assertEqual(run_sql["query_error_count"], 1)
            self.assertIn("GROUP BY", run_sql["executed_queries"][0]["error"])
            self.assertEqual(json.loads(run_sql["results_preview"]), {"query_01": []})
            self.assertFalse(run_sql["results_preview_truncated"])

    def test_compiled_data_subagent_tools_accept_assigned_workspace_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_workspace = Path(tmpdir) / "artifacts" / "data" / "reviewer_round_01"
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                results_dir=tmpdir,
            )
            tools = ControlledDataSubagentTools(
                settings=settings,
                sql_repository_factory=FakeSQLRepository,
                rag_repository_factory=FakeRAGRepository,
            )
            state = {
                "messages": [
                    HumanMessage(
                        content=(
                            "Data query assignment.\n"
                            f"Assigned workspace: {data_workspace}\n"
                            "Only use controlled SQL/RAG tools."
                        )
                    )
                ]
            }

            write_sql = json.loads(tools.write_sql_artifact(state=state, filename="query_01.sql", sql="SELECT 1"))
            self.assertTrue(write_sql["ok"])
            self.assertTrue((data_workspace / "query_01.sql").exists())

    def test_compiled_data_subagent_tools_prefer_runtime_config_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            data_workspace = run_root / "artifacts" / "data" / "reviewer_round_01"
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                results_dir=tmpdir,
            )
            tools = ControlledDataSubagentTools(
                settings=settings,
                run_root=run_root,
                sql_repository_factory=FakeSQLRepository,
                rag_repository_factory=FakeRAGRepository,
            )
            state = {"messages": [HumanMessage(content="Data query assignment without workspace binding.")]}
            config = {
                "configurable": {
                    "assigned_data_workspace": "artifacts/data/reviewer_round_01",
                    "assigned_data_workspace_abs": str(data_workspace),
                }
            }

            write_sql = json.loads(
                tools.write_sql_artifact(
                    state=state,
                    config=config,
                    filename="query_01.sql",
                    sql="SELECT 1",
                )
            )
            self.assertTrue(write_sql["ok"])
            self.assertTrue((data_workspace / "query_01.sql").exists())

            escape = json.loads(
                tools.write_sql_artifact(
                    state=state,
                    config=config,
                    filename="../escape.sql",
                    sql="SELECT 1",
                )
            )
            self.assertFalse(escape["ok"])

            relative_only_config = {
                "configurable": {
                    "assigned_data_workspace": "artifacts/data/reviewer_round_02",
                }
            }
            relative_workspace = run_root / "artifacts" / "data" / "reviewer_round_02"
            write_relative = json.loads(
                tools.write_sql_artifact(
                    state=state,
                    config=relative_only_config,
                    filename="query_02.sql",
                    sql="SELECT 1",
                )
            )
            self.assertTrue(write_relative["ok"])
            self.assertTrue((relative_workspace / "query_02.sql").exists())

    def test_compiled_data_subagent_tools_expose_only_controlled_names(self) -> None:
        settings = Settings.from_env(
            openai_api_key="test-key",
            openai_base_url="https://example.com/v1",
            openai_model="test-model",
            results_dir=tempfile.gettempdir(),
        )
        tool_names = {tool.name for tool in build_data_subagent_tools(settings=settings)}
        self.assertEqual(
            tool_names,
            {"write_sql_artifact", "run_sql_artifact", "run_rag_query", "write_data_summary"},
        )
        self.assertFalse({"execute", "ls", "read_file", "write_file", "glob", "grep"}.intersection(tool_names))

        finalizer_tool_names = {tool.name for tool in build_data_subagent_tools(settings=settings, agent_name="finalizer")}
        self.assertEqual(
            finalizer_tool_names,
            {"write_sql_artifact", "run_sql_artifact", "write_data_summary"},
        )
        self.assertNotIn("run_rag_query", finalizer_tool_names)
        self.assertFalse({"execute", "ls", "read_file", "write_file", "glob", "grep"}.intersection(finalizer_tool_names))

    def test_finalizer_data_subagent_sql_scope_rejects_non_review_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_workspace = Path(tmpdir) / "artifacts" / "data" / "finalizer_final"
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                results_dir=tmpdir,
            )
            tools = ControlledDataSubagentTools(
                settings=settings,
                enable_rag=False,
                allowed_sql_tables=frozenset({"papers", "reviews"}),
                sql_repository_factory=FakeSQLRepository,
                rag_repository_factory=FakeRAGRepository,
            )
            state = _data_state(data_workspace)

            self.assertTrue(
                json.loads(
                    tools.write_sql_artifact(
                        state=state,
                        filename="query_01.sql",
                        sql="SELECT meta_text FROM meta_reviews LIMIT 1",
                    )
                )["ok"]
            )
            run_sql = json.loads(
                tools.run_sql_artifact(
                    state=state,
                    sql_filename="query_01.sql",
                    output_filename="query_01.result.json",
                    name="query_01",
                    reason="finalizer style scope test",
                )
            )

            self.assertTrue(run_sql["ok"])
            self.assertTrue(run_sql["has_query_errors"])
            self.assertIn("Disallowed tables referenced: meta_reviews", run_sql["executed_queries"][0]["error"])
            self.assertEqual(json.loads(run_sql["results_preview"]), {"query_01": []})
            self.assertFalse(run_sql["results_preview_truncated"])
            self.assertFalse(json.loads(tools.run_rag_query(state=state, query="style", output_filename="rag.json"))["ok"])

    def test_compiled_external_research_tools_expose_only_controlled_names(self) -> None:
        settings = Settings.from_env(
            openai_api_key="test-key",
            openai_base_url="https://example.com/v1",
            openai_model="test-model",
            results_dir=tempfile.gettempdir(),
            external_research_cutoff_date="2024-05-01",
        )
        tool_names = {tool.name for tool in build_external_research_tools(settings=settings)}
        self.assertEqual(tool_names, {"search_semantic_scholar", "search_arxiv", "search_github"})
        self.assertFalse({"execute", "ls", "read_file", "write_file", "glob", "grep"}.intersection(tool_names))

    def test_raw_deep_agent_builds_with_compiled_data_subagent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                results_dir=tmpdir,
                console_log_enabled=False,
            )
            agent = RawDeepAgent(
                settings=settings,
                artifact_manager=ArtifactManager(run_root),
                logger=RunLogger(
                    run_id="test-run",
                    paper_stem="sample",
                    log_path=run_root / "sample.log.jsonl",
                    console=False,
                ),
                agent_name="reviewer",
                use_data_subagent=True,
            )
            self.assertIsNotNone(agent.agent)

    def test_raw_and_structured_deep_agent_build_compiled_data_subagent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                results_dir=tmpdir,
                console_log_enabled=False,
            )
            _capture_create_deep_agent.calls.clear()
            with patch("agent.core.deep_agents.create_deep_agent", new=_capture_create_deep_agent):
                RawDeepAgent(
                    settings=settings,
                    artifact_manager=ArtifactManager(run_root / "raw"),
                    logger=RunLogger(
                        run_id="test-run",
                        paper_stem="sample",
                        log_path=run_root / "raw.log.jsonl",
                        console=False,
                    ),
                    agent_name="reviewer",
                    use_data_subagent=True,
                )
                StructuredDeepAgent(
                    settings=settings,
                    artifact_manager=ArtifactManager(run_root / "structured"),
                    logger=RunLogger(
                        run_id="test-run",
                        paper_stem="sample",
                        log_path=run_root / "structured.log.jsonl",
                        console=False,
                    ),
                    agent_name="reviewer",
                    response_model=AgentRawResult,
                    use_data_subagent=True,
                )

            self.assertEqual(len(_capture_create_deep_agent.calls), 2)
            for call in _capture_create_deep_agent.calls:
                subagents = call["subagents"]
                self.assertEqual([spec["name"] for spec in subagents], ["data_specialist"])
                data_subagent = next(spec for spec in subagents if spec["name"] == "data_specialist")
                self.assertIn("runnable", data_subagent)
                self.assertNotIn("system_prompt", data_subagent)
                self.assertNotIn("permissions", data_subagent)
                self.assertNotIn("middleware", data_subagent)
                self.assertEqual(len(call["permissions"]), 1)
                permission = call["permissions"][0]
                self.assertEqual(permission.operations, ["read", "write"])
                self.assertEqual(permission.paths, ["/**"])
                self.assertEqual(permission.mode, "deny")
                self.assertTrue(
                    any(
                        MAIN_DEEP_AGENT_EXCLUDED_TOOLS.issubset(getattr(middleware, "excluded_tools", frozenset()))
                        for middleware in call["middleware"]
                    )
                )
                with self.assertRaises(NotImplementedError):
                    call["backend"].execute("echo ok")

    def test_reviewer_raw_agent_builds_external_researcher_with_data_specialist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                results_dir=tmpdir,
                console_log_enabled=False,
                external_research_cutoff_date="2024-05-01",
            )
            _capture_create_deep_agent.calls.clear()
            with patch("agent.core.deep_agents.create_deep_agent", new=_capture_create_deep_agent):
                RawDeepAgent(
                    settings=settings,
                    artifact_manager=ArtifactManager(run_root / "raw"),
                    logger=RunLogger(
                        run_id="test-run",
                        paper_stem="sample",
                        log_path=run_root / "raw.log.jsonl",
                        console=False,
                    ),
                    agent_name="reviewer",
                    use_data_subagent=True,
                    use_external_researcher=True,
                )

            subagents = _capture_create_deep_agent.calls[-1]["subagents"]
            self.assertEqual([spec["name"] for spec in subagents], ["data_specialist", "external_researcher"])
            self.assertTrue(all("runnable" in spec for spec in subagents))

    def test_author_and_sc_agents_reject_external_researcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                results_dir=tmpdir,
                console_log_enabled=False,
                external_research_cutoff_date="2024-05-01",
            )
            for agent_name in ("author", "sc"):
                with self.assertRaisesRegex(Exception, "reviewer"):
                    RawDeepAgent(
                        settings=settings,
                        artifact_manager=ArtifactManager(run_root / agent_name),
                        logger=RunLogger(
                            run_id="test-run",
                            paper_stem="sample",
                            log_path=run_root / f"{agent_name}.log.jsonl",
                            console=False,
                        ),
                        agent_name=agent_name,
                        use_data_subagent=True,
                        use_external_researcher=True,
                    )

    def test_restricted_harness_profile_disables_todos_and_general_purpose_subagent(self) -> None:
        calls: list[tuple[str, object]] = []

        def capture_profile(key, profile):  # noqa: ANN001
            calls.append((key, profile))

        with patch("agent.core.deep_agents.register_harness_profile", new=capture_profile):
            _register_restricted_harness_profile("test-no-todo-model")

        self.assertEqual(len(calls), 1)
        key, profile = calls[0]
        self.assertEqual(key, "openai:test-no-todo-model")
        self.assertIn(TodoListMiddleware, profile.excluded_middleware)
        self.assertIsNotNone(profile.general_purpose_subagent)
        self.assertFalse(profile.general_purpose_subagent.enabled)

    def test_subagent_trace_uses_runtime_workspace_before_message_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            message_workspace = run_root / "workspace" / "_data_specialist" / "reviewer" / "round_01"
            runtime_workspace = run_root / "workspace" / "_data_specialist" / "reviewer" / "round_02"
            message_workspace.mkdir(parents=True)
            artifact_path = message_workspace / "data_summary.md"
            artifact_path.write_text("summary", encoding="utf-8")
            runtime = type(
                "Runtime",
                (),
                {"config": {"configurable": {"assigned_data_workspace_abs": str(runtime_workspace)}}},
            )()
            middleware = SubAgentTraceMiddleware(
                run_root=run_root,
                subagent_name="data_specialist",
                system_prompt="system",
            )

            resolved = middleware._resolve_parent_workspace(f"Existing artifact: {artifact_path}", runtime)

            self.assertEqual(resolved, runtime_workspace.resolve())

    def test_subagent_trace_message_artifact_path_resolves_to_parent_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            data_workspace = run_root / "artifacts" / "data" / "reviewer_round_01"
            data_workspace.mkdir(parents=True)
            artifact_path = data_workspace / "data_summary.md"
            artifact_path.write_text("summary", encoding="utf-8")
            middleware = SubAgentTraceMiddleware(
                run_root=run_root,
                subagent_name="data_specialist",
                system_prompt="system",
            )

            resolved = middleware._resolve_parent_workspace(f"Existing artifact: {artifact_path}")

            self.assertEqual(resolved, data_workspace.resolve())

    def test_external_subagent_trace_prefers_external_runtime_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            data_workspace = run_root / "artifacts" / "data" / "reviewer_round_01"
            external_workspace = run_root / "artifacts" / "external" / "reviewer_round_01"
            runtime = type(
                "Runtime",
                (),
                {
                    "config": {
                        "configurable": {
                            "assigned_data_workspace_abs": str(data_workspace),
                            "assigned_external_workspace_abs": str(external_workspace),
                            "timeline_role": "reviewer",
                            "timeline_round": 1,
                        }
                    }
                },
            )()
            middleware = SubAgentTraceMiddleware(
                run_root=run_root,
                subagent_name="external_researcher",
                system_prompt="system",
            )

            resolved = middleware._resolve_parent_workspace("External research assignment.", runtime)

            self.assertEqual(resolved, external_workspace.resolve())

    def test_deep_agent_runtime_payload_exposes_only_data_workspace_to_main_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                results_dir=tmpdir,
                console_log_enabled=False,
            )
            with patch("agent.core.deep_agents.create_deep_agent", new=_capture_create_deep_agent):
                agent = RawDeepAgent(
                    settings=settings,
                    artifact_manager=ArtifactManager(run_root / "raw"),
                    logger=RunLogger(
                        run_id="test-run",
                        paper_stem="sample",
                        log_path=run_root / "raw.log.jsonl",
                        console=False,
                    ),
                    agent_name="reviewer",
                    use_data_subagent=True,
                )
            capture = _CapturingRawAgent()
            agent.agent = capture
            agent.run(
                payload={"paper_markdown": "# Sample\n\nBody.", "discussion_history": []},
                artifact_stem="round_01",
                node_name="reviewer_round",
                agent_role="reviewer",
                round_id=1,
                stage="initial_review",
            )
            self.assertIsNotNone(capture.state)
            raw_payload = capture.state["messages"][0]["content"].removeprefix("Task payload:\n")
            payload = json.loads(raw_payload)
            self.assertIn("assigned_data_workspace", payload)
            self.assertIn("assigned_data_workspace_abs", payload)
            self.assertNotIn("assigned_external_workspace", payload)
            self.assertNotIn("assigned_external_workspace_abs", payload)
            self.assertNotIn("assigned_workspace", payload)
            self.assertNotIn("assigned_workspace_abs", payload)
            self.assertIsNotNone(capture.config)
            configurable = capture.config["configurable"]
            self.assertEqual(configurable["assigned_data_workspace"], payload["assigned_data_workspace"])
            self.assertEqual(configurable["assigned_data_workspace_abs"], payload["assigned_data_workspace_abs"])
            self.assertNotIn("assigned_external_workspace", configurable)
            self.assertNotIn("assigned_external_workspace_abs", configurable)

    def test_reviewer_runtime_payload_exposes_external_workspace_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                results_dir=tmpdir,
                console_log_enabled=False,
                external_research_cutoff_date="2024-05-01",
            )
            with patch("agent.core.deep_agents.create_deep_agent", new=_capture_create_deep_agent):
                agent = RawDeepAgent(
                    settings=settings,
                    artifact_manager=ArtifactManager(run_root / "raw"),
                    logger=RunLogger(
                        run_id="test-run",
                        paper_stem="sample",
                        log_path=run_root / "raw.log.jsonl",
                        console=False,
                    ),
                    agent_name="reviewer",
                    use_data_subagent=True,
                    use_external_researcher=True,
                )
            capture = _CapturingRawAgent()
            agent.agent = capture
            agent.run(
                payload={"paper_markdown": "# Sample\n\nBody.", "discussion_history": []},
                artifact_stem="round_01",
                node_name="reviewer_round",
                agent_role="reviewer",
                round_id=1,
                stage="initial_review",
            )
            raw_payload = capture.state["messages"][0]["content"].removeprefix("Task payload:\n")
            payload = json.loads(raw_payload)
            self.assertIn("assigned_data_workspace_abs", payload)
            self.assertIn("assigned_external_workspace_abs", payload)
            self.assertIn("artifacts/external/reviewer_round_01", payload["assigned_external_workspace"])
            configurable = capture.config["configurable"]
            self.assertEqual(configurable["assigned_external_workspace"], payload["assigned_external_workspace"])
            self.assertEqual(configurable["assigned_external_workspace_abs"], payload["assigned_external_workspace_abs"])

    def test_structured_agent_without_data_subagent_does_not_add_workspace_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                results_dir=tmpdir,
                console_log_enabled=False,
            )
            structured = AgentRawResult(
                role="finalizer",
                round_id=0,
                stage="finalizer",
                content="Captured structured result.",
                artifact_refs=[],
                state_artifact=None,
            )
            with patch("agent.core.deep_agents.create_deep_agent", new=_capture_create_deep_agent):
                agent = StructuredDeepAgent(
                    settings=settings,
                    artifact_manager=ArtifactManager(run_root / "structured"),
                    logger=RunLogger(
                        run_id="test-run",
                        paper_stem="sample",
                        log_path=run_root / "structured.log.jsonl",
                        console=False,
                    ),
                    agent_name="finalizer",
                    response_model=AgentRawResult,
                    use_data_subagent=False,
                )
            capture = _CapturingStructuredAgent(structured)
            agent.agent = capture
            agent.run(
                payload={
                    "paper_id": "sample",
                    "title": "Sample",
                    "discussion_history": [
                        {
                            "index": 1,
                            "role": "reviewer",
                            "stage": "finalizer",
                            "round_id": 1,
                            "authority": "source_review",
                            "content": "Final review only.",
                        }
                    ],
                    "source_review_anchor": {
                        "role": "reviewer",
                        "round_id": 1,
                        "stage": "finalizer",
                        "content": "Final review only.",
                    },
                    "output_contract": {"schema": "AgentRawResult"},
                },
                artifact_stem="final",
                node_name="finalizer",
                agent_role="finalizer",
            )
            self.assertIsNotNone(capture.state)
            self.assertIsNone(capture.config)
            raw_payload = capture.state["messages"][0]["content"].removeprefix("Task payload:\n")
            payload = json.loads(raw_payload)
            self.assertEqual(
                set(payload),
                {"paper_id", "title", "discussion_history", "source_review_anchor", "output_contract"},
            )

    def test_finalizer_parses_json_message_when_native_structured_output_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            settings = Settings.from_env(
                openai_api_key="test-key",
                openai_base_url="https://example.com/v1",
                openai_model="test-model",
                results_dir=tmpdir,
                console_log_enabled=False,
            )
            json_content = json.dumps(
                {
                    "review_type": "official_review",
                    "conference_style": "ICLR_2026",
                    "paper_id": "sample",
                    "title": "Sample",
                    "summary": "Summary.",
                    "strengths": ["Strong point."],
                    "weaknesses": ["Weak point."],
                    "questions": [],
                    "required_clarifications": [],
                    "recommendation": "reject",
                    "recommendation_rationale": "Rationale.",
                    "unsupported_extra": "ignored",
                    "supporting_arguments": [],
                    "additional_feedback": [],
                    "code_of_ethics": {"potential_violation": False, "explanation": None},
                    "scores": {"rating": 3, "confidence": 4, "soundness": 5},
                    "discussion_history_summary": "",
                    "sc_meta_summary": "",
                    "evidence_refs": [],
                    "output_path": None,
                }
            )
            with patch("agent.core.deep_agents.create_deep_agent", new=_capture_create_deep_agent):
                _capture_create_deep_agent.calls.clear()
                agent = StructuredDeepAgent(
                    settings=settings,
                    artifact_manager=ArtifactManager(run_root / "structured"),
                    logger=RunLogger(
                        run_id="test-run",
                        paper_stem="sample",
                        log_path=run_root / "structured.log.jsonl",
                        console=False,
                    ),
                    agent_name="finalizer",
                    response_model=FinalReviewReport,
                    use_data_subagent=False,
                )
            self.assertNotIn("response_format", _capture_create_deep_agent.calls[-1])

            agent.agent = _CapturingMessageOnlyAgent(f"```json\n{json_content}\n```")
            report, artifacts = agent.run(
                payload={
                    "paper_id": "sample",
                    "title": "Sample",
                    "discussion_history": [
                        {
                            "index": 1,
                            "role": "reviewer",
                            "stage": "finalizer",
                            "round_id": 1,
                            "authority": "source_review",
                            "content": "Final review only.",
                        }
                    ],
                    "source_review_anchor": {
                        "role": "reviewer",
                        "round_id": 1,
                        "stage": "finalizer",
                        "content": "Final review only.",
                    },
                    "output_contract": {"schema": "FinalReviewReport"},
                },
                artifact_stem="final",
                node_name="finalizer",
                agent_role="finalizer",
            )

            self.assertEqual(report.paper_id, "sample")
            self.assertEqual(report.scores.rating, 3)
            state_payload = json.loads(Path(artifacts.state_artifact).read_text(encoding="utf-8"))
            self.assertEqual(state_payload["structured_response_source"], "message_json")
            self.assertNotIn("unsupported_extra", state_payload["structured_response"])
            self.assertNotIn("soundness", state_payload["structured_response"]["scores"])

    def test_deepagents_internal_artifacts_route_under_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            backend = _make_local_backend(run_root=run_root, env={})
            conversation_result = backend.write("/conversation_history/session_test.md", "history")
            large_result = backend.write("/large_tool_results/tool_test", "tool")

            self.assertIsNone(conversation_result.error)
            self.assertIsNone(large_result.error)
            self.assertTrue(
                (run_root / DEEPAGENTS_INTERNAL_WORKSPACE_ROOT / "conversation_history" / "session_test.md").exists()
            )
            self.assertTrue((run_root / DEEPAGENTS_INTERNAL_WORKSPACE_ROOT / "large_tool_results" / "tool_test").exists())


if __name__ == "__main__":
    unittest.main()
