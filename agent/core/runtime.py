from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any

from agent.core.deep_agents import DeepAgentExecutionError, RawDeepAgent, StructuredDeepAgent
from agent.core.markdown import parse_markdown_paper
from agent.core.openai_http import openai_retry_directive
from agent.core.settings import Settings
from agent.logging.artifact_manager import ArtifactManager
from agent.logging.execution_log import ExecutionLog
from agent.logging.run_logger import RunLogger
from agent.schemas.final_report import FinalReviewReport
from agent.schemas.paper import PaperInput, ParsedPaper
from agent.schemas.raw import AgentRawResult


class WorkflowRuntime:
    """Owns per-paper runtime state, artifacts, and role-specific Deep Agents."""

    def __init__(self, settings: Settings, paper_path: str | Path, *, batch_label: str | None = None) -> None:
        self.settings = settings
        self.paper_path = Path(paper_path)
        self.paper_stem = self.paper_path.stem
        self.final_review_filename = f"{self.paper_stem}.json"
        self.run_id = f"run_{uuid.uuid4().hex[:12]}"
        self.paper_results_dir = settings.results_root / self.paper_stem
        self.run_dir = self.paper_results_dir / "runs" / self.run_id
        self.artifact_manager = ArtifactManager(self.run_dir)
        self.execution_log = ExecutionLog(
            run_id=self.run_id,
            paper_id=self.paper_stem,
            paper_path=self.paper_path,
            run_dir=self.run_dir,
            models={
                "main": settings.openai_model,
                "finalizer": settings.finalizer_openai_model,
            },
        )
        self.logger = RunLogger(
            run_id=self.run_id,
            paper_stem=self.paper_stem,
            log_path=None,
            console=settings.console_log_enabled,
            batch_label=batch_label,
        )
        self.reviewer_agent = RawDeepAgent(
            settings=settings,
            artifact_manager=self.artifact_manager,
            logger=self.logger,
            execution_log=self.execution_log,
            agent_name="reviewer",
            use_data_subagent=settings.reviewer_data_agent_enabled,
        )
        self.author_agent = RawDeepAgent(
            settings=settings,
            artifact_manager=self.artifact_manager,
            logger=self.logger,
            execution_log=self.execution_log,
            agent_name="author",
            use_data_subagent=settings.author_data_agent_enabled,
        )
        self.sc_agent = RawDeepAgent(
            settings=settings,
            artifact_manager=self.artifact_manager,
            logger=self.logger,
            execution_log=self.execution_log,
            agent_name="sc",
            use_data_subagent=settings.sc_data_agent_enabled,
        )
        self._validate_finalizer_settings()
        self.finalizer_agent = StructuredDeepAgent(
            settings=settings,
            artifact_manager=self.artifact_manager,
            logger=self.logger,
            execution_log=self.execution_log,
            agent_name="finalizer",
            response_model=FinalReviewReport,
            use_data_subagent=False,
            openai_api_key=settings.finalizer_openai_api_key,
            openai_base_url=settings.finalizer_openai_base_url,
            openai_model=settings.finalizer_openai_model,
            temperature=settings.finalizer_temperature,
            use_default_openai_config=False,
        )

    def _validate_finalizer_settings(self) -> None:
        if not self.settings.finalizer_openai_api_key:
            raise DeepAgentExecutionError(
                "FINALIZER_OPENAI_API_KEY is missing. Configure .env before running the workflow."
            )
        if not self.settings.finalizer_openai_model:
            raise DeepAgentExecutionError(
                "FINALIZER_OPENAI_MODEL is missing. Configure .env before running the workflow."
            )

    def make_paper_input(self) -> PaperInput:
        """Create the initial workflow input object for the current markdown file."""

        return PaperInput(paper_id=self.paper_stem, file_path=str(self.paper_path))

    def ingest_paper(self, paper_input: PaperInput) -> ParsedPaper:
        """Read the markdown file and persist the parsed paper snapshot."""

        self.logger.log(
            event_type="node_enter",
            message="Loading markdown paper.",
            node_name="ingest_paper",
            input_refs=[paper_input.file_path],
        )
        parsed = parse_markdown_paper(paper_input)
        paper_snapshot = self.artifact_manager.write_text("artifacts/inputs/paper_snapshot.md", parsed.raw_markdown)
        paper_meta = self.artifact_manager.write_json(
            "artifacts/inputs/paper_meta.json",
            {
                "paper_id": parsed.paper_id,
                "source_type": paper_input.source_type,
                "file_path": paper_input.file_path,
                "title": self._paper_title(parsed),
                "paper_snapshot": self.execution_log.rel(paper_snapshot),
            },
        )
        self.logger.log(
            event_type="node_exit",
            message="Markdown paper parsed.",
            node_name="ingest_paper",
            output_refs=[paper_snapshot, paper_meta],
        )
        return parsed

    def reviewer_round(
        self,
        *,
        parsed_paper: ParsedPaper,
        round_id: int,
        latest_rebuttal: AgentRawResult | None,
        latest_review: AgentRawResult | None,
        latest_sc_decision: AgentRawResult | None,
        review_stage: str,
        discussion_history: list[dict[str, Any]],
    ) -> AgentRawResult:
        """Run the reviewer Deep Agent for an initial, revision, or verification pass."""

        self.logger.log(
            event_type="agent_started",
            message=f"Starting reviewer round {round_id}.",
            node_name="reviewer_round",
            agent_role="reviewer",
        )
        payload = self._build_reviewer_payload(
            parsed_paper=parsed_paper,
            round_id=round_id,
            review_stage=review_stage,
            latest_review=latest_review,
            latest_rebuttal=latest_rebuttal,
            latest_sc_decision=latest_sc_decision,
            discussion_history=discussion_history,
        )
        self._record_agent_input(role="reviewer", round_id=round_id, stage=review_stage, payload=payload)
        try:
            review, artifacts = self._run_raw_agent_with_retry(
                self.reviewer_agent,
                payload=payload,
                artifact_stem=f"round_{round_id:02d}",
                node_name="reviewer_round",
                agent_role="reviewer",
                round_id=round_id,
                stage=review_stage,
                retry_role="reviewer",
                retry_name=review_stage,
            )
        except Exception as exc:
            self._record_agent_error(role="reviewer", round_id=round_id, name=review_stage, exc=exc)
            raise
        output_artifact = self._write_raw_agent_output(role="reviewer", round_id=round_id, result=review, artifacts=artifacts)
        self.logger.log(
            event_type="agent_finished",
            message=f"Reviewer round {round_id} completed.",
            node_name="reviewer_round",
            agent_role="reviewer",
            input_refs=artifacts.created_files,
            output_refs=[output_artifact],
            metadata={"workspace": artifacts.workspace, "stage": review_stage},
        )
        return review

    def author_round(
        self,
        *,
        parsed_paper: ParsedPaper,
        latest_review: AgentRawResult,
        round_id: int,
        discussion_history: list[dict[str, Any]],
    ) -> AgentRawResult:
        """Run the author Deep Agent to produce a rebuttal to the latest review."""

        self.logger.log(
            event_type="agent_started",
            message=f"Starting author round {round_id}.",
            node_name="author_round",
            agent_role="author",
        )
        payload = self._build_author_payload(
            parsed_paper=parsed_paper,
            latest_review=latest_review,
            round_id=round_id,
            discussion_history=discussion_history,
        )
        self._record_agent_input(role="author", round_id=round_id, stage="author_rebuttal", payload=payload)
        try:
            rebuttal, artifacts = self._run_raw_agent_with_retry(
                self.author_agent,
                payload=payload,
                artifact_stem=f"round_{round_id:02d}",
                node_name="author_round",
                agent_role="author",
                round_id=round_id,
                stage="author_rebuttal",
                retry_role="author",
                retry_name="author_rebuttal",
            )
        except Exception as exc:
            self._record_agent_error(role="author", round_id=round_id, name="author_rebuttal", exc=exc)
            raise
        output_artifact = self._write_raw_agent_output(role="author", round_id=round_id, result=rebuttal, artifacts=artifacts)
        self.logger.log(
            event_type="agent_finished",
            message=f"Author round {round_id} completed.",
            node_name="author_round",
            agent_role="author",
            input_refs=artifacts.created_files,
            output_refs=[output_artifact],
            metadata={"workspace": artifacts.workspace},
        )
        return rebuttal

    def score_calibration_round(
        self,
        *,
        parsed_paper: ParsedPaper,
        latest_review: AgentRawResult,
        latest_rebuttal: AgentRawResult | None,
        round_id: int,
        discussion_history: list[dict[str, Any]],
    ) -> AgentRawResult:
        """Run the score calibration Deep Agent for one pass over the reviewer result."""

        self.logger.log(
            event_type="agent_started",
            message=f"Starting score calibration round {round_id}.",
            node_name="score_calibration_round",
            agent_role="sc",
        )
        payload = self._build_score_calibration_payload(
            parsed_paper=parsed_paper,
            latest_review=latest_review,
            latest_rebuttal=latest_rebuttal,
            round_id=round_id,
            discussion_history=discussion_history,
        )
        self._record_agent_input(role="sc", round_id=round_id, stage="score_calibration", payload=payload)
        try:
            sc_result, artifacts = self._run_raw_agent_with_retry(
                self.sc_agent,
                payload=payload,
                artifact_stem=f"round_{round_id:02d}",
                node_name="score_calibration_round",
                agent_role="sc",
                round_id=round_id,
                stage="score_calibration",
                retry_role="sc",
                retry_name="score_calibration",
            )
        except Exception as exc:
            self._record_agent_error(role="sc", round_id=round_id, name="score_calibration", exc=exc)
            raise
        output_artifact = self._write_raw_agent_output(role="sc", round_id=round_id, result=sc_result, artifacts=artifacts)
        self.logger.log(
            event_type="agent_finished",
            message=f"Score calibration round {round_id} completed.",
            node_name="score_calibration_round",
            agent_role="sc",
            input_refs=artifacts.created_files,
            output_refs=[output_artifact],
            metadata={"workspace": artifacts.workspace},
        )
        return sc_result

    def _run_raw_agent_with_retry(
        self,
        agent: RawDeepAgent,
        *,
        payload: dict[str, Any],
        artifact_stem: str,
        node_name: str,
        agent_role: str,
        round_id: int,
        stage: str,
        retry_role: str,
        retry_name: str,
    ) -> tuple[AgentRawResult, Any]:
        max_attempts = max(1, int(self.settings.agent_max_attempts))
        for attempt in range(1, max_attempts + 1):
            try:
                return agent.run(
                    payload=payload,
                    artifact_stem=artifact_stem,
                    node_name=node_name,
                    agent_role=agent_role,
                    round_id=round_id,
                    stage=stage,
                )
            except Exception as exc:
                if attempt < max_attempts:
                    retry_directive = openai_retry_directive(exc, attempt=attempt)
                    retry_output = {
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "retryable_http_error": retry_directive.retryable_http_error,
                    }
                    if retry_directive.sleep_seconds > 0:
                        retry_output["sleep_seconds"] = retry_directive.sleep_seconds
                    self.execution_log.record(
                        event="agent_retry",
                        role=retry_role,
                        round_id=round_id,
                        name=retry_name,
                        output=retry_output,
                    )
                    if retry_directive.sleep_seconds > 0:
                        time.sleep(retry_directive.sleep_seconds)
                    continue
                raise
        raise RuntimeError("raw agent retry loop exited unexpectedly")

    def finalize_output(
        self,
        *,
        parsed_paper: ParsedPaper,
        final_review: AgentRawResult,
        discussion_history: list[dict[str, Any]],
    ) -> FinalReviewReport:
        self.logger.log(
            event_type="agent_started",
            message="Starting finalizer.",
            node_name="finalizer",
            agent_role="finalizer",
        )
        payload = self._build_finalizer_payload(
            parsed_paper=parsed_paper,
            final_review=final_review,
            discussion_history=discussion_history,
        )
        self.execution_log.record(
            event="agent_input",
            role="finalizer",
            name="finalizer",
            input={
                "paper_id": payload["paper_id"],
                "source_review_stage": final_review.stage,
                "source_review_round": final_review.round_id,
                "final_review_chars": len(final_review.content),
                "discussion_items": len(payload["discussion_history"]),
            },
        )
        report, _artifacts = self._run_finalizer_with_retry(payload)
        output_path = str(self.run_dir / self.final_review_filename)
        report.output_path = output_path
        self.artifact_manager.write_json(self.final_review_filename, report.model_dump(mode="json"))
        finalizer_md = self._write_finalizer_markdown(report)
        self.execution_log.record(
            event="agent_output",
            role="finalizer",
            name="finalizer",
            output={
                "output_file": self.execution_log.rel(finalizer_md),
                "final_output": self.execution_log.rel(output_path),
                "recommendation": report.recommendation,
            },
        )
        self.logger.log(
            event_type="final_output_saved",
            message="Final review JSON saved.",
            node_name="finalizer",
            agent_role="finalizer",
            output_refs=[output_path, finalizer_md],
        )
        return report

    def _run_finalizer_with_retry(self, payload: dict[str, Any]) -> tuple[FinalReviewReport, Any]:
        max_attempts = max(1, int(self.settings.finalizer_max_attempts))
        try:
            return self._run_finalizer_agent(
                self.finalizer_agent,
                payload=payload,
                max_attempts=max_attempts,
            )
        except Exception as exc:
            self._record_agent_error(role="finalizer", round_id=None, name="finalizer", exc=exc)
            raise

    def _run_finalizer_agent(
        self,
        agent: StructuredDeepAgent,
        *,
        payload: dict[str, Any],
        max_attempts: int,
    ) -> tuple[FinalReviewReport, Any]:
        for attempt in range(1, max_attempts + 1):
            try:
                return agent.run(
                    payload=payload,
                    artifact_stem="final",
                    node_name="finalizer",
                    agent_role="finalizer",
                )
            except Exception as exc:
                if attempt < max_attempts:
                    retry_directive = openai_retry_directive(exc, attempt=attempt)
                    retry_output = {
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "retryable_http_error": retry_directive.retryable_http_error,
                    }
                    if retry_directive.sleep_seconds > 0:
                        retry_output["sleep_seconds"] = retry_directive.sleep_seconds
                    self.execution_log.record(
                        event="agent_retry",
                        role="finalizer",
                        name="finalizer",
                        output=retry_output,
                    )
                    if retry_directive.sleep_seconds > 0:
                        time.sleep(retry_directive.sleep_seconds)
                    continue
                raise
        raise RuntimeError("finalizer retry loop exited unexpectedly")

    @staticmethod
    def score_calibration_requires_revision(score_calibration: AgentRawResult) -> bool:
        """Return True unless SC explicitly marks the review as normal."""

        status_match = re.search(
            r"^\s*SC_STATUS\s*:\s*(NORMAL|NEEDS_REVIEWER_REVISION)\b",
            score_calibration.content,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if status_match is None:
            return True
        return status_match.group(1).upper() != "NORMAL"

    def append_discussion_history(
        self,
        history: list[dict[str, Any]],
        *,
        role: str,
        stage: str,
        result: AgentRawResult,
    ) -> list[dict[str, Any]]:
        """Append one raw agent output and persist the full discussion history."""

        updated = [
            *history,
            self._discussion_history_entry(role=role, stage=stage, result=result),
        ]
        return updated

    def persist_discussion_history(self, history: list[dict[str, Any]]) -> str:
        """Persist discussion history only as a debug artifact when explicitly requested."""

        return self.artifact_manager.write_json("debug/raw_messages/discussion_history.json", history)

    @staticmethod
    def _discussion_history_entry(
        *,
        role: str,
        stage: str,
        result: AgentRawResult,
    ) -> dict[str, Any]:
        return {
            "role": role,
            "round_id": result.round_id,
            "stage": stage,
            "content": result.model_dump(mode="json"),
        }

    @staticmethod
    def _raw_handoff(result: AgentRawResult | None) -> dict | None:
        """Return raw result context for the next role."""

        return result.model_dump(mode="json") if result is not None else None

    def _record_agent_input(self, *, role: str, round_id: int, stage: str, payload: dict[str, Any]) -> None:
        """Record a compact, ordered input event without embedding large text."""

        data_workspace = f"artifacts/data/{role}_round_{round_id:02d}" if self._role_uses_data_agent(role) else None
        self.execution_log.record(
            event="agent_input",
            role=role,
            round_id=round_id,
            name=stage,
            input={
                "paper": "artifacts/inputs/paper_snapshot.md" if "paper_markdown" in payload else None,
                "objective": stage,
                "historical_objective": payload.get("round_context", {}).get("historical_objective"),
                "discussion_items": len(payload.get("discussion_history") or []),
                "handoffs": [
                    key
                    for key in ("reviewer_handoff", "author_handoff", "reviewer_memory", "sc_handoff")
                    if payload.get(key)
                ],
                "data_workspace": data_workspace,
                "external_workspace": (
                    f"artifacts/external/{role}_round_{round_id:02d}"
                    if role == "reviewer" and self.reviewer_agent.use_external_researcher
                    else None
                ),
            },
        )

    def _role_uses_data_agent(self, role: str) -> bool:
        if role == "reviewer":
            return self.reviewer_agent.use_data_subagent
        if role == "author":
            return self.author_agent.use_data_subagent
        if role == "sc":
            return self.sc_agent.use_data_subagent
        if role == "finalizer":
            return self.finalizer_agent.use_data_subagent
        return False

    def _record_agent_error(self, *, role: str, round_id: int | None, name: str, exc: Exception) -> None:
        self.execution_log.record(
            event="agent_error",
            role=role,
            round_id=round_id,
            name=name,
            output={"error_type": type(exc).__name__, "error": str(exc)},
        )

    def _write_raw_agent_output(
        self,
        *,
        role: str,
        round_id: int,
        result: AgentRawResult,
        artifacts: Any,
    ) -> str:
        output_artifact = self.artifact_manager.write_text(
            f"artifacts/outputs/{role}_round_{round_id:02d}.md",
            result.content.strip() + "\n",
        )
        result.artifact_refs = [self.execution_log.rel(path) for path in artifacts.created_files]
        result.state_artifact = self.execution_log.rel(output_artifact)
        self.execution_log.record(
            event="agent_output",
            role=role,
            round_id=round_id,
            name=result.stage,
            output={
                "output_file": self.execution_log.rel(output_artifact),
                "data_artifacts": [self.execution_log.rel(path) for path in artifacts.created_files],
                "artifact_count": len(artifacts.created_files),
            },
        )
        return output_artifact

    def _write_finalizer_markdown(self, report: FinalReviewReport) -> str:
        strengths = "\n".join(f"- {item}" for item in report.strengths) or "- none"
        weaknesses = "\n".join(f"- {item}" for item in report.weaknesses) or "- none"
        questions = "\n".join(f"- {item}" for item in report.questions) or "- none"
        content = (
            f"# {report.title}\n\n"
            f"- recommendation: {report.recommendation}\n"
            f"- rating: {report.scores.rating}\n"
            f"- confidence: {report.scores.confidence}\n\n"
            "## Summary\n\n"
            f"{report.summary}\n\n"
            "## Strengths\n"
            f"{strengths}\n\n"
            "## Weaknesses\n"
            f"{weaknesses}\n\n"
            "## Questions\n"
            f"{questions}\n"
        )
        return self.artifact_manager.write_text("artifacts/outputs/finalizer.md", content)

    @staticmethod
    def _paper_title(parsed_paper: ParsedPaper) -> str:
        """Extract a display title from the markdown heading, falling back to paper id."""

        for line in parsed_paper.raw_markdown.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                if title:
                    return title
        return parsed_paper.paper_id.replace("_", " ").title()

    def _build_reviewer_payload(
        self,
        *,
        parsed_paper: ParsedPaper,
        round_id: int,
        review_stage: str,
        latest_review: AgentRawResult | None,
        latest_rebuttal: AgentRawResult | None,
        latest_sc_decision: AgentRawResult | None,
        discussion_history: list[dict[str, Any]],
    ) -> dict:
        """Build the reviewer input with explicit author and SC handoff sections."""

        return {
            "paper_markdown": parsed_paper.raw_markdown,
            "round_context": {
                "round_id": round_id,
                "review_stage": review_stage,
                "historical_objective": self._reviewer_historical_objective(),
                "public_output_policy": self._reviewer_public_output_policy(),
            },
            "reviewer_memory": self._raw_handoff(latest_review),
            "author_handoff": self._raw_handoff(latest_rebuttal),
            "sc_handoff": self._raw_handoff(latest_sc_decision),
            "discussion_history": discussion_history,
        }

    def _build_author_payload(
        self,
        *,
        parsed_paper: ParsedPaper,
        latest_review: AgentRawResult,
        round_id: int,
        discussion_history: list[dict[str, Any]],
    ) -> dict:
        """Build the author input around the current reviewer concerns."""

        historical_objective = (
            "Collect historical rebuttal and discussion priors before drafting the rebuttal."
            if self.author_agent.use_data_subagent
            else "Do not use a historical SQL/RAG data agent. Draft the rebuttal only from the current paper, reviewer handoff, and discussion history."
        )
        return {
            "paper_markdown": parsed_paper.raw_markdown,
            "round_context": {
                "round_id": round_id,
                "historical_objective": historical_objective,
            },
            "reviewer_handoff": self._raw_handoff(latest_review),
            "discussion_history": discussion_history,
        }

    def _build_score_calibration_payload(
        self,
        *,
        parsed_paper: ParsedPaper,
        latest_review: AgentRawResult,
        latest_rebuttal: AgentRawResult | None,
        round_id: int,
        discussion_history: list[dict[str, Any]],
    ) -> dict:
        """Build the SC input from the latest reviewer and author handoffs."""

        historical_objective = (
            "Collect historical score distributions and meta-review priors before calibrating the decision."
            if self.sc_agent.use_data_subagent
            else "Do not use a historical SQL/RAG data agent. Calibrate only from the current paper, reviewer handoff, author handoff, and discussion history."
        )
        return {
            "paper_markdown": parsed_paper.raw_markdown,
            "round_context": {
                "round_id": round_id,
                "historical_objective": historical_objective,
            },
            "reviewer_handoff": self._raw_handoff(latest_review),
            "author_handoff": self._raw_handoff(latest_rebuttal),
            "discussion_history": discussion_history,
        }

    def _reviewer_historical_objective(self) -> str:
        if self.reviewer_agent.use_data_subagent and self.reviewer_agent.use_external_researcher:
            return (
                "Collect historical SQL priors, vector RAG evidence, and cutoff-safe external related-work "
                "signals before finalizing the review."
            )
        if self.reviewer_agent.use_data_subagent:
            return "Collect historical SQL priors and vector RAG evidence before finalizing the review."
        if self.reviewer_agent.use_external_researcher:
            return (
                "Do not use a historical SQL/RAG data agent. Use only the current paper, discussion history, "
                "handoffs, and cutoff-safe external related-work signals before finalizing the review."
            )
        return (
            "Do not use a historical SQL/RAG data agent. Finalize the review only from the current paper, "
            "discussion history, and provided handoffs."
        )

    @staticmethod
    def _reviewer_public_output_policy() -> str:
        return (
            "The final reviewer text will be used as an official initial review. Use internal handoffs, "
            "author responses, discussion history, and SC context only to improve judgment. Do not mention "
            "rebuttal, author response, revision, SC, calibration, final manuscript, or any workflow history "
            "in the public review text; rewrite those signals as direct paper assessments."
        )

    def _build_finalizer_payload(
        self,
        *,
        parsed_paper: ParsedPaper,
        final_review: AgentRawResult,
        discussion_history: list[dict[str, Any]],
    ) -> dict:
        """Build the finalizer input with marked history and a reviewer source anchor."""

        marked_history = self._build_finalizer_discussion_history(
            discussion_history=discussion_history,
            final_review=final_review,
        )

        return {
            "paper_id": parsed_paper.paper_id,
            "title": self._paper_title(parsed_paper),
            "discussion_history": marked_history,
            "source_review_anchor": {
                "role": final_review.role,
                "round_id": final_review.round_id,
                "stage": final_review.stage,
                "content": final_review.content,
            },
            "output_contract": {
                "schema": "FinalReviewReport",
                "json_schema": FinalReviewReport.model_json_schema(),
                "output_path": "Return null; the workflow fills the real path after validation.",
            },
        }

    @classmethod
    def _build_finalizer_discussion_history(
        cls,
        *,
        discussion_history: list[dict[str, Any]],
        final_review: AgentRawResult,
    ) -> list[dict[str, Any]]:
        """Mark the final reviewer item as the sole review source for finalization."""

        marked: list[dict[str, Any]] = []
        found_source = False
        for entry in discussion_history:
            item = cls._finalizer_history_item(entry, final_review=final_review)
            if item["authority"] == "source_review":
                found_source = True
            item["index"] = len(marked) + 1
            marked.append(item)

        if not found_source:
            marked.append(
                {
                    "index": len(marked) + 1,
                    "role": final_review.role,
                    "stage": final_review.stage,
                    "round_id": final_review.round_id,
                    "authority": "source_review",
                    "content": final_review.content,
                }
            )
        return marked

    @classmethod
    def _finalizer_history_item(cls, entry: dict[str, Any], *, final_review: AgentRawResult) -> dict[str, Any]:
        raw_content = entry.get("content")
        if isinstance(raw_content, dict):
            role = str(raw_content.get("role") or entry.get("role") or "")
            stage = str(raw_content.get("stage") or entry.get("stage") or "")
            round_id = raw_content.get("round_id", entry.get("round_id"))
            content = str(raw_content.get("content") or "")
        else:
            role = str(entry.get("role") or "")
            stage = str(entry.get("stage") or "")
            round_id = entry.get("round_id")
            content = str(raw_content or "")

        authority = "source_review" if cls._is_source_review_history_item(
            role=role,
            stage=stage,
            round_id=round_id,
            content=content,
            final_review=final_review,
        ) else "context_only"

        return {
            "index": 0,
            "role": role,
            "stage": stage,
            "round_id": round_id,
            "authority": authority,
            "content": content,
        }

    @staticmethod
    def _is_source_review_history_item(
        *,
        role: str,
        stage: str,
        round_id: Any,
        content: str,
        final_review: AgentRawResult,
    ) -> bool:
        try:
            round_matches = int(round_id) == final_review.round_id
        except (TypeError, ValueError):
            round_matches = False
        return (
            role == final_review.role
            and stage == final_review.stage
            and round_matches
            and content.strip() == final_review.content.strip()
        )
