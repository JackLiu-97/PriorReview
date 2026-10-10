from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent.core.runtime import WorkflowRuntime
from agent.core.settings import Settings
from agent.schemas.final_report import FinalReviewReport
from agent.schemas.paper import PaperInput, ParsedPaper
from agent.schemas.raw import AgentRawResult


class AutoReviewGraphState(TypedDict, total=False):
    """Mutable workflow state threaded through the LangGraph nodes."""

    paper_input: PaperInput
    parsed_paper: ParsedPaper
    initial_review: AgentRawResult
    latest_review: AgentRawResult
    latest_rebuttal: AgentRawResult
    score_calibration: AgentRawResult
    final_output: FinalReviewReport
    discussion_history: list[dict]
    score_calibration_round_id: int
    sc_revision_count: int


def build_auto_review_graph(runtime: WorkflowRuntime, settings: Settings):
    """Create the LangGraph orchestrator that coordinates the role-specific Deep Agents."""

    graph = StateGraph(AutoReviewGraphState)
    max_reviewer_author_rounds = _positive_int(settings.max_reviewer_author_rounds, default=2)
    max_sc_rereview_rounds = _non_negative_int(settings.max_sc_rereview_rounds, default=1)

    def ingest_paper(state: AutoReviewGraphState) -> AutoReviewGraphState:
        paper_input = state["paper_input"]
        parsed = runtime.ingest_paper(paper_input)
        return {"parsed_paper": parsed}

    def reviewer_round(state: AutoReviewGraphState) -> AutoReviewGraphState:
        history = state.get("discussion_history", [])
        review = runtime.reviewer_round(
            parsed_paper=state["parsed_paper"],
            round_id=1,
            latest_rebuttal=None,
            latest_review=None,
            latest_sc_decision=None,
            review_stage="initial_review",
            discussion_history=history,
        )
        updated_history = runtime.append_discussion_history(
            history,
            role="reviewer",
            stage="initial_review",
            result=review,
        )
        return {
            "initial_review": review,
            "latest_review": review,
            "discussion_history": updated_history,
        }

    def author_round(state: AutoReviewGraphState) -> AutoReviewGraphState:
        history = state.get("discussion_history", [])
        round_id = state["latest_review"].round_id
        rebuttal = runtime.author_round(
            parsed_paper=state["parsed_paper"],
            latest_review=state["latest_review"],
            round_id=round_id,
            discussion_history=history,
        )
        updated_history = runtime.append_discussion_history(
            history,
            role="author",
            stage="author_rebuttal",
            result=rebuttal,
        )
        return {
            "latest_rebuttal": rebuttal,
            "discussion_history": updated_history,
        }

    def reviewer_revision_round(state: AutoReviewGraphState) -> AutoReviewGraphState:
        history = state.get("discussion_history", [])
        round_id = state["latest_review"].round_id + 1
        review = runtime.reviewer_round(
            parsed_paper=state["parsed_paper"],
            round_id=round_id,
            latest_rebuttal=state.get("latest_rebuttal"),
            latest_review=state.get("latest_review"),
            latest_sc_decision=None,
            review_stage="revision_review",
            discussion_history=history,
        )
        updated_history = runtime.append_discussion_history(
            history,
            role="reviewer",
            stage="revision_review",
            result=review,
        )
        return {
            "latest_review": review,
            "discussion_history": updated_history,
        }

    def score_calibration_round(state: AutoReviewGraphState) -> AutoReviewGraphState:
        history = state.get("discussion_history", [])
        round_id = state.get("score_calibration_round_id", 1)
        sc_result = runtime.score_calibration_round(
            parsed_paper=state["parsed_paper"],
            latest_review=state["latest_review"],
            latest_rebuttal=state.get("latest_rebuttal"),
            round_id=round_id,
            discussion_history=history,
        )
        updated_history = runtime.append_discussion_history(
            history,
            role="sc",
            stage="score_calibration",
            result=sc_result,
        )
        return {
            "score_calibration": sc_result,
            "discussion_history": updated_history,
            "score_calibration_round_id": round_id + 1,
        }

    def reviewer_verification_round(state: AutoReviewGraphState) -> AutoReviewGraphState:
        history = state.get("discussion_history", [])
        round_id = state["latest_review"].round_id + 1
        sc_revision_count = state.get("sc_revision_count", 0) + 1
        review = runtime.reviewer_round(
            parsed_paper=state["parsed_paper"],
            round_id=round_id,
            latest_rebuttal=state.get("latest_rebuttal"),
            latest_review=state.get("latest_review"),
            latest_sc_decision=state.get("score_calibration"),
            review_stage="verification_review",
            discussion_history=history,
        )
        updated_history = runtime.append_discussion_history(
            history,
            role="reviewer",
            stage="verification_review",
            result=review,
        )
        return {
            "latest_review": review,
            "discussion_history": updated_history,
            "sc_revision_count": sc_revision_count,
        }

    def finalizer(state: AutoReviewGraphState) -> AutoReviewGraphState:
        final_output = runtime.finalize_output(
            parsed_paper=state["parsed_paper"],
            final_review=state["latest_review"],
            discussion_history=state.get("discussion_history", []),
        )
        return {"final_output": final_output}

    def route_after_reviewer(state: AutoReviewGraphState) -> str:
        if state["latest_review"].round_id < max_reviewer_author_rounds:
            return "author_round"
        return "score_calibration_round"

    def route_after_score_calibration(state: AutoReviewGraphState) -> str:
        if (
            runtime.score_calibration_requires_revision(state["score_calibration"])
            and state.get("sc_revision_count", 0) < max_sc_rereview_rounds
        ):
            return "reviewer_verification_round"
        return "finalizer"

    def route_after_reviewer_verification(state: AutoReviewGraphState) -> str:
        if state.get("sc_revision_count", 0) < max_sc_rereview_rounds:
            return "score_calibration_round"
        return "finalizer"

    graph.add_node("ingest_paper", ingest_paper)
    graph.add_node("reviewer_round", reviewer_round)
    graph.add_node("author_round", author_round)
    graph.add_node("reviewer_revision_round", reviewer_revision_round)
    graph.add_node("score_calibration_round", score_calibration_round)
    graph.add_node("reviewer_verification_round", reviewer_verification_round)
    graph.add_node("finalizer", finalizer)

    graph.add_edge(START, "ingest_paper")
    graph.add_edge("ingest_paper", "reviewer_round")
    graph.add_conditional_edges(
        "reviewer_round",
        route_after_reviewer,
        {
            "author_round": "author_round",
            "score_calibration_round": "score_calibration_round",
        },
    )
    graph.add_edge("author_round", "reviewer_revision_round")
    graph.add_conditional_edges(
        "reviewer_revision_round",
        route_after_reviewer,
        {
            "author_round": "author_round",
            "score_calibration_round": "score_calibration_round",
        },
    )
    graph.add_conditional_edges(
        "score_calibration_round",
        route_after_score_calibration,
        {
            "reviewer_verification_round": "reviewer_verification_round",
            "finalizer": "finalizer",
        },
    )
    graph.add_conditional_edges(
        "reviewer_verification_round",
        route_after_reviewer_verification,
        {
            "score_calibration_round": "score_calibration_round",
            "finalizer": "finalizer",
        },
    )
    graph.add_edge("finalizer", END)
    return graph.compile()


def _positive_int(value: int, *, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, default)


def _non_negative_int(value: int, *, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, default)


def run_review_workflow(
    paper_path: str,
    *,
    settings: Settings | None = None,
    batch_label: str | None = None,
) -> FinalReviewReport:
    """Run the full review workflow for a single markdown paper."""

    effective_settings = settings or Settings.from_env()
    runtime = WorkflowRuntime(effective_settings, paper_path, batch_label=batch_label)
    runtime.execution_log.start_run()
    runtime.logger.log(
        event_type="run_started",
        message="Starting auto review workflow.",
        node_name="workflow",
        input_refs=[paper_path],
    )
    graph = build_auto_review_graph(runtime, effective_settings)
    initial_state: AutoReviewGraphState = {
        "paper_input": runtime.make_paper_input(),
        "discussion_history": [],
        "score_calibration_round_id": 1,
        "sc_revision_count": 0,
    }
    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        runtime.execution_log.fail_run(error_type=type(exc).__name__, error=str(exc))
        runtime.logger.log(
            event_type="run_failed",
            message=f"Workflow failed: {exc}",
            node_name="workflow",
            metadata={"error_type": type(exc).__name__},
        )
        raise
    runtime.logger.log(
        event_type="node_exit",
        message="Workflow completed.",
        node_name="workflow",
        output_refs=[final_state["final_output"].output_path or ""],
    )
    runtime.execution_log.complete_run(final_output=final_state["final_output"].output_path or "")
    return final_state["final_output"]
