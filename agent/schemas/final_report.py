from __future__ import annotations

from pydantic import Field

from agent.schemas.common import EvidenceRef, StrictBaseModel
from agent.schemas.review import ReviewScores


class FinalReviewReport(StrictBaseModel):
    review_type: str = "official_review"
    conference_style: str = "ICLR_2026"
    paper_id: str
    title: str
    summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    required_clarifications: list[str] = Field(default_factory=list)
    recommendation: str
    recommendation_rationale: str
    supporting_arguments: list[str] = Field(default_factory=list)
    additional_feedback: list[str] = Field(default_factory=list)
    code_of_ethics: dict = Field(default_factory=lambda: {"potential_violation": False, "explanation": None})
    scores: ReviewScores
    discussion_history_summary: str = ""
    sc_meta_summary: str = ""
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    output_path: str | None = None
