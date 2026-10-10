from __future__ import annotations

from typing import Literal

from pydantic import Field

from agent.schemas.common import EvidenceRef, StrictBaseModel


class ReviewScores(StrictBaseModel):
    rating: int
    confidence: int


class ReviewRoundResult(StrictBaseModel):
    round_id: int
    review_stage: Literal["initial_review", "revision_review", "verification_review"]
    summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    required_author_actions: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    scores: ReviewScores
    recommendation: str
    recommendation_rationale: str
    discussion_update: str | None = None
    confidence_explanation: str
    needs_author_response: bool = True
