from __future__ import annotations

from pydantic import Field

from agent.schemas.common import EvidenceRef, StrictBaseModel


class PointResponse(StrictBaseModel):
    question: str
    response: str
    status: str


class AuthorRebuttalResult(StrictBaseModel):
    round_id: int
    overall_response: str
    point_by_point_responses: list[PointResponse] = Field(default_factory=list)
    revision_notes: list[str] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    can_further_respond: bool
    stop_reason: str | None = None
