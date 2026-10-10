from __future__ import annotations

from typing import Literal

from agent.schemas.common import ScoreDistributionSummary, StrictBaseModel


class ScoreCalibrationResult(StrictBaseModel):
    round_id: int
    deviation_detected: bool
    deviation_summary: str
    historical_distribution: ScoreDistributionSummary
    reviewer_evidence_sufficient: bool
    action: Literal["normal", "needs_reviewer_revision"]
    action_reason: str
    meta_summary: str
