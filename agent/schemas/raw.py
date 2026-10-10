from __future__ import annotations

from pydantic import Field

from agent.schemas.common import StrictBaseModel


class AgentRawResult(StrictBaseModel):
    role: str
    round_id: int
    stage: str
    content: str
    artifact_refs: list[str] = Field(default_factory=list)
    state_artifact: str | None = None
