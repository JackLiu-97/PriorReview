from __future__ import annotations

from pydantic import BaseModel, Field


class RunLogEvent(BaseModel):
    ts: str
    run_id: str
    paper_stem: str
    batch_label: str | None = None
    node_name: str
    agent_role: str | None = None
    event_type: str
    message: str
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
