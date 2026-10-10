from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _enforce_strict_json_schema(schema: dict[str, Any]) -> None:
    properties = schema.get("properties")
    if isinstance(properties, dict):
        schema["required"] = list(properties)
    schema["additionalProperties"] = False


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_enforce_strict_json_schema)


class EvidenceRef(StrictBaseModel):
    source: str
    snippet: str
    reason: str | None = None


class QuerySpec(StrictBaseModel):
    role: Literal["reviewer", "author", "sc"]
    objective: str
    keywords: list[str] = Field(default_factory=list)
    sql_focus: list[str] = Field(default_factory=list)
    rag_focus: list[str] = Field(default_factory=list)


class SQLQueryCandidate(StrictBaseModel):
    name: str
    sql: str
    reason: str


class SQLQueryPlan(StrictBaseModel):
    queries: list[SQLQueryCandidate] = Field(default_factory=list)


class ScoreDistributionSummary(StrictBaseModel):
    sample_size: int = 0
    mean_rating: float | None = None
    median_rating: float | None = None
    avg_confidence: float | None = None
    top_decisions: list[str] = Field(default_factory=list)


class DataSubAgentTask(StrictBaseModel):
    task_id: str
    agent_role: Literal["reviewer", "author", "sc"]
    objective: str
    query_spec: QuerySpec
    input_refs: list[str] = Field(default_factory=list)
    allowed_read_roots: list[str] = Field(default_factory=list)
    allowed_write_root: str
    expected_artifacts: list[str] = Field(default_factory=list)


class DataSubAgentResult(StrictBaseModel):
    task_id: str
    status: Literal["success", "failed"]
    summary: str
    artifact_paths: list[str] = Field(default_factory=list)
    structured_result: dict = Field(default_factory=dict)
    error_message: str | None = None
