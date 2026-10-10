from __future__ import annotations

from pydantic import Field

from agent.schemas.common import StrictBaseModel


class PaperInput(StrictBaseModel):
    paper_id: str
    source_type: str = "markdown"
    file_path: str
    title: str | None = None
    abstract: str | None = None
    metadata: dict = Field(default_factory=dict)


class ParsedPaper(StrictBaseModel):
    paper_id: str
    raw_markdown: str

