from __future__ import annotations

from pathlib import Path

from agent.schemas.paper import PaperInput, ParsedPaper


def parse_markdown_paper(paper_input: PaperInput) -> ParsedPaper:
    path = Path(paper_input.file_path)
    raw_markdown = path.read_text(encoding="utf-8")

    return ParsedPaper(
        paper_id=paper_input.paper_id,
        raw_markdown=raw_markdown,
    )
