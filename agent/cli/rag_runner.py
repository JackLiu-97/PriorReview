from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agent.core.rag_context import (
    DEFAULT_RAG_EMBEDDING_DIMENSIONS,
    DEFAULT_RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_FIELDS,
    RAG_TABLE_NAMES,
    RAGRepository,
)
from agent.core.settings import Settings


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser used by Deep Agent data specialists."""

    parser = argparse.ArgumentParser(description="Run read-only pgvector RAG retrieval.")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        default=[],
        help="Retrieval query text. Repeat the flag to add multiple queries.",
    )
    parser.add_argument("--output", help="Optional path to save the JSON result.")
    parser.add_argument("--top-k", type=int, default=8, help="Maximum merged results per query.")
    parser.add_argument(
        "--per-source-limit",
        type=int,
        help="Maximum results per table/embedding-field search. Defaults to --top-k.",
    )
    parser.add_argument("--conference", help="Optional conference filter, for example ICLR or NeurIPS.")
    parser.add_argument("--year", type=int, help="Optional year filter.")
    parser.add_argument(
        "--accepted",
        choices=("true", "false"),
        help="Optional accepted filter. Omit to search accepted and rejected papers.",
    )
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        choices=RAG_TABLE_NAMES,
        help="RAG table to search. Repeat to search multiple tables. Defaults to all.",
    )
    parser.add_argument(
        "--embedding-field",
        choices=("both", *RAG_EMBEDDING_FIELDS),
        default="both",
        help="Embedding column family to search. Defaults to both label and keywords embeddings.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_RAG_EMBEDDING_MODEL,
        help="Embedding model used for the query vector.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_RAG_EMBEDDING_DIMENSIONS,
        help="Embedding dimensions expected by the pgvector tables.",
    )
    parser.add_argument(
        "--max-snippet-chars",
        type=int,
        default=800,
        help="Maximum snippet characters included per retrieved document.",
    )
    return parser


def parse_accepted(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "true"


def resolve_embedding_fields(value: str) -> tuple[str, ...]:
    if value == "both":
        return RAG_EMBEDDING_FIELDS
    return (value,)


def build_payload(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) == 1:
        return results[0]

    documents: list[dict[str, Any]] = []
    for result in results:
        for document in result.get("documents") or []:
            documents.append({"query": result.get("query"), **document})

    return {
        "available": any(result.get("available") for result in results),
        "mode": "multi_query",
        "queries": results,
        "documents": documents,
        "error": None if all(result.get("available") for result in results) else "One or more RAG queries failed.",
    }


def main() -> int:
    """Run one or more real pgvector RAG queries and emit JSON."""

    parser = build_parser()
    args = parser.parse_args()

    queries = [query.strip() for query in args.queries if query and query.strip()]
    if not queries:
        error_payload = {"available": False, "mode": "invalid_query", "documents": [], "error": "At least one --query is required."}
        print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        return 1

    try:
        settings = Settings.from_env()
        repository = RAGRepository(
            settings,
            embedding_model=args.model,
            dimensions=args.dimensions,
        )
        results = [
            repository.query(
                query,
                top_k=args.top_k,
                per_source_limit=args.per_source_limit,
                conference=args.conference,
                year=args.year,
                accepted=parse_accepted(args.accepted),
                tables=args.tables,
                embedding_fields=resolve_embedding_fields(args.embedding_field),
                max_snippet_chars=args.max_snippet_chars,
            )
            for query in queries
        ]
        payload = build_payload(results)
    except Exception as exc:  # noqa: BLE001
        payload = {"available": False, "mode": "runner_error", "documents": [], "error": str(exc)}

    output = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    print(output)
    return 0 if payload.get("available") else 1


if __name__ == "__main__":
    sys.exit(main())
