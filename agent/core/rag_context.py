from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Iterable

import psycopg
from psycopg.rows import dict_row

from agent.core.settings import Settings
from agent.core.embedding_client import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    OpenAIEmbeddingClient,
    normalize_text,
    vector_to_pg,
)


DEFAULT_RAG_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
DEFAULT_RAG_EMBEDDING_DIMENSIONS = int(DEFAULT_EMBEDDING_DIMENSIONS or 1536)

RAG_EMBEDDING_FIELDS = ("label", "keywords")


@dataclass(frozen=True)
class RAGTableConfig:
    name: str
    kind: str
    full_text_sql: str
    question_text_sql: str
    response_text_sql: str
    severity_sql: str
    importance_sql: str
    reviewer_satisfaction_sql: str


RAG_TABLE_CONFIGS: dict[str, RAGTableConfig] = {
    "review_weakness": RAGTableConfig(
        name="review_weakness",
        kind="weakness",
        full_text_sql="t.full_text",
        question_text_sql="NULL::TEXT",
        response_text_sql="NULL::TEXT",
        severity_sql="t.severity",
        importance_sql="NULL::TEXT",
        reviewer_satisfaction_sql="NULL::TEXT",
    ),
    "review_strength": RAGTableConfig(
        name="review_strength",
        kind="strength",
        full_text_sql="t.full_text",
        question_text_sql="NULL::TEXT",
        response_text_sql="NULL::TEXT",
        severity_sql="NULL::TEXT",
        importance_sql="t.importance",
        reviewer_satisfaction_sql="NULL::TEXT",
    ),
    "rebuttal_pair": RAGTableConfig(
        name="rebuttal_pair",
        kind="rebuttal_pair",
        full_text_sql="NULL::TEXT",
        question_text_sql="t.question_text",
        response_text_sql="t.response_text",
        severity_sql="t.severity",
        importance_sql="NULL::TEXT",
        reviewer_satisfaction_sql="t.reviewer_satisfaction",
    ),
}

RAG_TABLE_NAMES = tuple(RAG_TABLE_CONFIGS)

RAG_QUERY_CONTEXT = """
Real vector RAG is available through PostgreSQL + pgvector.

Vector tables:
- review_weakness
- review_strength
- rebuttal_pair

Each table has:
- keywords_text / keywords_embedding
- label_text / label_embedding

The RAG helper embeds retrieval queries with openai/text-embedding-3-small
through an OpenAI-compatible API, then ranks rows with pgvector cosine distance.
Use RAG for semantic historical priors; use SQL for exact statistics and joins.
""".strip()


EmbeddingClientFactory = Callable[..., OpenAIEmbeddingClient]


class RAGRepository:
    """Read-only pgvector retrieval over historical review/rebuttal priors."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_RAG_EMBEDDING_DIMENSIONS,
        embedding_client_factory: EmbeddingClientFactory = OpenAIEmbeddingClient,
    ) -> None:
        self.settings = settings
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.embedding_client_factory = embedding_client_factory

    def query(
        self,
        query: str,
        *,
        top_k: int = 8,
        per_source_limit: int | None = None,
        conference: str | None = None,
        year: int | None = None,
        accepted: bool | None = None,
        tables: Iterable[str] | None = None,
        embedding_fields: Iterable[str] | None = None,
        max_snippet_chars: int = 800,
    ) -> dict[str, Any]:
        """Embed one query and search the configured pgvector tables."""

        normalized_query = normalize_text(query)
        if not normalized_query:
            return self._error_payload(query=query, mode="invalid_query", error="RAG query text is empty.")

        top_k = max(1, int(top_k))
        per_source_limit = max(1, int(per_source_limit or top_k))
        table_names = self._resolve_tables(tables)
        field_names = self._resolve_embedding_fields(embedding_fields)

        try:
            query_vector = self._embed_query(normalized_query)
        except Exception as exc:  # noqa: BLE001
            return self._error_payload(query=query, mode="embedding_error", error=str(exc))

        try:
            conn = self._connect()
        except Exception as exc:  # pragma: no cover - external DB dependency
            return self._error_payload(query=query, mode="unavailable", error=str(exc))

        rows: list[dict[str, Any]] = []
        searched_sources: list[dict[str, Any]] = []
        try:
            with conn.cursor() as cur:
                vector_literal = vector_to_pg(query_vector)
                for table_name in table_names:
                    config = RAG_TABLE_CONFIGS[table_name]
                    for field_name in field_names:
                        embedding_column = self._embedding_column(field_name)
                        source_rows = self._query_source(
                            cur,
                            config=config,
                            embedding_field=field_name,
                            embedding_column=embedding_column,
                            vector_literal=vector_literal,
                            per_source_limit=per_source_limit,
                            conference=conference,
                            year=year,
                            accepted=accepted,
                        )
                        rows.extend(source_rows)
                        searched_sources.append(
                            {
                                "table": table_name,
                                "embedding_field": field_name,
                                "embedding_column": embedding_column,
                                "row_count": len(source_rows),
                            }
                        )
        except Exception as exc:  # noqa: BLE001
            return self._error_payload(query=query, mode="query_error", error=str(exc))
        finally:
            conn.close()

        documents = self._rank_documents(rows, top_k=top_k, max_snippet_chars=max_snippet_chars)
        return {
            "available": True,
            "mode": "live",
            "query": normalized_query,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.dimensions,
            "filters": {
                "conference": conference,
                "year": year,
                "accepted": accepted,
                "tables": list(table_names),
                "embedding_fields": list(field_names),
            },
            "searched_sources": searched_sources,
            "documents": documents,
            "error": None,
        }

    def _embed_query(self, query: str) -> list[float]:
        embedder = self.embedding_client_factory(
            model=self.embedding_model,
            dimensions=self.dimensions,
            batch_size=1,
        )
        store: dict[str, list[float]] = {}
        embedder.embed_missing([query], store, stage_name="rag_query")
        vector = store.get(query)
        if vector is None:
            raise RuntimeError("Embedding API returned no vector for the RAG query.")
        return vector

    def _query_source(
        self,
        cur,
        *,
        config: RAGTableConfig,
        embedding_field: str,
        embedding_column: str,
        vector_literal: str | None,
        per_source_limit: int,
        conference: str | None,
        year: int | None,
        accepted: bool | None,
    ) -> list[dict[str, Any]]:
        if vector_literal is None:
            raise RuntimeError("Vector literal is unexpectedly empty.")

        filter_sql, filter_params = self._build_filters(conference=conference, year=year, accepted=accepted)
        sql = f"""
            WITH q AS (SELECT %s::vector AS embedding)
            SELECT
                %s AS source_table,
                %s AS source_kind,
                %s AS embedding_field,
                t.id,
                t.conference,
                t.year,
                t.paper_id,
                t.review_id,
                t.source_file,
                t.title,
                t.decision,
                t.accepted,
                t.score_numeric,
                t.keywords_text,
                t.label_text,
                {config.full_text_sql} AS full_text,
                {config.question_text_sql} AS question_text,
                {config.response_text_sql} AS response_text,
                {config.severity_sql} AS severity,
                {config.importance_sql} AS importance,
                {config.reviewer_satisfaction_sql} AS reviewer_satisfaction,
                (t.{embedding_column} <=> q.embedding) AS distance
            FROM {config.name} t, q
            WHERE t.{embedding_column} IS NOT NULL
            {filter_sql}
            ORDER BY t.{embedding_column} <=> q.embedding
            LIMIT %s
        """
        params = [
            vector_literal,
            config.name,
            config.kind,
            embedding_field,
            *filter_params,
            per_source_limit,
        ]
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    @staticmethod
    def _build_filters(
        *,
        conference: str | None,
        year: int | None,
        accepted: bool | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if conference:
            clauses.append("t.conference = %s")
            params.append(conference)
        if year is not None:
            clauses.append("t.year = %s")
            params.append(year)
        if accepted is not None:
            clauses.append("t.accepted = %s")
            params.append(accepted)
        if not clauses:
            return "", params
        return "AND " + " AND ".join(clauses), params

    def _connect(self):
        connection_attempts: list[dict[str, Any]] = [
            {"host": self.settings.db_host, "port": self.settings.db_port},
        ]
        if self.settings.db_host == "localhost":
            connection_attempts.append({"host": "127.0.0.1", "port": self.settings.db_port})
        if self.settings.db_host == "127.0.0.1":
            connection_attempts.append({"host": "localhost", "port": self.settings.db_port})

        last_error: Exception | None = None
        for attempt in connection_attempts:
            try:
                kwargs: dict[str, Any] = {
                    "dbname": self.settings.db_name,
                    "row_factory": dict_row,
                    "options": "-c default_transaction_read_only=on",
                    "connect_timeout": 10,
                    **attempt,
                }
                if self.settings.db_user:
                    kwargs["user"] = self.settings.db_user
                if self.settings.db_password:
                    kwargs["password"] = self.settings.db_password
                return psycopg.connect(**kwargs)
            except Exception as exc:  # pragma: no cover - external DB dependency
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("No database connection attempts were configured.")

    @staticmethod
    def _resolve_tables(tables: Iterable[str] | None) -> tuple[str, ...]:
        if tables is None:
            return RAG_TABLE_NAMES
        resolved = tuple(dict.fromkeys(tables))
        invalid = sorted(set(resolved) - set(RAG_TABLE_NAMES))
        if invalid:
            raise ValueError(f"Unknown RAG table(s): {', '.join(invalid)}")
        return resolved or RAG_TABLE_NAMES

    @staticmethod
    def _resolve_embedding_fields(embedding_fields: Iterable[str] | None) -> tuple[str, ...]:
        if embedding_fields is None:
            return RAG_EMBEDDING_FIELDS
        resolved = tuple(dict.fromkeys(embedding_fields))
        invalid = sorted(set(resolved) - set(RAG_EMBEDDING_FIELDS))
        if invalid:
            raise ValueError(f"Unknown RAG embedding field(s): {', '.join(invalid)}")
        return resolved or RAG_EMBEDDING_FIELDS

    @staticmethod
    def _embedding_column(field_name: str) -> str:
        return f"{field_name}_embedding"

    @classmethod
    def _rank_documents(
        cls,
        rows: list[dict[str, Any]],
        *,
        top_k: int,
        max_snippet_chars: int,
    ) -> list[dict[str, Any]]:
        best_by_id: dict[tuple[str, str], dict[str, Any]] = {}
        matched_fields: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            source_table = str(row.get("source_table") or "")
            row_id = str(row.get("id") or "")
            key = (source_table, row_id)
            doc = cls._row_to_document(row, max_snippet_chars=max_snippet_chars)
            matched_fields.setdefault(key, set()).add(str(row.get("embedding_field") or ""))
            existing = best_by_id.get(key)
            if existing is None or cls._distance_value(doc) < cls._distance_value(existing):
                best_by_id[key] = doc

        ranked = sorted(best_by_id.values(), key=cls._distance_value)[:top_k]
        for rank, doc in enumerate(ranked, start=1):
            key = (str(doc.get("source_table") or ""), str(doc.get("id") or ""))
            doc["rank"] = rank
            doc["matched_embedding_fields"] = sorted(field for field in matched_fields.get(key, set()) if field)
        return ranked

    @classmethod
    def _row_to_document(cls, row: dict[str, Any], *, max_snippet_chars: int) -> dict[str, Any]:
        distance = cls._normalize_json_value(row.get("distance"))
        similarity = None
        if isinstance(distance, (int, float)):
            similarity = 1.0 - float(distance)

        return {
            "rank": None,
            "id": row.get("id"),
            "source_table": row.get("source_table"),
            "source_kind": row.get("source_kind"),
            "embedding_field": row.get("embedding_field"),
            "distance": distance,
            "similarity": similarity,
            "conference": row.get("conference"),
            "year": row.get("year"),
            "paper_id": row.get("paper_id"),
            "review_id": row.get("review_id"),
            "source_file": row.get("source_file"),
            "title": row.get("title"),
            "decision": row.get("decision"),
            "accepted": row.get("accepted"),
            "score_numeric": cls._normalize_json_value(row.get("score_numeric")),
            "severity": row.get("severity"),
            "importance": row.get("importance"),
            "reviewer_satisfaction": row.get("reviewer_satisfaction"),
            "keywords_text": row.get("keywords_text"),
            "label_text": row.get("label_text"),
            "full_text": row.get("full_text"),
            "question_text": row.get("question_text"),
            "response_text": row.get("response_text"),
            "snippet": cls._build_snippet(row, max_chars=max_snippet_chars),
        }

    @staticmethod
    def _build_snippet(row: dict[str, Any], *, max_chars: int) -> str:
        parts = [
            row.get("label_text"),
            row.get("full_text"),
            row.get("question_text"),
            row.get("response_text"),
        ]
        text = " ".join(str(part) for part in parts if part)
        text = normalize_text(text) or ""
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 3)].rstrip() + "..."

    @staticmethod
    def _distance_value(doc: dict[str, Any]) -> float:
        value = doc.get("distance")
        if isinstance(value, (int, float)):
            return float(value)
        return float("inf")

    @classmethod
    def _normalize_json_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._normalize_json_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._normalize_json_value(item) for item in value]
        if isinstance(value, tuple):
            return [cls._normalize_json_value(item) for item in value]
        if isinstance(value, Decimal):
            return int(value) if value == value.to_integral_value() else float(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return value

    def _error_payload(self, *, query: str, mode: str, error: str) -> dict[str, Any]:
        return {
            "available": False,
            "mode": mode,
            "query": query,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.dimensions,
            "documents": [],
            "error": error,
        }
