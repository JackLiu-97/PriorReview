from __future__ import annotations

import unittest

from agent.core.rag_context import DEFAULT_RAG_EMBEDDING_DIMENSIONS, DEFAULT_RAG_EMBEDDING_MODEL, RAGRepository
from agent.core.settings import Settings


class FakeEmbeddingClient:
    instances: list["FakeEmbeddingClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)

    def embed_missing(self, texts, store, *, stage_name="default", logger=None):
        items = list(texts)
        for text in items:
            store[" ".join(text.split())] = [0.1, 0.2, 0.3]
        return {"requested": len(items), "reused": 0, "missing": 0, "api_batches": 0}


class FakeCursor:
    def __init__(self):
        self.executed: list[tuple[str, list[object]]] = []
        self.rows: list[dict[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql: str, params: list[object]) -> None:
        self.executed.append((sql, params))
        embedding_field = params[3]
        self.rows = [
            {
                "source_table": params[1],
                "source_kind": params[2],
                "embedding_field": embedding_field,
                "id": "row-1",
                "conference": "ICLR",
                "year": 2024,
                "paper_id": "paper-1",
                "review_id": "review-1",
                "source_file": "paper.json",
                "title": "A test paper",
                "decision": "Accept",
                "accepted": True,
                "score_numeric": 8.0,
                "keywords_text": "novelty, retrieval",
                "label_text": "novel retrieval weakness",
                "full_text": "The reviewer discusses retrieval novelty in detail.",
                "question_text": None,
                "response_text": None,
                "severity": "medium",
                "importance": None,
                "reviewer_satisfaction": None,
                "distance": 0.2 if embedding_field == "label" else 0.1,
            }
        ]

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self):
        self.cursor_obj = FakeCursor()
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def close(self) -> None:
        self.closed = True


class TestRAGRepository(RAGRepository):
    def __init__(self, settings: Settings):
        super().__init__(settings, embedding_client_factory=FakeEmbeddingClient)
        self.connection = FakeConnection()

    def _connect(self):
        return self.connection


class RAGContextTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeEmbeddingClient.instances.clear()
        settings = Settings(db_host="127.0.0.1", db_port=5432, db_name="priorreview")
        self.repository = TestRAGRepository(settings)

    def test_query_embeds_with_configured_model_and_searches_pgvector(self) -> None:
        result = self.repository.query(
            " review prior about novelty ",
            top_k=1,
            conference="ICLR",
            year=2024,
            accepted=True,
            tables=["review_weakness"],
        )

        self.assertTrue(result["available"])
        self.assertEqual(result["embedding_model"], DEFAULT_RAG_EMBEDDING_MODEL)
        self.assertEqual(result["embedding_dimensions"], DEFAULT_RAG_EMBEDDING_DIMENSIONS)
        self.assertEqual(FakeEmbeddingClient.instances[0].kwargs["model"], DEFAULT_RAG_EMBEDDING_MODEL)
        self.assertEqual(FakeEmbeddingClient.instances[0].kwargs["dimensions"], DEFAULT_RAG_EMBEDDING_DIMENSIONS)

        executed = self.repository.connection.cursor_obj.executed
        self.assertEqual(len(executed), 2)
        first_sql, first_params = executed[0]
        self.assertIn("review_weakness", first_sql)
        self.assertIn("label_embedding <=> q.embedding", first_sql)
        self.assertEqual(first_params[0], "[0.1,0.2,0.3]")
        self.assertEqual(first_params[4:7], ["ICLR", 2024, True])

        documents = result["documents"]
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["distance"], 0.1)
        self.assertEqual(documents[0]["matched_embedding_fields"], ["keywords", "label"])
        self.assertIn("retrieval novelty", documents[0]["snippet"])
        self.assertTrue(self.repository.connection.closed)

    def test_empty_query_returns_invalid_query_without_api_or_db(self) -> None:
        result = self.repository.query("   ")

        self.assertFalse(result["available"])
        self.assertEqual(result["mode"], "invalid_query")
        self.assertEqual(FakeEmbeddingClient.instances, [])
        self.assertFalse(self.repository.connection.closed)


if __name__ == "__main__":
    unittest.main()
