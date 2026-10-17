# Historical-prior interface

PriorReview queries a prepared historical-prior resource during review. The **ICLR** and **NIPS/NeurIPS** PRD archives are released separately through Baidu Netdisk; see [Dataset & Download in the README](../README.md#dataset) for both download links and extraction codes. This code package supplies the runtime clients; the corpus archives, crawl/extraction scripts, database loaders, and vector-generation pipeline are not bundled in the code directory.

The manuscript describes the Peer-Review Prior Dataset (PRD), built from ICLR and NeurIPS review records. Its reported retained-corpus statistics are 76,282 papers, 294,332 reviews, and 3,382,889 knowledge units. These counts describe the study's corpus, not files bundled in this repository.

## Connecting an existing store

1. Make the prepared PostgreSQL database available to the review process.
2. Ensure the relational and vector tables below are visible on the connection's search path.
3. Set the `PRIORREVIEW_DB_*` and `OPENAI_EMBEDDING_*` values in `.env`.
4. Use the same embedding model and vector dimension used to build the store.

No database is created or populated by `uv sync` or the review CLI. A Markdown input and model key alone do not reproduce the historical-prior experiments.

## Relational tables

The SQL helper exposes four tables. Their detailed column descriptions are embedded in [`SQL_SCHEMA_CONTEXT`](../agent/core/sql_context.py).

| Table | Contents | Main relationships |
| --- | --- | --- |
| `papers` | Title, venue, year, source identifiers, and decision. | `id` identifies the source paper. |
| `reviews` | Review text, strengths, weaknesses, questions, ratings, and confidence. | `paper_id` links to `papers.id`. |
| `meta_reviews` | Meta-review text and associated decision information. | `paper_id` links to `papers.id`. |
| `comments` | Author responses and discussion records. | `paper_id` links to `papers.id`; source note/reply fields preserve discussion relationships. |

Use a database account restricted to read access to these resources. Runtime SQL helpers validate queries through [`SQLRepository`](../agent/core/sql_context.py), and SQL/RAG connections request read-only transactions. Autonomous scripts can also query through Python, so database account privileges remain the enforceable boundary. The finalizer does not query the historical store; it formats the supplied review and summarizes the discussion history directly.

## Vector tables

[`RAGRepository`](../agent/core/rag_context.py) reads three tables using PostgreSQL's pgvector cosine-distance operator:

| Table | Prior type | Type-specific fields |
| --- | --- | --- |
| `review_strength` | Positive review observations | `full_text`, `importance` |
| `review_weakness` | Critical review observations | `full_text`, `severity` |
| `rebuttal_pair` | Reviewer-question / author-response pairs | `question_text`, `response_text`, `reviewer_satisfaction` |

The retrieval query expects these common columns:

```text
id, conference, year, paper_id, review_id, source_file, title,
decision, accepted, score_numeric, keywords_text, label_text,
keywords_embedding, label_embedding
```

`keywords_embedding` supports topical retrieval; `label_embedding` supports retrieval of review patterns. Both vector columns must match the query embedding dimension. The runtime supports filters for conference, year, and acceptance status. See the source for the exact query and field mapping.

## Check the connection

These commands use the configured external services:

```shell
uv run python -m agent.cli.sql_runner --sql "SELECT id, title FROM papers LIMIT 3"
uv run python -m agent.cli.rag_runner --query "missing ablation study" --table review_weakness --top-k 3
```

The RAG helper reports availability and errors in its JSON response. An unavailable database or embedding service is not evidence that a historical pattern is absent.

The repository's unit and workflow tests use mock services. Passing those tests verifies code behavior without confirming access to a real prior database.
