# Release preparation notes

Prepared on 2026-10-09 from the existing PriorReview development project.

## Scope

Included: the review CLI, active Agent implementation and prompts, runtime SQL/RAG clients, report schemas, logging, selected tests, configuration examples, and the existing manuscript framework figure.

Excluded: historical-data crawling/extraction/import pipelines, experiment evaluation scripts, preprocessing utilities, unused application placeholders, `prompts_v0.py`, real datasets and run outputs, local environments, credentials, and development Git history.

## Changes made for this package

- Moved the existing embedding client into `agent/core/embedding_client.py` and changed its import in `rag_context.py`. The client implementation was preserved.
- Removed personal-machine command examples from `main.py`; CLI behavior was preserved.
- Removed data-ingestion-only dependency declarations and explicitly declared `httpx` and `pydantic`, which the runtime imports directly. Regenerated `uv.lock` while retaining the versions of dependencies shared with the original lockfile.
- Added `.env.example`, a focused `.gitignore`, synthetic input/output-format examples, and concise usage documents.
- Updated the copied workflow-test mocks for the current streaming API and artifact fields, and aligned obsolete prompt assertions with the current prompts. Added package markers so standard `unittest` discovery finds the selected tests.

The initial release preparation preserved the workflow, active prompts, role implementations, data-agent tools, and finalizer behavior. The subsequent autonomous-data update below expands the data specialists' execution capabilities.

## Autonomous data-agent update — 2026-10-09

- Enabled file browsing/reading/writing/search and local Python/shell execution for all data-specialist roles, with runtime-bound workspaces and read-only source/documentation mounts through file tools.
- Retained SQL/RAG helpers; complete result artifacts are now readable and can be analyzed with generated code. Prompts allow independent query planning, debugging, additional evidence gathering, and concise summaries.
- Added command timeouts with process-tree cleanup and persistent stdout/stderr. Added settings for execution enablement, command timeout, and task step budget.
- Required trusted runtime directory assignments for live data tools and corrected trace assignment in bare LangGraph runtime hooks.
- Requested read-only transactions for SQL/RAG repository connections and default PostgreSQL command sessions; documented that local execution is not an OS sandbox and database accounts enforce access rights.
- Corrected the SQL CLI's exit status for unavailable databases, so autonomous scripts can distinguish failed queries from successful empty results.

See [autonomous data analysis](autonomous-data-agent.md) for examples and the local execution boundary. The main role agents retain their existing filesystem restrictions; the finalizer's data task remains style guidance. The dependency lockfile is unchanged.

Validated on Windows with the project's Python 3.13.11 environment: all 79 tests passed with `uv run python -m unittest discover -s tests -v`; `uv lock --check` passed; the synthetic-paper `--dry-run` found one input. Syntax checks and local links in the updated documentation passed. No live model, embedding, or database calls were made.

## Initial package validation

Validated on Windows with Python 3.13.11 in a fresh temporary environment outside the release directory.

| Check | Outcome |
| --- | --- |
| `uv sync --locked` | Passed; clean environment installation. |
| `uv lock --check` | Passed. |
| `uv run python -m unittest discover -s tests -v` | 65 tests passed. |
| `uv run python main.py --help` | Passed. |
| `uv run python main.py examples/sample_paper.md --dry-run` | Found the synthetic manuscript without service calls. |
| Core workflow and SQL/RAG module imports | Resolved from this standalone package. |
| Illustrative report | Validated against `FinalReviewReport`. |
| Documentation | Local links checked; README rendered and visually inspected in a local GitHub-style preview. |
| Release contents | Checked for excluded folders, personal absolute paths, credential patterns, and required files accidentally ignored by Git. |

Live model calls, a real historical-prior database, and full manuscript-result reproduction were not exercised. The included example is explicitly synthetic.

The software license, author list, and public paper link remain to be supplied. No license was selected on the author's behalf.
