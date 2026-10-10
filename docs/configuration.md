# Configuration

Copy the root `.env.example` to `.env` and fill in the services you will use. Run the CLI from the repository root. Process environment variables take precedence over the local `.env` file for `Settings` values.

## Model services

| Variables | Purpose |
| --- | --- |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | Main reviewer, author, and score-calibration model. |
| `FINALIZER_OPENAI_API_KEY`, `FINALIZER_OPENAI_BASE_URL`, `FINALIZER_OPENAI_MODEL` | Model used to produce the structured final report. Configure separately; there is no fallback to the main-model configuration. |
| `PRIORREVIEW_TEMPERATURE` | Main-agent temperature; default `0.2`. |
| `FINALIZER_OPENAI_TEMPERATURE` | Finalizer temperature; default `0.0`. |
| `PRIORREVIEW_AGENT_MAX_ATTEMPTS` | Main-agent retry attempts; default `3`. |
| `FINALIZER_MAX_ATTEMPTS` | Finalizer retry attempts; default `3`. |

Use provider-supported model identifiers. The finalizer must support the structured-output behavior used by the implementation. The same provider/model can be entered in both sets of variables.

## Historical priors

The current runtime reads an existing PostgreSQL database. Semantic retrieval uses pgvector and an OpenAI-compatible embedding API.

| Variables | Purpose |
| --- | --- |
| `PRIORREVIEW_DB_HOST`, `PRIORREVIEW_DB_PORT` | Database host and port; defaults `127.0.0.1` and `5432`. |
| `PRIORREVIEW_DB_USER`, `PRIORREVIEW_DB_PASSWORD`, `PRIORREVIEW_DB_NAME` | Database credentials and database name. |
| `OPENAI_EMBEDDING_API_KEY`, `OPENAI_EMBEDDING_BASE_URL` | Query-embedding service; configured separately from the review model. |
| `OPENAI_EMBEDDING_MODEL` | Query model; implementation default `openai/text-embedding-3-small`. |
| `OPENAI_EMBEDDING_DIMENSIONS` | Query-vector dimensions; default `1536`. Must match stored vectors. |

Embedding configuration is loaded when the embedding module is imported; set it before starting the process. Database aliases `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, and `PGDATABASE` are also supported.

See [historical-priors.md](historical-priors.md) for the required table and field interface. The data files, vector store, and construction/import pipeline are not bundled.

## Workflow and outputs

| Variable | Default | Meaning |
| --- | ---: | --- |
| `PRIORREVIEW_MAX_REVIEWER_AUTHOR_ROUNDS` | `2` | Initial review plus one revision after an author response. `1` skips the response stage; larger values add more exchanges. |
| `PRIORREVIEW_MAX_SC_REREVIEW_ROUNDS` | `1` | At most one reviewer verification after calibration. `0` disables it; larger values allow additional calibration cycles. |
| `PRIORREVIEW_REVIEWER_DATA_AGENT_ENABLED` | `true` | Historical evidence delegation for the reviewer. |
| `PRIORREVIEW_AUTHOR_DATA_AGENT_ENABLED` | `true` | Historical evidence delegation for the author. |
| `PRIORREVIEW_SC_DATA_AGENT_ENABLED` | `true` | Historical evidence delegation for score calibration. |
| `PRIORREVIEW_RESULTS_DIR` | `data/results` | Output root; overridden for one process by `--output-dir`. |
| `PRIORREVIEW_CONSOLE_LOG` | `true` | Console event logging. |

The three data-agent switches apply to their named roles. They are not a global database-disable switch. The current finalizer also uses SQL-based style guidance. Preserve the default round settings when comparing with the default workflow described in the manuscript.

## Autonomous data specialists

| Variable | Default | Meaning |
| --- | ---: | --- |
| `PRIORREVIEW_DATA_AGENT_CODE_EXECUTION_ENABLED` | `true` | Add file browsing/reading/writing/search and local command execution to every enabled data specialist, including the finalizer's specialist. `false` retains only its fixed SQL/RAG/summary helpers. |
| `PRIORREVIEW_DATA_AGENT_COMMAND_TIMEOUT_SECONDS` | `120` | Default and maximum timeout for one command, from 1 to 3600 seconds. An individual command may request a shorter timeout. |
| `PRIORREVIEW_DATA_AGENT_MAX_STEPS` | `100` | Graph step budget for one delegated data task, from 10 to 1000. Model and tool nodes each consume steps; this is not a SQL-query count. |

The execution switch controls tools and prompts together. It does not enable a role whose data-agent switch is off. Code runs using the workflow's Python environment with the role/round data directory as its initial working directory. Complete command stdout/stderr are stored under that directory's `_execution/` folder.

Local execution has the host user's permissions; virtual file paths do not isolate commands. SQL/RAG repository connections request read-only transactions, and the command environment defaults PostgreSQL sessions to read-only. Use a database account restricted to `SELECT` for an enforceable database boundary: arbitrary code can open its own connections or change session defaults. See [the analysis workflow and scope](autonomous-data-agent.md).

## Input and batch options

The root CLI accepts `.md` and `.markdown`. It discovers papers recursively unless `--no-recursive` is used.

```shell
uv run python main.py path/to/papers --limit 1 --dry-run
uv run python main.py path/to/papers --skip-existing --continue-on-error --jobs 2
```

`--skip-existing` skips papers for which a final report already exists under the selected results root. Keep paper filename stems unique within a batch, because result directories use those stems.

## Optional external research

External research tools are retained for development but are not enabled by the default workflow. Their configuration is not needed for the standard review path. Refer to [agent/README.md](../agent/README.md) for their scope. Setting an API key alone does not enable them.
