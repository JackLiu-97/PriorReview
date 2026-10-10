# Review agents

The review runtime uses LangGraph for routing and DeepAgents/LangChain for role execution. The entry point is `agent.workflows.auto_review_graph.run_review_workflow`.

## Roles

| Role | Responsibility |
| --- | --- |
| `reviewer` | Initial assessment, revision after the author response, and optional verification after calibration. |
| `author` | A manuscript-grounded response to the latest review. |
| `sc` | Rating/recommendation consistency checks and an advisory revision status. |
| `finalizer` | A structured `FinalReviewReport` based on the final reviewer output. |
| `data_specialist` | Role-specific historical SQL/RAG evidence and concise summaries. |

Data specialists can independently browse reference code and documentation, write Python/SQL analysis scripts, execute and debug them, inspect complete results, and summarize their findings. They have `ls`, `glob`, `grep`, `read_file`, `write_file`, `edit_file`, and `execute`, alongside `write_sql_artifact`, `run_sql_artifact`, `run_rag_query`, and `write_data_summary`. Each invocation binds the current run/role/round workspace from runtime configuration.

The finalizer's specialist also has coding tools. Its evidence task remains style guidance from `papers` and `reviews`; its SQL helper enforces those tables and its RAG helper is absent. Arbitrary local code is governed by the operating-system and database account privileges, so helper checks are not a sandbox. Main role agents keep their existing tool restrictions. See [autonomous data analysis](../docs/autonomous-data-agent.md) for the execution model and examples.

## Source map

- `workflows/auto_review_graph.py`: stage transitions and round limits.
- `core/runtime.py`: per-paper runtime, role payloads, retries, and finalization.
- `core/deep_agents.py`: model wrappers and subagent configuration.
- `core/prompts.py`: active role prompts.
- `core/compiled_data_subagent.py`: controlled historical-data tools.
- `core/data_agent_workspace.py`: role/round workspace binding, reference mounts, local command execution, and full execution logs.
- `core/sql_context.py`, `core/rag_context.py`: SQL and semantic retrieval.
- `core/embedding_client.py`: query embeddings; bundled here to keep the runtime independent of the data-engineering project.
- `schemas/`: review, response, calibration, and final-report models.
- `logging/`: input snapshots, outputs, timelines, and traces.

## Optional external research

`external_researcher` and its tools are implemented but not enabled in the default workflow. It is reviewer-only and provides Semantic Scholar, arXiv, and GitHub queries with a configured date cutoff. The relevant implementation is in `core/compiled_external_research_subagent.py` and `tools/external_research.py`.

Keep these modules when using the existing wrappers: they are imported even when external research is disabled. Tool configuration is defined in `core/settings.py`.

See [configuration](../docs/configuration.md), [historical priors](../docs/historical-priors.md), and [test instructions](../tests/README.md).
