# Autonomous data analysis

The parent role delegates an evidence question, compact context, constraints, and expected deliverables. The data specialist chooses a query strategy, writes and executes analysis code when useful, inspects results, repairs errors, and returns findings with provenance and limitations. Large data stays in artifacts instead of being copied into the parent's context.

Code execution is enabled by default. It adds capabilities to each existing `data_specialist`; it does not change the reviewer/author/SC workflow or enable the optional external researcher.

## Tools and paths

| Purpose | Tools | Paths |
| --- | --- | --- |
| Explore and inspect files | `ls`, `glob`, `grep`, `read_file` | `/` is the assigned data workspace. Read files in chunks using `offset` and `limit`. |
| Write and revise code | `write_file`, `edit_file` | For example, `/analysis.py`, `/queries/cases.sql`, `/analysis.json`. |
| Execute and debug | `execute` | Commands start in the real assigned directory. Use `python analysis.py`, with a relative path. |
| Read project references | File tools | `/project/agent/`, `/project/docs/`, `/project/examples/`; these mounts are read-only through file tools. |
| Query the historical store | `write_sql_artifact`, `run_sql_artifact`, `run_rag_query` | Existing data helpers accept relative artifact filenames. |
| Deliver results | `write_data_summary` | Saves `data_summary.md`; return a short evidence summary to the parent. |

Assignments come from trusted runtime configuration, not from paths inferred from model messages. The normal directory is `artifacts/data/<role>_<round-or-stage>/` within the current run. Different runs, roles, and rounds have separate directories. Multiple delegations in the same role/round share that directory, so use distinct filenames for distinct investigations and preserve meaningful failed attempts.

Virtual file paths and shell paths differ: `write_file('/analysis.py', ...)` writes `analysis.py` in the workspace; run it with `execute('python analysis.py')`. `/project/agent/` is a file-tool mount, not an absolute host path. The command environment puts the workflow's Python first on `PATH` and the project on `PYTHONPATH`, so scripts can import `agent.core.*` without changing directories. Windows commands use `cmd.exe`; POSIX commands use the system shell. Prefer saved Python scripts over complex shell quoting.

Each command starts a fresh process. Persist reusable work in files. Successful output, errors, and truncated previews include paths to complete stdout/stderr logs under `/_execution/`. Python uses UTF-8; output from other native programs may require reading the log with that program's encoding. Check exit codes; read the logs, revise the script, and rerun as needed. Commands exceeding the configured timeout have their process tree terminated. The task also has a configurable graph-step budget.

## Example: query and analyze in Python

The specialist can write the following as `/analysis.py` and run `python analysis.py`. It uses existing project clients and the injected service configuration; credentials are never embedded in the script. This example requires a configured prior database.

```python
import json
from pathlib import Path
from statistics import mean

from agent.core.settings import Settings
from agent.core.sql_context import SQLRepository
from agent.schemas.common import SQLQueryCandidate, SQLQueryPlan

repository = SQLRepository(Settings.from_env())
result = repository.execute_query_plan(
    SQLQueryPlan(queries=[SQLQueryCandidate(
        name="ratings",
        sql="SELECT rating_numeric FROM reviews WHERE rating_numeric IS NOT NULL LIMIT 200",
        reason="Inspect a limited historical rating sample",
    )]),
    max_rows=200,
)
Path("ratings.result.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
if not result["available"] or any(q.get("error") for q in result["executed_queries"]):
    raise RuntimeError("Query failed; inspect ratings.result.json before continuing")
values = [row["rating_numeric"] for row in result["results"]["ratings"]]
analysis = {"sample_size": len(values), "sample_mean": mean(values) if values else None}
Path("analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
print(json.dumps(analysis))
```

The sample is limited and is not a corpus-wide estimate. The specialist should select queries appropriate to the parent's task, document selection criteria, and state the scope of its conclusions. Existing CLI modules (`agent.cli.sql_runner`, `agent.cli.rag_runner`) are also executable through `python -m ...`. General specialists can reuse `RAGRepository` in scripts for retrieval and subsequent analysis.

The SQL helper's 5,000-character preview and the RAG helper's short snippets remain concise. They are no longer the only accessible evidence: use `read_file` or Python to inspect the complete saved result artifact. Saved RAG documents still have the snippet length requested from retrieval; if more source text is required, request a larger snippet or query the relevant source fields.

## Role scope and execution boundary

Reviewer, author, and SC specialists may use their SQL/RAG resources for the assigned evidence task. The finalizer's specialist may write code and inspect results but should use only `papers` and `reviews` for review-writing style, preserving the source review's judgments. Its controlled SQL helper enforces that table scope, and no `run_rag_query` tool is exposed to it.

**Commands execute locally with the host user's privileges. This is not an OS sandbox.** Read-only reference mounts and directory checks apply to file tools; arbitrary shell/Python code can access other host paths or the network. Instructions require generated files to stay in the workspace and prohibit modifying the source database, project, environment, installing dependencies, or starting background services. Those instructions are not OS-level access controls. No approval prompt is added to the autonomous loop.

The command environment includes database and embedding configuration, not main/finalizer model keys or unrelated environment tokens. It disables automatic dotenv loading in subprocesses. This reduces accidental credential exposure but cannot prevent local code from deliberately reading files. SQL/RAG helper connections and default PostgreSQL command sessions request read-only transactions; use a `SELECT`-only database account for enforcement. Role-specific hard database isolation requires corresponding database privileges, not just a Python table allowlist. Use an isolated host/container if code must be contained at the OS level.

Set `PRIORREVIEW_DATA_AGENT_CODE_EXECUTION_ENABLED=false` to remove coding/file tools and switch the prompt back to fixed-helper mode. Timeout and step settings are documented in [configuration.md](configuration.md).

## Offline verification

```shell
uv run python -m unittest tests.unit.test_data_agent_workspace tests.e2e.test_autonomous_data_subagent -v
uv run python -m unittest discover -s tests -v
```

The integration test delegates through the actual `task` tool into a scripted model with real file and process tools. A mocked SQL service supplies a result larger than the preview. The specialist writes code, encounters a real Python error, edits and reruns the code, analyzes the complete result, reads its output, and writes a summary. No live model, embedding service, or database is needed for this check.
