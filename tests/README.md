# Tests

From the repository root:

```shell
uv sync --locked
uv run python -m unittest discover -s tests -v
```

The selected suite uses Python's standard `unittest` runner. It covers settings, model HTTP handling, SQL validation, RAG queries, optional external-research tools, and workflow routing/retries with mock services.

Autonomous-data tests also run real local Python commands in temporary workspaces. They exercise file operations, error repair, complete result analysis beyond preview limits, output logs, command timeouts, workspace binding, and the code-execution switch. The delegation test uses a scripted model and a mocked database with the real `task` tool and local execution backend.

No live model, embedding, or database credentials are required. These tests do not measure review quality or reproduce the manuscript experiments.

Useful subsets:

```shell
uv run python -m unittest discover -s tests/unit -v
uv run python -m unittest discover -s tests/e2e -v
```

Evaluation-only tests from the development repository are outside this agent release.
