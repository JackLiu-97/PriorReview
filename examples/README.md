# Examples

- `sample_paper.md` is an explicitly synthetic manuscript for input discovery and configured review runs.
- `review_format.example.json` is a hand-written output-format illustration matching `FinalReviewReport`. Its text and scores are illustrative; it is not a model run or an experimental result.

Check input discovery without external service calls:

```shell
uv run python main.py examples/sample_paper.md --dry-run
```

After configuring model and prior services:

```shell
uv run python main.py examples/sample_paper.md --output-dir outputs
```

The generated report may differ from the format illustration. Use your own Markdown manuscript for a substantive review.
