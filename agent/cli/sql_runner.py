from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.core.settings import Settings
from agent.core.sql_context import SQLRepository
from agent.schemas.common import SQLQueryCandidate, SQLQueryPlan


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser used by Deep Agent data specialists."""

    parser = argparse.ArgumentParser(description="Validate and execute a read-only SQL query.")
    parser.add_argument("--sql", help="Inline SQL query text.")
    parser.add_argument("--sql-file", help="Path to a .sql file containing the query.")
    parser.add_argument("--name", default="ad_hoc_query", help="Logical name for the query.")
    parser.add_argument("--reason", default="Ad hoc query requested by a Deep Agent.", help="Reason for the query.")
    parser.add_argument("--output", help="Optional path to save the JSON result.")
    parser.add_argument("--max-rows", type=int, default=20, help="Maximum number of rows allowed in the result.")
    return parser


def load_sql(args: argparse.Namespace) -> str:
    """Load SQL text from either `--sql` or `--sql-file`."""

    if bool(args.sql) == bool(args.sql_file):
        raise ValueError("Provide exactly one of --sql or --sql-file.")
    if args.sql:
        return str(args.sql).strip()
    return Path(args.sql_file).read_text(encoding="utf-8").strip()


def main() -> int:
    """Run a single validated SQL query and emit JSON for the caller."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        sql = load_sql(args)
        settings = Settings.from_env()
        repository = SQLRepository(settings)
        result = repository.execute_query_plan(
            SQLQueryPlan(
                queries=[
                    SQLQueryCandidate(
                        name=args.name,
                        sql=sql,
                        reason=args.reason,
                    )
                ]
            ),
            max_rows=args.max_rows,
        )
    except Exception as exc:
        error_payload = {"available": False, "mode": "runner_error", "error": str(exc)}
        print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        return 1

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    print(output)

    queries = result.get("executed_queries") or []
    succeeded = result.get("available") and not result.get("error") and queries and not any(
        query.get("error") for query in queries
    )
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
