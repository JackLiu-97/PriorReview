from __future__ import annotations

from datetime import date
from decimal import Decimal
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from agent.core.settings import Settings
from agent.core.sql_context import SQLRepository, SQLValidationError
from agent.cli.sql_runner import main as sql_runner_main


class SQLContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = SQLRepository(Settings.from_env())

    def test_validate_adds_limit_to_select(self) -> None:
        sql = "SELECT p.year, p.title FROM papers p ORDER BY p.year DESC"
        normalized = self.repository.validate_read_only_sql(sql, max_rows=7)
        self.assertIn("LIMIT 7", normalized.upper())

    def test_validate_rejects_delete(self) -> None:
        with self.assertRaises(SQLValidationError):
            self.repository.validate_read_only_sql("DELETE FROM papers WHERE id = 1")

    def test_validate_rejects_disallowed_table(self) -> None:
        with self.assertRaises(SQLValidationError):
            self.repository.validate_read_only_sql("SELECT * FROM users LIMIT 5")

    def test_connection_requests_read_only_database_transactions(self) -> None:
        with patch("agent.core.sql_context.psycopg.connect") as connect:
            self.repository._connect()
        self.assertIn("default_transaction_read_only=on", connect.call_args.kwargs["options"])
        self.assertEqual(connect.call_args.kwargs["connect_timeout"], 10)

    def test_sql_cli_exit_status_distinguishes_unavailable_failed_and_empty_success(self) -> None:
        cases = [
            ({"available": False, "error": "connection unavailable", "executed_queries": []}, 1),
            ({"available": True, "error": None, "executed_queries": [{"error": "invalid query"}]}, 1),
            ({"available": True, "error": None, "executed_queries": [{"error": None, "row_count": 0}]}, 0),
        ]
        for payload, expected in cases:
            with self.subTest(expected=expected, payload=payload), patch(
                "sys.argv", ["sql_runner", "--sql", "SELECT 1"]
            ), patch("agent.cli.sql_runner.SQLRepository") as repository, redirect_stdout(StringIO()):
                repository.return_value.execute_query_plan.return_value = payload
                self.assertEqual(sql_runner_main(), expected)

    def test_normalize_json_value_handles_decimal_and_date(self) -> None:
        normalized = self.repository._normalize_json_value(
            {
                "avg_rating": Decimal("4.25"),
                "review_count": Decimal("3"),
                "created_at": date(2026, 3, 28),
            }
        )
        self.assertEqual(normalized["avg_rating"], 4.25)
        self.assertEqual(normalized["review_count"], 3)
        self.assertEqual(normalized["created_at"], "2026-03-28")


if __name__ == "__main__":
    unittest.main()
