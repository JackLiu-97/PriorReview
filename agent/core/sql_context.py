from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg
import sqlglot
from psycopg.rows import dict_row
from sqlglot import exp

from agent.core.settings import Settings
from agent.schemas.common import SQLQueryPlan


ALLOWED_TABLES = {"papers", "reviews", "meta_reviews", "comments"}
FORBIDDEN_EXPRESSIONS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Command,
    exp.Transaction,
)

SQL_SCHEMA_CONTEXT = """
Database: priorreview

Allowed tables
- papers
- reviews
- meta_reviews
- comments

Table papers — one row per submitted paper/forum.
- id: Internal primary key.
- conference: Conference name, for example ICLR or NeurIPS.
- year: Conference year.
- forum_id: OpenReview forum id for the paper.
- paper_number: Submission number parsed from the source filename when available.
- title: Paper title.
- decision: Final decision text when available, for example Accept or Reject.
- decision_source: Source note/type used to extract the decision.
- filename: Original crawled JSON filename.

Table reviews — official reviewer reports linked to papers.
- id: Internal primary key.
- paper_id: Foreign key to papers.id.
- note_id: OpenReview note id for the official review.
- reviewer_id: Reviewer identity string, often anonymized.
- summary: Reviewer-provided paper summary when the year/schema exposes one.
- review_text: Main review body or unified review text.
- strengths: Strengths section when the year/schema exposes one.
- weaknesses: Weaknesses section when the year/schema exposes one.
- questions: Reviewer questions when the year/schema exposes them.
- rating_raw: Original rating/recommendation string preserved from OpenReview.
- rating_numeric: Numeric rating extracted from rating_raw; may be NULL depending on year/conference schema.
- rating_max: Maximum rating value for the original scale when available.
- rating_normalized: Rating normalized to [0, 1] for cross-year comparisons.
- confidence_raw: Original confidence string preserved from OpenReview.
- confidence_numeric: Numeric confidence extracted from confidence_raw; may be NULL depending on year/conference schema.
- confidence_max: Maximum confidence value for the original scale, usually 5 when available.
- soundness: Numeric soundness/correctness score; may be NULL depending on year/conference schema.
- presentation: Numeric presentation/clarity score; may be NULL depending on year/conference schema.
- contribution: Numeric contribution score; may be NULL depending on year/conference schema.
- novelty: Technical novelty/significance field, mainly populated for schemas that expose it separately.
- empirical_novelty: Empirical novelty/significance field, mainly populated for schemas that expose it separately.
- ethics_flag: Ethics-review flag or related ethics marker when available.
- ethics_concerns: Free-text ethics concerns when available.
- extra_fields: JSON object containing year-specific review fields not mapped to standard columns.
- created_at: OpenReview creation timestamp value, not a Python datetime string.
- modified_at: OpenReview modification timestamp value, not a Python datetime string.

Table meta_reviews — area-chair/meta-review and decision-summary notes linked to papers.
- id: Internal primary key.
- paper_id: Foreign key to papers.id.
- note_id: OpenReview note id for the meta-review or decision-summary note.
- ac_id: Area chair or meta-reviewer identity string when available.
- meta_text: Main meta-review, decision summary, or AC summary text.
- additional_comments: Additional AC comments on the reviewer discussion when available.
- reviewer_concerns: AC summary of reviewer concerns when available.
- reviewer_scores: AC summary or prediction of reviewer scores when available.
- justification_higher: AC explanation for why the score/recommendation was not higher when available.
- justification_lower: AC explanation for why the score/recommendation was not lower when available.
- recommendation: AC recommendation or decision text when available.
- ac_confidence_raw: Original AC confidence string preserved from OpenReview.
- ac_confidence_num: Numeric AC confidence extracted from ac_confidence_raw; may be NULL depending on year/conference schema.
- created_at: OpenReview creation timestamp value, not a Python datetime string.
- modified_at: OpenReview modification timestamp value, not a Python datetime string.

Table comments — discussion/rebuttal/comment notes linked to papers.
- id: Internal primary key.
- paper_id: Foreign key to papers.id.
- note_id: OpenReview note id for the discussion comment.
- reply_to: OpenReview note id that this comment replies to, when available.
- author_id: Comment author identity string.
- author_role: Author role such as Reviewer, Author, AC, or Public when available.
- title: Comment title or subject line.
- comment_text: Main discussion, rebuttal, or comment body.
- created_at: OpenReview creation timestamp value, not a Python datetime string.
- modified_at: OpenReview modification timestamp value, not a Python datetime string.

Key joins
- reviews.paper_id = papers.id
- meta_reviews.paper_id = papers.id
- comments.paper_id = papers.id
""".strip()

SQL_PLANNER_RULES = """
You are generating SQL for a read-only PostgreSQL database.

Rules:
- Return only read-only SQL queries.
- Allowed statement type: SELECT only. WITH + SELECT is allowed.
- Do not generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE, CALL, or multiple statements.
- Use only these tables: papers, reviews, meta_reviews, comments.
- Prefer simple queries with explicit joins.
- Always include a LIMIT unless an aggregate query naturally returns a very small result set.
- Keep each query focused and useful for the requesting role.
- Each query-plan call accepts at most 3 queries. Run additional focused calls when needed.
""".strip()


class SQLValidationError(ValueError):
    pass


class SQLRepository:
    """Read-only SQL executor used by Deep Agent data specialists."""

    def __init__(self, settings: Settings, *, allowed_tables: set[str] | frozenset[str] | None = None) -> None:
        self.settings = settings
        self.allowed_tables = frozenset(allowed_tables or ALLOWED_TABLES)

    @property
    def schema_context(self) -> str:
        return SQL_SCHEMA_CONTEXT

    @property
    def planner_rules(self) -> str:
        return SQL_PLANNER_RULES

    def planner_prompt(self) -> str:
        return f"{SQL_PLANNER_RULES}\n\n{SQL_SCHEMA_CONTEXT}"

    def execute_query_plan(self, plan: SQLQueryPlan, *, max_rows: int = 20) -> dict[str, Any]:
        """Validate and execute up to three read-only SQL queries."""

        try:
            conn = self._connect()
        except Exception as exc:  # pragma: no cover - external DB dependency
            return {
                "available": False,
                "mode": "unavailable",
                "schema_context": self.schema_context,
                "planner_rules": self.planner_rules,
                "planned_queries": [query.model_dump(mode="json") for query in plan.queries],
                "executed_queries": [],
                "results": {},
                "error": str(exc),
            }

        executed_queries: list[dict[str, Any]] = []
        results: dict[str, list[dict[str, Any]]] = {}
        try:
            with conn.cursor() as cur:
                for query in plan.queries[:3]:
                    try:
                        normalized_sql = self.validate_read_only_sql(query.sql, max_rows=max_rows)
                        cur.execute(normalized_sql)
                        rows = list(cur.fetchall())
                        executed_queries.append(
                            {
                                "name": query.name,
                                "reason": query.reason,
                                "sql": normalized_sql,
                                "row_count": len(rows),
                                "error": None,
                            }
                        )
                        results[query.name] = self._normalize_json_value(rows)
                    except Exception as exc:
                        executed_queries.append(
                            {
                                "name": query.name,
                                "reason": query.reason,
                                "sql": query.sql,
                                "row_count": 0,
                                "error": str(exc),
                            }
                        )
                        results[query.name] = []
            return {
                "available": True,
                "mode": "live",
                "schema_context": self.schema_context,
                "planner_rules": self.planner_rules,
                "planned_queries": [query.model_dump(mode="json") for query in plan.queries],
                "executed_queries": executed_queries,
                "results": results,
                "error": None,
            }
        finally:
            conn.close()

    def validate_read_only_sql(self, sql: str, *, max_rows: int = 20) -> str:
        """Reject mutating SQL and normalize the remaining query with a safe LIMIT."""

        statements = sqlglot.parse(sql, read="postgres")
        if len(statements) != 1:
            raise SQLValidationError("Only a single SQL statement is allowed.")

        parsed = statements[0]
        if not isinstance(parsed, exp.Select):
            raise SQLValidationError("Only SELECT statements are allowed.")

        for forbidden in FORBIDDEN_EXPRESSIONS:
            if parsed.find(forbidden):
                raise SQLValidationError(f"Forbidden SQL expression detected: {forbidden.__name__}.")

        cte_aliases = {cte.alias for cte in parsed.ctes}
        referenced_tables = {table.name for table in parsed.find_all(exp.Table)}
        underlying_tables = {name for name in referenced_tables if name not in cte_aliases}
        disallowed_tables = sorted(underlying_tables - self.allowed_tables)
        if disallowed_tables:
            raise SQLValidationError(
                f"Disallowed tables referenced: {', '.join(disallowed_tables)}. "
                f"Allowed tables: {', '.join(sorted(self.allowed_tables))}."
            )

        limit_expression = parsed.args.get("limit")
        if limit_expression:
            limit_value = self._extract_limit_value(limit_expression)
            if limit_value is None:
                raise SQLValidationError("LIMIT must be a literal integer.")
            if limit_value > max_rows:
                parsed.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        else:
            parsed.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))

        return parsed.sql(dialect="postgres")

    @staticmethod
    def _extract_limit_value(limit_expression: exp.Expression) -> int | None:
        expression = limit_expression.expression
        if isinstance(expression, exp.Literal) and expression.is_int:
            return int(expression.this)
        return None

    @classmethod
    def _normalize_json_value(cls, value: Any) -> Any:
        """Convert DB-native values into JSON-safe Python primitives."""

        if isinstance(value, dict):
            return {key: cls._normalize_json_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._normalize_json_value(item) for item in value]
        if isinstance(value, tuple):
            return [cls._normalize_json_value(item) for item in value]
        if isinstance(value, Decimal):
            return int(value) if value == value.to_integral_value() else float(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return value

    def _connect(self):
        """Try the configured PostgreSQL host first, then common local host aliases."""

        connection_attempts: list[dict[str, Any]] = [
            {"host": self.settings.db_host, "port": self.settings.db_port},
        ]
        if self.settings.db_host == "localhost":
            connection_attempts.append({"host": "127.0.0.1", "port": self.settings.db_port})
        if self.settings.db_host == "127.0.0.1":
            connection_attempts.append({"host": "localhost", "port": self.settings.db_port})

        last_error: Exception | None = None
        for attempt in connection_attempts:
            try:
                kwargs: dict[str, Any] = {
                    "dbname": self.settings.db_name,
                    "row_factory": dict_row,
                    "options": "-c default_transaction_read_only=on",
                    "connect_timeout": 10,
                    **attempt,
                }
                if self.settings.db_user:
                    kwargs["user"] = self.settings.db_user
                if self.settings.db_password:
                    kwargs["password"] = self.settings.db_password
                return psycopg.connect(**kwargs)
            except Exception as exc:  # pragma: no cover - network and local env dependent
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("No database connection attempts were configured.")
