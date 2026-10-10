from __future__ import annotations

import base64
import json
import tempfile
import unittest
import urllib.parse
from datetime import date
from pathlib import Path

from agent.tools.external_research import (
    ControlledExternalResearchTools,
    ExternalResearchClient,
)
from agent.core.settings import Settings
from agent.logging.execution_log import ExecutionLog


def _settings(**overrides) -> Settings:  # noqa: ANN003
    defaults = {
        "external_research_cutoff_date": "2024-05-01",
        "semantic_scholar_proxy_base_url": "https://s2.example.test/s2",
        "semantic_scholar_proxy_api_key": "s2-token",
        "arxiv_api_base_url": "https://arxiv.example.test/api/query",
        "github_api_key": "gh-token",
        "github_api_base_url": "https://api.github.example.test",
    }
    defaults.update(overrides)
    return Settings.from_env(env_path="/tmp/priorreview-missing-env", **defaults)


def _external_config(workspace: Path) -> dict:
    return {
        "configurable": {
            "assigned_external_workspace": "artifacts/external/reviewer_round_01",
            "assigned_external_workspace_abs": str(workspace),
            "timeline_role": "reviewer",
            "timeline_round": 1,
        }
    }


class ExternalResearchTest(unittest.TestCase):
    def test_semantic_scholar_filters_by_cutoff_and_year_only_rule(self) -> None:
        def fake_json(url: str, headers: dict[str, str]):  # noqa: ANN001
            self.assertIn("/graph/v1/paper/search", url)
            self.assertEqual(headers["Authorization"], "Bearer s2-token")
            return {
                "data": [
                    {
                        "paperId": "old-date",
                        "title": "Old Full Date",
                        "abstract": "old abstract",
                        "year": 2024,
                        "publicationDate": "2024-04-30",
                    },
                    {
                        "paperId": "after-date",
                        "title": "After Cutoff",
                        "year": 2024,
                        "publicationDate": "2024-05-01",
                    },
                    {"paperId": "old-year", "title": "Old Year Only", "year": 2023},
                    {"paperId": "same-year", "title": "Same Year Only", "year": 2024},
                    {"paperId": "missing-date", "title": "Missing Date"},
                ]
            }

        client = ExternalResearchClient(_settings(), json_getter=fake_json)
        result = client.search_semantic_scholar(
            query="rope long context",
            cutoff=date(2024, 5, 1),
            top_k=10,
            max_snippet_chars=200,
        )

        self.assertEqual([item["title"] for item in result["items"]], ["Old Full Date", "Old Year Only"])
        self.assertEqual(result["excluded_count"], 3)
        self.assertIsNone(result["error"])

    def test_arxiv_filters_by_published_before_cutoff(self) -> None:
        feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Old arXiv Paper</title>
    <summary>Older result.</summary>
    <published>2023-01-01T00:00:00Z</published>
    <updated>2023-01-02T00:00:00Z</updated>
    <author><name>Alice</name></author>
    <arxiv:primary_category term="cs.LG" />
    <category term="cs.LG" />
    <link href="http://arxiv.org/abs/2301.00001v1" rel="alternate" type="text/html" />
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2405.00001v1</id>
    <title>Cutoff Day Paper</title>
    <summary>Not allowed.</summary>
    <published>2024-05-01T00:00:00Z</published>
    <updated>2024-05-02T00:00:00Z</updated>
  </entry>
</feed>"""

        def fake_text(url: str, headers: dict[str, str]) -> str:
            self.assertIn("cat%3Acs.LG", url)
            self.assertEqual(headers["User-Agent"], "PriorReview/0.1")
            return feed

        client = ExternalResearchClient(_settings(), text_getter=fake_text)
        result = client.search_arxiv(
            query="long context",
            cutoff=date(2024, 5, 1),
            max_results=5,
            category="cs.LG",
            max_snippet_chars=200,
        )

        self.assertEqual([item["title"] for item in result["items"]], ["Old arXiv Paper"])
        self.assertEqual(result["items"][0]["published"], "2023-01-01")
        self.assertEqual(result["excluded_count"], 1)

    def test_github_uses_created_query_and_readme_snapshot_ref(self) -> None:
        calls: list[str] = []
        readme = base64.b64encode(b"Historical README").decode("ascii")

        def fake_json(url: str, headers: dict[str, str]):  # noqa: ANN001
            calls.append(url)
            decoded = urllib.parse.unquote_plus(url)
            self.assertEqual(headers["Authorization"], "Bearer gh-token")
            if "/search/repositories" in url:
                self.assertIn("created:<2024-05-01", decoded)
                return {
                    "items": [
                        {
                            "full_name": "owner/keep",
                            "html_url": "https://github.example/owner/keep",
                            "description": "kept",
                            "created_at": "2020-01-01T00:00:00Z",
                            "stargazers_count": 10,
                            "forks_count": 2,
                            "open_issues_count": 1,
                        },
                        {
                            "full_name": "owner/no-commit",
                            "created_at": "2020-01-01T00:00:00Z",
                        },
                        {
                            "full_name": "owner/new",
                            "created_at": "2024-05-01T00:00:00Z",
                        },
                    ]
                }
            if "/repos/owner/keep/commits" in url:
                self.assertIn("until=2024-05-01T00%3A00%3A00Z", url)
                return [
                    {
                        "sha": "abc123",
                        "html_url": "https://github.example/owner/keep/commit/abc123",
                        "commit": {"committer": {"date": "2024-04-30T23:59:00Z"}, "message": "before cutoff"},
                    }
                ]
            if "/repos/owner/no-commit/commits" in url:
                return []
            if "/repos/owner/keep/readme" in url:
                self.assertIn("ref=abc123", url)
                return {"name": "README.md", "path": "README.md", "content": readme}
            raise AssertionError(f"unexpected URL: {url}")

        client = ExternalResearchClient(_settings(), json_getter=fake_json)
        result = client.search_github(
            query="long-context-rope",
            cutoff=date(2024, 5, 1),
            top_k=5,
            include_readme=True,
            max_readme_chars=200,
        )

        self.assertEqual([item["full_name"] for item in result["items"]], ["owner/keep"])
        self.assertEqual(result["items"][0]["snapshot_before_cutoff"]["sha"], "abc123")
        self.assertEqual(result["items"][0]["readme_snapshot"]["text"], "Historical README")
        self.assertEqual(result["excluded_count"], 2)
        self.assertTrue(any("/search/repositories" in call for call in calls))

    def test_tools_write_missing_and_invalid_cutoff_artifacts_without_requests(self) -> None:
        class NoRequestClient:
            def __init__(self, settings):  # noqa: ANN001
                del settings

            def search_arxiv(self, **kwargs):  # noqa: ANN003
                raise AssertionError("network client should not be called")

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "artifacts" / "external" / "reviewer_round_01"
            tools = ControlledExternalResearchTools(
                settings=_settings(external_research_cutoff_date=None),
                client_factory=NoRequestClient,
            )
            response = json.loads(
                tools.search_arxiv(
                    state={"messages": []},
                    config=_external_config(workspace),
                    query="rope",
                    output_filename="missing.result.json",
                )
            )
            self.assertTrue(response["ok"])
            payload = json.loads((workspace / "missing.result.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "missing_cutoff_date")
            self.assertEqual(payload["items"], [])

            tools = ControlledExternalResearchTools(
                settings=_settings(external_research_cutoff_date="2024/05/01"),
                client_factory=NoRequestClient,
            )
            response = json.loads(
                tools.search_arxiv(
                    state={"messages": []},
                    config=_external_config(workspace),
                    query="rope",
                    output_filename="invalid.result.json",
                )
            )
            self.assertTrue(response["ok"])
            payload = json.loads((workspace / "invalid.result.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "invalid_cutoff_date")

    def test_tools_are_workspace_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "artifacts" / "external" / "reviewer_round_01"
            tools = ControlledExternalResearchTools(settings=_settings(external_research_cutoff_date=None))
            for filename in ("../escape.json", "subagent_traces/out.json", "out.txt"):
                response = json.loads(
                    tools.search_arxiv(
                        state={"messages": []},
                        config=_external_config(workspace),
                        query="rope",
                        output_filename=filename,
                    )
                )
                self.assertFalse(response["ok"])

    def test_tool_logs_and_artifacts_do_not_include_api_tokens(self) -> None:
        def fake_json(url: str, headers: dict[str, str]):  # noqa: ANN001
            self.assertEqual(headers["Authorization"], "Bearer s2-token")
            return {
                "data": [
                    {
                        "paperId": "old",
                        "title": "Old Paper",
                        "year": 2023,
                        "publicationDate": "2023-01-01",
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            workspace = run_root / "artifacts" / "external" / "reviewer_round_01"
            execution_log = ExecutionLog(
                run_id="run-test",
                paper_id="paper",
                paper_path="paper.md",
                run_dir=run_root,
                models={"main": "test"},
            )
            execution_log.start_run()
            tools = ControlledExternalResearchTools(
                settings=_settings(),
                run_root=run_root,
                execution_log=execution_log,
                client_factory=lambda settings: ExternalResearchClient(settings, json_getter=fake_json),
            )
            response = json.loads(
                tools.search_semantic_scholar(
                    state={"messages": []},
                    config=_external_config(workspace),
                    query="rope",
                    output_filename="semantic.result.json",
                )
            )
            self.assertTrue(response["ok"])

            combined = (
                execution_log.timeline_path.read_text(encoding="utf-8")
                + (workspace / "semantic.result.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("s2-token", combined)
            self.assertNotIn("gh-token", combined)


if __name__ == "__main__":
    unittest.main()
