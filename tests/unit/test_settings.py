from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.core.settings import Settings


class SettingsTest(unittest.TestCase):
    def test_uses_postgres_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env(env_path="/tmp/priorreview-missing-env")
        self.assertEqual(settings.db_host, "127.0.0.1")
        self.assertEqual(settings.db_port, 5432)
        self.assertEqual(settings.db_name, "priorreview")
        self.assertEqual(settings.agent_max_attempts, 3)
        self.assertEqual(settings.finalizer_max_attempts, 3)
        self.assertTrue(settings.reviewer_data_agent_enabled)
        self.assertTrue(settings.author_data_agent_enabled)
        self.assertTrue(settings.sc_data_agent_enabled)
        env = settings.to_env_dict()
        self.assertEqual(env["PRIORREVIEW_REVIEWER_DATA_AGENT_ENABLED"], "true")
        self.assertEqual(env["PRIORREVIEW_AUTHOR_DATA_AGENT_ENABLED"], "true")
        self.assertEqual(env["PRIORREVIEW_SC_DATA_AGENT_ENABLED"], "true")

    def test_loads_openai_model_with_legacy_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=test-key\nOPENAI_BASE_URL=https://example.com\nmodel =test-model\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(env_path)
            self.assertEqual(settings.openai_model, "test-model")
            self.assertEqual(settings.openai_base_url, "https://example.com")

    def test_loads_finalizer_openai_settings_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=review-key",
                        "OPENAI_BASE_URL=https://review.example.com/v1",
                        "OPENAI_MODEL=review-model",
                        "PRIORREVIEW_AGENT_MAX_ATTEMPTS=2",
                        "FINALIZER_OPENAI_API_KEY=finalizer-key",
                        "FINALIZER_OPENAI_BASE_URL=https://finalizer.example.com/v1",
                        "FINALIZER_OPENAI_MODEL=finalizer-model",
                        "FINALIZER_OPENAI_TEMPERATURE=0.1",
                        "FINALIZER_MAX_ATTEMPTS=2",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(env_path)
            self.assertEqual(settings.openai_model, "review-model")
            self.assertEqual(settings.agent_max_attempts, 2)
            self.assertEqual(settings.finalizer_openai_api_key, "finalizer-key")
            self.assertEqual(settings.finalizer_openai_base_url, "https://finalizer.example.com/v1")
            self.assertEqual(settings.finalizer_openai_model, "finalizer-model")
            self.assertEqual(settings.finalizer_temperature, 0.1)
            self.assertEqual(settings.finalizer_max_attempts, 2)

    def test_loads_score_calibration_round_setting_with_ac_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("PRIORREVIEW_MAX_AC_REREVIEW_ROUNDS=3\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(env_path)
            self.assertEqual(settings.max_sc_rereview_rounds, 3)

            env_path.write_text(
                "PRIORREVIEW_MAX_AC_REREVIEW_ROUNDS=3\nPRIORREVIEW_MAX_SC_REREVIEW_ROUNDS=2\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(env_path)
            self.assertEqual(settings.max_sc_rereview_rounds, 2)
            env = settings.to_env_dict()
            self.assertEqual(env["PRIORREVIEW_MAX_REVIEWER_AUTHOR_ROUNDS"], "2")
            self.assertEqual(env["PRIORREVIEW_MAX_SC_REREVIEW_ROUNDS"], "2")

    def test_loads_role_data_agent_switches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "PRIORREVIEW_REVIEWER_DATA_AGENT_ENABLED=false",
                        "PRIORREVIEW_AUTHOR_DATA_AGENT_ENABLED=0",
                        "PRIORREVIEW_SC_DATA_AGENT_ENABLED=off",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(env_path)

            self.assertFalse(settings.reviewer_data_agent_enabled)
            self.assertFalse(settings.author_data_agent_enabled)
            self.assertFalse(settings.sc_data_agent_enabled)
            env = settings.to_env_dict()
            self.assertEqual(env["PRIORREVIEW_REVIEWER_DATA_AGENT_ENABLED"], "false")
            self.assertEqual(env["PRIORREVIEW_AUTHOR_DATA_AGENT_ENABLED"], "false")
            self.assertEqual(env["PRIORREVIEW_SC_DATA_AGENT_ENABLED"], "false")

    def test_loads_autonomous_data_agent_settings(self) -> None:
        with patch.dict(os.environ, {
            "PRIORREVIEW_DATA_AGENT_CODE_EXECUTION_ENABLED": "false",
            "PRIORREVIEW_DATA_AGENT_COMMAND_TIMEOUT_SECONDS": "240",
            "PRIORREVIEW_DATA_AGENT_MAX_STEPS": "150",
        }, clear=True):
            settings = Settings.from_env(env_path="/tmp/priorreview-missing-env")
        self.assertFalse(settings.data_agent_code_execution_enabled)
        self.assertEqual(settings.data_agent_command_timeout_seconds, 240)
        self.assertEqual(settings.data_agent_max_steps, 150)
        env = settings.to_env_dict()
        self.assertEqual(env["PRIORREVIEW_DATA_AGENT_CODE_EXECUTION_ENABLED"], "false")
        self.assertEqual(env["PRIORREVIEW_DATA_AGENT_COMMAND_TIMEOUT_SECONDS"], "240")
        self.assertEqual(env["PRIORREVIEW_DATA_AGENT_MAX_STEPS"], "150")
        self.assertTrue(Settings().data_agent_code_execution_enabled)
        with self.assertRaises(ValueError):
            Settings(data_agent_command_timeout_seconds=0)

    def test_loads_postgres_pg_env_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "PGHOST=localhost\nPGPORT=15432\nPGUSER=postgres\nPGPASSWORD=test-pass\nPGDATABASE=test_db\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(env_path)
            self.assertEqual(settings.db_host, "localhost")
            self.assertEqual(settings.db_port, 15432)
            self.assertEqual(settings.db_user, "postgres")
            self.assertEqual(settings.db_password, "test-pass")
            self.assertEqual(settings.db_name, "test_db")

    def test_loads_external_research_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "PRIORREVIEW_EXTERNAL_RESEARCH_CUTOFF_DATE=2024-05-01",
                        "SEMANTIC_SCHOLAR_PROXY_BASE_URL=https://s2.example.test/s2",
                        "SEMANTIC_SCHOLAR_PROXY_API_KEY=s2-key",
                        "ARXIV_API_BASE_URL=https://arxiv.example.test/api/query",
                        "ARXIV_DEFAULT_MAX_RESULTS=11",
                        "ARXIV_USER_AGENT=PriorReviewTest/1.0",
                        "GITHUB_API_KEY=gh-key",
                        "GITHUB_API_BASE_URL=https://github.example.test",
                        "GITHUB_GRAPHQL_URL=https://github.example.test/graphql",
                        "GITHUB_API_VERSION=2022-11-28",
                        "GITHUB_USER_AGENT=PriorReviewGitHubTest/1.0",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(env_path)
            self.assertEqual(settings.external_research_cutoff_date, "2024-05-01")
            self.assertEqual(settings.semantic_scholar_proxy_base_url, "https://s2.example.test/s2")
            self.assertEqual(settings.semantic_scholar_proxy_api_key, "s2-key")
            self.assertEqual(settings.arxiv_api_base_url, "https://arxiv.example.test/api/query")
            self.assertEqual(settings.arxiv_default_max_results, 11)
            self.assertEqual(settings.arxiv_user_agent, "PriorReviewTest/1.0")
            self.assertEqual(settings.github_api_key, "gh-key")
            self.assertEqual(settings.github_api_base_url, "https://github.example.test")
            self.assertEqual(settings.github_graphql_url, "https://github.example.test/graphql")
            self.assertEqual(settings.github_api_version, "2022-11-28")
            self.assertEqual(settings.github_user_agent, "PriorReviewGitHubTest/1.0")

            env = settings.to_env_dict()
            self.assertEqual(env["PRIORREVIEW_EXTERNAL_RESEARCH_CUTOFF_DATE"], "2024-05-01")
            self.assertEqual(env["SEMANTIC_SCHOLAR_PROXY_API_KEY"], "s2-key")
            self.assertEqual(env["GITHUB_API_KEY"], "gh-key")
            self.assertEqual(env["ARXIV_DEFAULT_MAX_RESULTS"], "11")


if __name__ == "__main__":
    unittest.main()
