from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Settings(BaseModel):
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    agent_max_attempts: int = 3
    finalizer_openai_api_key: str | None = None
    finalizer_openai_base_url: str | None = None
    finalizer_openai_model: str | None = None
    finalizer_temperature: float = 0.0
    finalizer_max_attempts: int = 3

    db_host: str = "127.0.0.1"
    db_port: int = 5432
    db_user: str = ""
    db_password: str = ""
    db_name: str = "priorreview"

    results_dir: str = "data/results"
    workflow_version: str = "v0.1"
    max_reviewer_author_rounds: int = 2
    max_sc_rereview_rounds: int = 1
    reviewer_data_agent_enabled: bool = True
    author_data_agent_enabled: bool = True
    sc_data_agent_enabled: bool = True
    data_agent_code_execution_enabled: bool = True
    data_agent_command_timeout_seconds: int = Field(default=120, ge=1, le=3600)
    data_agent_max_steps: int = Field(default=100, ge=10, le=1000)

    console_log_enabled: bool = True

    external_research_cutoff_date: str | None = None
    semantic_scholar_proxy_base_url: str = "https://s2api.ominiai.cn/s2"
    semantic_scholar_proxy_api_key: str | None = None
    arxiv_api_base_url: str = "https://export.arxiv.org/api/query"
    arxiv_default_max_results: int = 5
    arxiv_user_agent: str = "PriorReview/0.1"
    github_api_key: str | None = None
    github_api_base_url: str = "https://api.github.com"
    github_graphql_url: str = "https://api.github.com/graphql"
    github_api_version: str = "2022-11-28"
    github_user_agent: str = "PriorReview-GitHub-Research/0.1"

    @classmethod
    def from_env(cls, env_path: str | Path = ".env", **overrides: Any) -> "Settings":
        raw = cls._load_env_file(env_path)
        merged = {**raw, **os.environ}

        data = {
            "openai_api_key": cls._pick(merged, "OPENAI_API_KEY"),
            "openai_base_url": cls._pick(merged, "OPENAI_BASE_URL"),
            "openai_model": cls._pick(merged, "OPENAI_MODEL", "model", "MODEL") or "gpt-4.1-mini",
            "temperature": cls._as_float(cls._pick(merged, "PRIORREVIEW_TEMPERATURE"), 0.2),
            "agent_max_attempts": cls._as_int(cls._pick(merged, "PRIORREVIEW_AGENT_MAX_ATTEMPTS"), 3),
            "finalizer_openai_api_key": cls._pick(merged, "FINALIZER_OPENAI_API_KEY"),
            "finalizer_openai_base_url": cls._pick(merged, "FINALIZER_OPENAI_BASE_URL"),
            "finalizer_openai_model": cls._pick(merged, "FINALIZER_OPENAI_MODEL"),
            "finalizer_temperature": cls._as_float(cls._pick(merged, "FINALIZER_OPENAI_TEMPERATURE"), 0.0),
            "finalizer_max_attempts": cls._as_int(cls._pick(merged, "FINALIZER_MAX_ATTEMPTS"), 3),
            "db_host": cls._pick(merged, "PRIORREVIEW_DB_HOST", "PGHOST", "DB_HOST") or "127.0.0.1",
            "db_port": cls._as_int(cls._pick(merged, "PRIORREVIEW_DB_PORT", "PGPORT", "DB_PORT"), 5432),
            "db_user": cls._pick(merged, "PRIORREVIEW_DB_USER", "PGUSER", "DB_USER") or "",
            "db_password": cls._pick(merged, "PRIORREVIEW_DB_PASSWORD", "PGPASSWORD", "DB_PASSWORD") or "",
            "db_name": cls._pick(merged, "PRIORREVIEW_DB_NAME", "PGDATABASE", "DB_NAME") or "priorreview",
            "results_dir": cls._pick(merged, "PRIORREVIEW_RESULTS_DIR") or "data/results",
            "workflow_version": cls._pick(merged, "PRIORREVIEW_WORKFLOW_VERSION") or "v0.1",
            "max_reviewer_author_rounds": cls._as_int(
                cls._pick(merged, "PRIORREVIEW_MAX_REVIEWER_AUTHOR_ROUNDS"), 2
            ),
            "max_sc_rereview_rounds": cls._as_int(
                cls._pick(merged, "PRIORREVIEW_MAX_SC_REREVIEW_ROUNDS", "PRIORREVIEW_MAX_AC_REREVIEW_ROUNDS"), 1
            ),
            "reviewer_data_agent_enabled": cls._as_bool(
                cls._pick(merged, "PRIORREVIEW_REVIEWER_DATA_AGENT_ENABLED"), True
            ),
            "author_data_agent_enabled": cls._as_bool(
                cls._pick(merged, "PRIORREVIEW_AUTHOR_DATA_AGENT_ENABLED"), True
            ),
            "sc_data_agent_enabled": cls._as_bool(
                cls._pick(merged, "PRIORREVIEW_SC_DATA_AGENT_ENABLED"), True
            ),
            "data_agent_code_execution_enabled": cls._as_bool(
                cls._pick(merged, "PRIORREVIEW_DATA_AGENT_CODE_EXECUTION_ENABLED"), True
            ),
            "data_agent_command_timeout_seconds": cls._as_int(
                cls._pick(merged, "PRIORREVIEW_DATA_AGENT_COMMAND_TIMEOUT_SECONDS"), 120
            ),
            "data_agent_max_steps": cls._as_int(cls._pick(merged, "PRIORREVIEW_DATA_AGENT_MAX_STEPS"), 100),
            "console_log_enabled": cls._as_bool(cls._pick(merged, "PRIORREVIEW_CONSOLE_LOG"), True),
            "external_research_cutoff_date": cls._pick(merged, "PRIORREVIEW_EXTERNAL_RESEARCH_CUTOFF_DATE"),
            "semantic_scholar_proxy_base_url": cls._pick(merged, "SEMANTIC_SCHOLAR_PROXY_BASE_URL")
            or "https://s2api.ominiai.cn/s2",
            "semantic_scholar_proxy_api_key": cls._pick(merged, "SEMANTIC_SCHOLAR_PROXY_API_KEY"),
            "arxiv_api_base_url": cls._pick(merged, "ARXIV_API_BASE_URL") or "https://export.arxiv.org/api/query",
            "arxiv_default_max_results": cls._as_int(cls._pick(merged, "ARXIV_DEFAULT_MAX_RESULTS"), 5),
            "arxiv_user_agent": cls._pick(merged, "ARXIV_USER_AGENT") or "PriorReview/0.1",
            "github_api_key": cls._pick(merged, "GITHUB_API_KEY"),
            "github_api_base_url": cls._pick(merged, "GITHUB_API_BASE_URL") or "https://api.github.com",
            "github_graphql_url": cls._pick(merged, "GITHUB_GRAPHQL_URL") or "https://api.github.com/graphql",
            "github_api_version": cls._pick(merged, "GITHUB_API_VERSION") or "2022-11-28",
            "github_user_agent": cls._pick(merged, "GITHUB_USER_AGENT") or "PriorReview-GitHub-Research/0.1",
        }
        data.update(overrides)
        return cls(**data)

    @staticmethod
    def _load_env_file(env_path: str | Path) -> dict[str, str]:
        path = Path(env_path)
        if not path.exists():
            return {}

        values: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            values[key] = value
        return values

    @staticmethod
    def _pick(values: dict[str, str], *keys: str) -> str | None:
        for key in keys:
            if key in values and str(values[key]).strip():
                return str(values[key]).strip()
        return None

    @staticmethod
    def _as_bool(value: str | None, default: bool) -> bool:
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _as_int(value: str | None, default: int) -> int:
        try:
            return int(value) if value is not None else default
        except ValueError:
            return default

    @staticmethod
    def _as_float(value: str | None, default: float) -> float:
        try:
            return float(value) if value is not None else default
        except ValueError:
            return default

    @property
    def results_root(self) -> Path:
        return Path(self.results_dir)

    def to_env_dict(self) -> dict[str, str]:
        env: dict[str, str] = {
            "OPENAI_MODEL": self.openai_model,
            "PRIORREVIEW_DB_HOST": self.db_host,
            "PRIORREVIEW_DB_PORT": str(self.db_port),
            "PRIORREVIEW_DB_USER": self.db_user,
            "PRIORREVIEW_DB_PASSWORD": self.db_password,
            "PRIORREVIEW_DB_NAME": self.db_name,
            "PRIORREVIEW_RESULTS_DIR": self.results_dir,
            "PRIORREVIEW_AGENT_MAX_ATTEMPTS": str(self.agent_max_attempts),
            "PRIORREVIEW_MAX_REVIEWER_AUTHOR_ROUNDS": str(self.max_reviewer_author_rounds),
            "PRIORREVIEW_MAX_SC_REREVIEW_ROUNDS": str(self.max_sc_rereview_rounds),
            "PRIORREVIEW_REVIEWER_DATA_AGENT_ENABLED": str(self.reviewer_data_agent_enabled).lower(),
            "PRIORREVIEW_AUTHOR_DATA_AGENT_ENABLED": str(self.author_data_agent_enabled).lower(),
            "PRIORREVIEW_SC_DATA_AGENT_ENABLED": str(self.sc_data_agent_enabled).lower(),
            "PRIORREVIEW_DATA_AGENT_CODE_EXECUTION_ENABLED": str(self.data_agent_code_execution_enabled).lower(),
            "PRIORREVIEW_DATA_AGENT_COMMAND_TIMEOUT_SECONDS": str(self.data_agent_command_timeout_seconds),
            "PRIORREVIEW_DATA_AGENT_MAX_STEPS": str(self.data_agent_max_steps),
            "SEMANTIC_SCHOLAR_PROXY_BASE_URL": self.semantic_scholar_proxy_base_url,
            "ARXIV_API_BASE_URL": self.arxiv_api_base_url,
            "ARXIV_DEFAULT_MAX_RESULTS": str(self.arxiv_default_max_results),
            "ARXIV_USER_AGENT": self.arxiv_user_agent,
            "GITHUB_API_BASE_URL": self.github_api_base_url,
            "GITHUB_GRAPHQL_URL": self.github_graphql_url,
            "GITHUB_API_VERSION": self.github_api_version,
            "GITHUB_USER_AGENT": self.github_user_agent,
        }
        if self.external_research_cutoff_date:
            env["PRIORREVIEW_EXTERNAL_RESEARCH_CUTOFF_DATE"] = self.external_research_cutoff_date
        if self.openai_api_key:
            env["OPENAI_API_KEY"] = self.openai_api_key
        if self.openai_base_url:
            env["OPENAI_BASE_URL"] = self.openai_base_url
        if self.finalizer_openai_api_key:
            env["FINALIZER_OPENAI_API_KEY"] = self.finalizer_openai_api_key
        if self.finalizer_openai_base_url:
            env["FINALIZER_OPENAI_BASE_URL"] = self.finalizer_openai_base_url
        if self.finalizer_openai_model:
            env["FINALIZER_OPENAI_MODEL"] = self.finalizer_openai_model
        if self.semantic_scholar_proxy_api_key:
            env["SEMANTIC_SCHOLAR_PROXY_API_KEY"] = self.semantic_scholar_proxy_api_key
        if self.github_api_key:
            env["GITHUB_API_KEY"] = self.github_api_key
        return env
