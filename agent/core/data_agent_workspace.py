"""Workspace and local execution support for autonomous data specialists.

File tools use virtual paths; commands run as ordinary local processes. These
paths and read-only reference mounts are tool guardrails, not an OS sandbox.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents.backends.protocol import EditResult, ExecuteResponse, FileUploadResponse, WriteResult
from deepagents.middleware.filesystem import FilesystemMiddleware

from agent.core.settings import Settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_PREVIEW_BYTES = 16_000


def resolve_data_workspace(
    *, config: Any, run_root: Path, agent_name: str | None = None,
) -> Path:
    """Resolve an orchestrator-owned assignment; never infer authority from text."""

    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    if not isinstance(configurable, dict):
        configurable = {}
    value = configurable.get("assigned_data_workspace_abs") or configurable.get("assigned_data_workspace")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("The runtime must assign a data workspace through configurable.")
    root = run_root.resolve()
    workspace = Path(value).expanduser()
    if not workspace.is_absolute():
        workspace = root / workspace
    workspace = workspace.resolve()
    data_root = (root / "artifacts" / "data").resolve()
    if not data_root.is_relative_to(root) or workspace.parent != data_root:
        raise ValueError("Data workspace must be a role directory immediately under this run's artifacts/data.")
    if agent_name and not workspace.name.startswith(f"{agent_name}_"):
        raise ValueError(f"Data workspace is not assigned to role {agent_name}.")
    return workspace


class _ReadOnlyReferenceBackend(FilesystemBackend):
    """Reference code and documentation cannot be changed through file tools."""

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error="Project references are read-only. Write analysis files in the data workspace.")

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return EditResult(error="Project references are read-only. Copy the relevant code into the data workspace.")

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]


def data_agent_environment(settings: Settings, *, workspace: Path, agent_name: str) -> dict[str, str]:
    """Pass data-service configuration and the current Python, without model keys."""

    # Only inherit process essentials and the embedding service used by RAG.
    inherited = {
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
        "HOME", "USERPROFILE", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR",
        "OPENAI_EMBEDDING_API_KEY", "OPENAI_EMBEDDING_BASE_URL", "OPENAI_EMBEDDING_MODEL",
        "OPENAI_EMBEDDING_DIMENSIONS", "OPENROUTER_HTTP_REFERER", "OPENROUTER_TITLE",
        "OPENROUTER_APP_TITLE",
    }
    env = {key: value for key, value in os.environ.items() if key.upper() in inherited}
    env.update({key: value for key, value in settings.to_env_dict().items() if key.startswith("PRIORREVIEW_DB_")})
    python_dir = str(Path(sys.executable).parent)
    env["PATH"] = os.pathsep.join([python_dir, env.get("PATH", "")])
    env.update({
        "VIRTUAL_ENV": sys.prefix,
        "PYTHONPATH": str(PROJECT_ROOT),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHON_DOTENV_DISABLED": "1",
        "PRIORREVIEW_DATA_AGENT_ROLE": agent_name,
        "PRIORREVIEW_DATA_WORKSPACE": str(workspace),
        "PRIORREVIEW_PROJECT_ROOT": str(PROJECT_ROOT),
        "PGHOST": settings.db_host,
        "PGPORT": str(settings.db_port),
        "PGDATABASE": settings.db_name,
        "PGOPTIONS": "-c default_transaction_read_only=on",
        "PGCONNECT_TIMEOUT": "10",
    })
    if settings.db_user:
        env["PGUSER"] = settings.db_user
    if settings.db_password:
        env["PGPASSWORD"] = settings.db_password
    return env


class DataAgentLocalBackend(LocalShellBackend):
    """Local shell with bounded previews, persistent output, and timeout cleanup."""

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        effective_timeout = self._default_timeout if timeout is None else timeout
        if effective_timeout <= 0 or effective_timeout > self._default_timeout:
            return ExecuteResponse(output=f"Timeout must be between 1 and {self._default_timeout} seconds.", exit_code=1)
        if not isinstance(command, str) or not command.strip():
            return ExecuteResponse(output="Command must be a non-empty string.", exit_code=1)

        self.cwd.mkdir(parents=True, exist_ok=True)
        log_dir = self.cwd / "_execution"
        log_dir.mkdir(exist_ok=True)
        stem = uuid.uuid4().hex[:12]
        stdout_path = log_dir / f"{stem}.stdout.log"
        stderr_path = log_dir / f"{stem}.stderr.log"
        process: subprocess.Popen | None = None
        timed_out = False
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                kwargs: dict[str, Any] = {}
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    kwargs["start_new_session"] = True
                process = subprocess.Popen(
                    command, shell=True, cwd=self.cwd, env=self._env,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, **kwargs,
                )
                try:
                    process.wait(timeout=effective_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process_tree(process)
            exit_code = 124 if timed_out else process.returncode
        except OSError as exc:
            return ExecuteResponse(output=f"Could not execute command: {exc}", exit_code=1)

        stdout_preview, stdout_truncated = _log_preview(stdout_path)
        stderr_preview, stderr_truncated = _log_preview(stderr_path)
        parts = [stdout_preview or "<no stdout>"]
        if stderr_preview:
            parts.append(f"[stderr]\n{stderr_preview}")
        if timed_out:
            parts.append(f"Command timed out after {effective_timeout} seconds; its process tree was terminated.")
        parts.append(f"Complete logs: /_execution/{stem}.stdout.log and /_execution/{stem}.stderr.log")
        return ExecuteResponse(
            output="\n".join(parts), exit_code=exit_code,
            truncated=stdout_truncated or stderr_truncated,
        )


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW, check=False, timeout=10,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def _log_preview(path: Path) -> tuple[str, bool]:
    with path.open("rb") as stream:
        content = stream.read(EXECUTION_PREVIEW_BYTES + 1)
    return content[:EXECUTION_PREVIEW_BYTES].decode("utf-8", errors="replace"), len(content) > EXECUTION_PREVIEW_BYTES


def build_data_filesystem_middleware(
    *, settings: Settings, run_root: Path, agent_name: str, config: Any,
) -> FilesystemMiddleware:
    workspace = resolve_data_workspace(config=config, run_root=run_root, agent_name=agent_name)
    workspace.mkdir(parents=True, exist_ok=True)
    local = DataAgentLocalBackend(
        root_dir=workspace, virtual_mode=True,
        timeout=settings.data_agent_command_timeout_seconds,
        env=data_agent_environment(settings, workspace=workspace, agent_name=agent_name),
    )
    # Mount only source/docs/examples, excluding the project root's .env and
    # credentials. Shell execution still has the local user's OS privileges.
    backend = CompositeBackend(default=local, routes={
        f"/project/{name}/": _ReadOnlyReferenceBackend(root_dir=PROJECT_ROOT / name, virtual_mode=True)
        for name in ("agent", "docs", "examples")
    })

    shell = "Windows cmd.exe" if os.name == "nt" else "the system POSIX shell"
    return FilesystemMiddleware(
        backend=backend,
        max_execute_timeout=settings.data_agent_command_timeout_seconds,
        system_prompt=(
            "You have file tools (ls, glob, grep, read_file, write_file, edit_file) and execute. "
            "File tool paths use a virtual root / for your assigned data workspace. "
            "Read project references at /project/agent/, /project/docs/, and /project/examples/. "
            "These mounts are read-only through file tools. "
            f"execute runs locally in your assigned workspace using {shell}, not an OS sandbox. "
            "Use relative paths in commands: write_file('/analysis.py', ...) then execute('python analysis.py'). "
            "Virtual /project paths are not shell paths; import the agent package via the configured PYTHONPATH. "
            "python resolves to the workflow's Python environment. Database and embedding settings are injected. "
            "Never print credentials or read .env files. Save code and generated data in your workspace. "
            "Commands start fresh processes: persist work in files; changes to cwd or environment do not carry over. "
            "Use read_file with offset/limit to inspect complete results or execution logs in manageable chunks. "
            "Long stdout/stderr are saved under /_execution/; a truncated preview is not the complete result. "
            "Do not modify source data, project code, or the environment, install packages, or start background services."
        ),
        custom_tool_descriptions={
            "execute": "Execute a command locally in the assigned data workspace, with the project's Python and data-service configuration. Returns exit status and paths to complete output logs. This is local execution, not a sandbox.",
        },
    )
