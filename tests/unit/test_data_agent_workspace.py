from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.core.compiled_data_subagent import ControlledDataSubagentTools
from agent.core.data_agent_workspace import (
    EXECUTION_PREVIEW_BYTES,
    build_data_filesystem_middleware,
    data_agent_environment,
    resolve_data_workspace,
)
from agent.core.settings import Settings


class DataAgentWorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "中文 run with spaces"
        self.workspace = self.root / "artifacts" / "data" / "reviewer_round_01"
        self.config = {"configurable": {"assigned_data_workspace_abs": str(self.workspace)}}
        self.settings = Settings(data_agent_command_timeout_seconds=10)

    def backend(self, config=None, settings=None):
        middleware = build_data_filesystem_middleware(
            settings=settings or self.settings, run_root=self.root, agent_name="reviewer",
            config=config or self.config,
        )
        return middleware.backend

    def test_file_write_execute_and_read_use_project_python_in_assigned_directory(self) -> None:
        backend = self.backend()
        script = (
            "import json, sys\nfrom pathlib import Path\n"
            "from agent.core.settings import Settings\n"
            "Path('result.json').write_text(json.dumps({'cwd': str(Path.cwd()), 'python': sys.executable, "
            "'database': Settings.from_env().db_name, 'answer': sum([1, 2, 3])}), encoding='utf-8')\n"
            "print('分析完成')\n"
        )
        self.assertIsNone(backend.write("/analysis.py", script).error)
        result = backend.execute("python analysis.py")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("分析完成", result.output)
        payload = json.loads((self.workspace / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(payload["cwd"]), self.workspace.resolve())
        self.assertEqual(Path(payload["python"]), Path(sys.executable))
        self.assertEqual(payload["answer"], 6)
        self.assertIsNone(backend.read("/result.json").error)

    def test_failure_can_be_inspected_edited_and_rerun(self) -> None:
        backend = self.backend()
        backend.write("/analysis.py", "print(unknown_name)\n")
        failed = backend.execute("python analysis.py")
        self.assertNotEqual(failed.exit_code, 0)
        self.assertIn("NameError", failed.output)
        self.assertIsNone(backend.edit("/analysis.py", "unknown_name", "6 * 7").error)
        repaired = backend.execute("python analysis.py")
        self.assertEqual(repaired.exit_code, 0, repaired.output)
        self.assertIn("42", repaired.output)
        self.assertEqual(len(list((self.workspace / "_execution").glob("*.stderr.log"))), 2)

    def test_complete_logs_survive_preview_truncation(self) -> None:
        backend = self.backend()
        backend.write("/long_output.py", f"print('x' * {EXECUTION_PREVIEW_BYTES + 100} + 'END_MARKER')\n")
        result = backend.execute("python long_output.py")
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.truncated)
        self.assertNotIn("END_MARKER", result.output)
        log = next((self.workspace / "_execution").glob("*.stdout.log"))
        self.assertIn("END_MARKER", log.read_text(encoding="utf-8"))

    def test_timeout_terminates_shell_child_before_it_can_write(self) -> None:
        backend = self.backend(settings=Settings(data_agent_command_timeout_seconds=1))
        backend.write("/slow.py", "import time\nfrom pathlib import Path\ntime.sleep(2)\nPath('late.txt').write_text('late')\n")
        result = backend.execute("python slow.py")
        self.assertEqual(result.exit_code, 124, result.output)
        time.sleep(1.3)
        self.assertFalse((self.workspace / "late.txt").exists())
        self.assertEqual(backend.execute("python slow.py", timeout=2).exit_code, 1)

    def test_reference_file_tools_are_read_only_and_do_not_mount_root_secrets(self) -> None:
        backend = self.backend()
        reference = "/project/agent/core/sql_context.py"
        self.assertIsNone(backend.read(reference).error)
        self.assertIn("read-only", backend.write(reference, "replacement").error)
        self.assertIn("read-only", backend.edit(reference, "a", "b").error)
        self.assertEqual(backend.upload_files([(reference, b"replacement")])[0].error, "permission_denied")
        self.assertIsNotNone(backend.read("/project/.env").error)
        with self.assertRaises(ValueError):
            backend.write("/../escape.py", "pass")

    def test_different_rounds_get_different_workspaces(self) -> None:
        first = self.backend()
        first.write("/round.txt", "first")
        second_workspace = self.workspace.with_name("reviewer_round_02")
        second = self.backend({"configurable": {"assigned_data_workspace_abs": str(second_workspace)}})
        self.assertIsNotNone(second.read("/round.txt").error)
        second.write("/round.txt", "second")
        self.assertEqual((self.workspace / "round.txt").read_text(), "first")
        self.assertEqual((second_workspace / "round.txt").read_text(), "second")

    def test_runtime_assignment_rejects_missing_outside_and_other_role_paths(self) -> None:
        for value in (None, str(self.root / "outside"), str(self.workspace.with_name("author_round_01"))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_data_workspace(
                    config={"configurable": {"assigned_data_workspace_abs": value}},
                    run_root=self.root, agent_name="reviewer",
                )
        # The live helper path must not regain authority from model-written text.
        tools = ControlledDataSubagentTools(settings=self.settings, run_root=self.root, agent_name="reviewer")
        state = {"messages": [{"content": f'assigned_data_workspace_abs: "{self.root / "outside"}"'}]}
        result = json.loads(tools.write_sql_artifact(state=state, filename="query.sql", sql="SELECT 1"))
        self.assertFalse(result["ok"])
        self.assertFalse((self.root / "outside").exists())

    def test_environment_passes_data_settings_without_model_credentials(self) -> None:
        settings = Settings(openai_api_key="main-secret", finalizer_openai_api_key="final-secret", db_password="db-secret")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "parent-secret", "UNRELATED_TOKEN": "other-secret"}):
            env = data_agent_environment(settings, workspace=self.workspace, agent_name="reviewer")
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("FINALIZER_OPENAI_API_KEY", env)
        self.assertNotIn("UNRELATED_TOKEN", env)
        self.assertEqual(env["PGPASSWORD"], settings.db_password)
        self.assertIn("default_transaction_read_only=on", env["PGOPTIONS"])
        self.assertEqual(env["PYTHON_DOTENV_DISABLED"], "1")


if __name__ == "__main__":
    unittest.main()
