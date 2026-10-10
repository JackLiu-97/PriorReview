from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ExecutionLog:
    """Own the compact per-run timeline, transcript, and run metadata."""

    def __init__(self, *, run_id: str, paper_id: str, paper_path: str | Path, run_dir: Path, models: dict[str, str | None]) -> None:
        self.run_id = run_id
        self.paper_id = paper_id
        self.paper_path = str(paper_path)
        self.run_dir = run_dir
        self.models = models
        self.timeline_path = self.run_dir / "timeline.jsonl"
        self.transcript_path = self.run_dir / "transcript.md"
        self.run_json_path = self.run_dir / "run.json"
        self._lock = threading.RLock()
        self._seq = 0
        self._started_at: str | None = None

    def start_run(self) -> None:
        with self._lock:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.timeline_path.write_text("", encoding="utf-8")
            self._started_at = self._now()
            self._write_run_json(status="running", started_at=self._started_at, ended_at=None, final_output=None)
            self.transcript_path.write_text(
                f"# Run {self.run_id}\n\n"
                f"- paper: {self.paper_id}\n"
                f"- source: {self.paper_path}\n"
                f"- status: running\n\n",
                encoding="utf-8",
            )
            self.record(
                event="run_start",
                name="workflow",
                input={"paper_path": self.paper_path},
                output={"run_dir": self.rel(self.run_dir)},
            )

    def complete_run(self, *, final_output: str | Path) -> None:
        ended_at = self._now()
        final_output_ref = self.rel(final_output)
        self.record(
            event="run_end",
            name="workflow",
            output={"status": "completed", "final_output": final_output_ref},
        )
        self._write_run_json(
            status="completed",
            started_at=self._started_at,
            ended_at=ended_at,
            final_output=final_output_ref,
        )

    def fail_run(self, *, error_type: str, error: str) -> None:
        ended_at = self._now()
        self.record(
            event="agent_error",
            name="workflow",
            output={"status": "failed", "error_type": error_type, "error": error},
        )
        self._write_run_json(
            status="failed",
            started_at=self._started_at,
            ended_at=ended_at,
            final_output=None,
            error={"type": error_type, "message": error},
        )

    def record(
        self,
        *,
        event: str,
        role: str | None = None,
        round_id: int | None = None,
        name: str | None = None,
        input: dict[str, Any] | None = None,  # noqa: A002 - matches the event schema.
        output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            payload = {
                "seq": self._seq,
                "time": self._now(),
                "role": role,
                "round": round_id,
                "event": event,
                "name": name,
                "input": self._safe_data(input or {}),
                "output": self._safe_data(output or {}),
            }
            with self.timeline_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._append_transcript(payload)
            return payload

    def rel(self, path: str | Path) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            return candidate.as_posix()
        try:
            return candidate.resolve(strict=False).relative_to(self.run_dir.resolve(strict=False)).as_posix()
        except ValueError:
            return str(candidate)

    def _write_run_json(
        self,
        *,
        status: str,
        started_at: str | None,
        ended_at: str | None,
        final_output: str | None,
        error: dict[str, str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "paper_id": self.paper_id,
            "paper_path": self.paper_path,
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at,
            "models": self.models,
            "final_output": final_output,
        }
        if error is not None:
            payload["error"] = error
        self.run_json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.run_json_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def _append_transcript(self, event: dict[str, Any]) -> None:
        text = self._transcript_text(event)
        if not text:
            return
        with self.transcript_path.open("a", encoding="utf-8") as fh:
            fh.write(text)

    def _transcript_text(self, event: dict[str, Any]) -> str:
        event_type = event["event"]
        role = event.get("role")
        round_id = event.get("round")
        name = event.get("name") or event_type
        input_payload = event.get("input") or {}
        output_payload = event.get("output") or {}

        if event_type == "run_start":
            return "## Run Start\n\n" + self._bullets({**input_payload, **output_payload}) + "\n"
        if event_type == "run_end":
            return "## Run End\n\n" + self._bullets(output_payload) + "\n"
        if event_type == "agent_input":
            return f"## {self._section_title(role, round_id)}\n\n### Input\n" + self._bullets(input_payload) + "\n"
        if event_type == "agent_output":
            return "### Agent Output\n" + self._bullets(output_payload) + "\n"
        if event_type == "tool_call":
            return f"### Tool Call: {name}\n" + self._bullets(input_payload) + "\n"
        if event_type == "tool_result":
            return f"### Tool Result: {name}\n" + self._bullets(output_payload) + "\n"
        if event_type == "agent_error":
            return f"### Error: {name}\n" + self._bullets(output_payload) + "\n"
        return ""

    @staticmethod
    def _section_title(role: str | None, round_id: int | None) -> str:
        if role == "finalizer":
            return "Finalizer"
        label = (role or "agent").replace("_", " ").title()
        return f"{label} Round {round_id:02d}" if round_id is not None else label

    @classmethod
    def _bullets(cls, payload: dict[str, Any]) -> str:
        if not payload:
            return "- none\n"
        lines: list[str] = []
        for key, value in payload.items():
            lines.append(f"- {key}: {cls._format_value(value)}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_value(value: Any) -> str:
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (str, int, float)):
            return str(value)
        return json.dumps(value, ensure_ascii=False)

    @classmethod
    def _safe_data(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): cls._safe_data(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._safe_data(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            json.dumps(value)
            return value
        except TypeError:
            return repr(value)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")
