from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ArtifactManager:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def ensure_dir(self, relative_dir: str) -> Path:
        path = self.run_dir / relative_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative_path: str, data: Any) -> str:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        return str(path)

    def write_text(self, relative_path: str, text: str) -> str:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

