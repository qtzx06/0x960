from __future__ import annotations

import importlib.util
import shutil
import tempfile
from collections.abc import Callable
from importlib import resources
from pathlib import Path

import chess


WorkspaceEvalFn = Callable[[chess.Board], int]


class WorkspaceManager:
    allowed_files = {"eval.py"}

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="zero960_"))
        self._write_template_files()

    def _write_template_files(self) -> None:
        template = resources.files("zero960.workspace_template").joinpath("eval.py").read_text()
        self.write_file("eval.py", template)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def resolve_path(self, relative_path: str) -> Path:
        if relative_path not in self.allowed_files:
            raise ValueError(f"writes are restricted to {sorted(self.allowed_files)}")
        return self.root / relative_path

    def read_file(self, relative_path: str) -> str:
        path = self.resolve_path(relative_path)
        return path.read_text()

    def write_file(self, relative_path: str, content: str) -> None:
        path = self.resolve_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def load_eval_function(self) -> WorkspaceEvalFn:
        path = self.resolve_path("eval.py")
        module_name = f"zero960_eval_{id(path)}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to create module spec for eval.py")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        eval_fn = getattr(module, "evaluate", None)
        if eval_fn is None or not callable(eval_fn):
            raise RuntimeError("eval.py must define evaluate(board)")
        return eval_fn

