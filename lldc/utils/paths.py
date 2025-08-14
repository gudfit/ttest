# lldc/utils/paths.py
from __future__ import annotations
from pathlib import Path


class Paths:
    def __init__(self, root: str | Path = "."):
        root = Path(root)
        self.root = root
        self.artifacts = root / "artifacts"
        self.checkpoints = self.artifacts / "checkpoints"
        self.logs = self.artifacts / "logs"
        self.runs = self.artifacts / "runs"
        self.payloads = self.artifacts / "payloads"
        self.position_codecs = self.artifacts / "position_codecs"
        self.rd_curves = self.artifacts / "rd_curves"
        self.results = self.artifacts / "results"

    def ensure(self):
        for p in [
            self.artifacts,
            self.checkpoints,
            self.logs,
            self.runs,
            self.payloads,
            self.position_codecs,
            self.rd_curves,
            self.results,
        ]:
            p.mkdir(parents=True, exist_ok=True)
        return self
