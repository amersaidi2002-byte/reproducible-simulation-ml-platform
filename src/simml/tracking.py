"""Local experiment registry: config, immutable inputs, events and artifacts."""

import json
import time
import uuid
from pathlib import Path

from .provenance import atomic_json, canonical, digest, environment, file_hash


class Experiment:
    def __init__(self, root, kind, config, inputs):
        self.id = f"{kind}-{uuid.uuid4().hex}"
        self.path = Path(root) / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        self.record = {
            "run_id": self.id,
            "kind": kind,
            "status": "running",
            "started_unix": time.time(),
            "config": config,
            "config_sha256": digest(config),
            "inputs": inputs,
            "environment": environment(),
            "artifacts": {},
        }
        atomic_json(self.path / "run.json", self.record)

    def __enter__(self):
        return self

    def event(self, kind, **values):
        with (self.path / "events.jsonl").open("ab") as stream:
            stream.write(
                canonical({"time_unix": time.time(), "kind": kind, **values}) + b"\n"
            )
            stream.flush()

    def artifact(self, name, path):
        path = Path(path)
        self.record["artifacts"][name] = {
            "file_name": path.name,
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
        atomic_json(self.path / "run.json", self.record)

    def __exit__(self, typ, value, traceback):
        self.record.update(
            status="failed" if typ else "succeeded", finished_unix=time.time()
        )
        if typ:
            self.record["error"] = f"{typ.__name__}: {value}"
        atomic_json(self.path / "run.json", self.record)
        return False


def list_runs(root):
    return [json.loads(p.read_text()) for p in sorted(Path(root).glob("*/run.json"))]
