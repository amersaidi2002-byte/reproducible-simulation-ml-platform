"""Immutable, content-addressed datasets assembled from resumable jobs."""

import io
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .jobs import JobCache, simulator_identity
from .provenance import arrays_hash, atomic_bytes, atomic_json, digest, file_hash
from .simulator import BOUNDS, OUTPUTS, PARAMETERS, SimulationConfig


def verify_dataset(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    data_path = path / "designs.npz"
    if file_hash(data_path) != manifest["file_sha256"]:
        raise ValueError("dataset file checksum mismatch")
    with np.load(data_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    if arrays_hash(arrays) != manifest["arrays_sha256"]:
        raise ValueError("dataset logical content checksum mismatch")
    identity = {
        "schema": manifest["schema"],
        "config": manifest["config"],
        "arrays_sha256": manifest["arrays_sha256"],
    }
    if digest(identity) != manifest["dataset_id"]:
        raise ValueError("dataset manifest identity mismatch")
    return arrays, manifest


def build_dataset(
    root,
    cache_path,
    families=60,
    members=20,
    seed=2026,
    workers=4,
    simulation=SimulationConfig(),
):
    if families < 10 or members < 2 or workers < 1:
        raise ValueError("at least 10 families, 2 members and 1 worker required")
    rng = np.random.default_rng(seed)
    family = np.repeat(np.arange(families), members)
    anchors = rng.uniform(0.1, 0.9, (families, 5))
    u = np.clip(anchors[family] + rng.normal(0, 0.12, (len(family), 5)), 0, 1)
    parameters = np.clip(
        BOUNDS[:, 0] + u * np.diff(BOUNDS, axis=1).ravel(), BOUNDS[:, 0], BOUNDS[:, 1]
    )
    split_family = np.full(families, 2, dtype=np.int64)
    order = rng.permutation(families)
    split_family[order[: int(0.7 * families)]] = 0
    split_family[order[int(0.7 * families) : int(0.85 * families)]] = 1
    cache = JobCache(cache_path)

    def run(row):
        result, job = cache.run(row, simulation)
        return [result[k] for k in OUTPUTS], job

    # executor.map preserves input order irrespective of worker scheduling.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run, parameters))
    arrays = {
        "parameters": parameters,
        "targets": np.asarray([r[0] for r in rows]),
        "family_id": family,
        "split": split_family[family],
        "job_key": np.array([r[1]["key"] for r in rows], dtype="U64"),
    }
    config = {
        "families": families,
        "members": members,
        "seed": seed,
        "simulation": simulation.as_dict(),
        "simulator": simulator_identity(),
        "generator": "anchor-normal-0.12/v1",
    }
    identity = {
        "schema": "simml-dataset/1",
        "config": config,
        "arrays_sha256": arrays_hash(arrays),
    }
    dataset_id = digest(identity)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / dataset_id
    if not destination.exists():
        staging = Path(tempfile.mkdtemp(dir=root, prefix=".pending-"))
        try:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            atomic_bytes(staging / "designs.npz", buffer.getvalue())
            manifest = {
                **identity,
                "dataset_id": dataset_id,
                "file_sha256": file_hash(staging / "designs.npz"),
                "rows": len(family),
                "parameters": list(PARAMETERS),
                "outputs": list(OUTPUTS),
                "splits": {
                    name: int(np.sum(arrays["split"] == i))
                    for i, name in enumerate(["train", "validation", "test"])
                },
            }
            atomic_json(staging / "manifest.json", manifest)
            try:
                staging.rename(destination)
            except OSError:
                if not destination.exists():
                    raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    _, manifest = verify_dataset(destination)
    return destination, {
        "dataset_id": dataset_id,
        "rows": manifest["rows"],
        "cache_hits": sum(r[1]["cache_hit"] for r in rows),
        "simulator_executions": sum(not r[1]["cache_hit"] for r in rows),
    }
