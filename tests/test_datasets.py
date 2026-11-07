import json

import pytest

from simml.datasets import build_dataset, verify_dataset
from simml.simulator import SimulationConfig
from simml.tracking import Experiment, list_runs


def test_immutable_versions_group_splits_and_resume(tmp_path):
    args = (tmp_path / "datasets", tmp_path / "cache.sqlite")
    first, report = build_dataset(*args, families=10, members=3, workers=2)
    second, warm = build_dataset(*args, families=10, members=3, workers=1)
    assert first == second and report["simulator_executions"] == 30
    assert warm["cache_hits"] == 30 and warm["simulator_executions"] == 0
    arrays, manifest = verify_dataset(first)
    groups = [set(arrays["family_id"][arrays["split"] == i]) for i in range(3)]
    assert all(groups[i].isdisjoint(groups[j]) for i, j in [(0, 1), (0, 2), (1, 2)])
    changed, _ = build_dataset(
        *args, families=10, members=3, simulation=SimulationConfig(128)
    )
    assert changed != first
    assert json.loads((first / "manifest.json").read_text()) == manifest


def test_dataset_tampering_detected(tmp_path):
    path, _ = build_dataset(
        tmp_path / "datasets", tmp_path / "cache.sqlite", families=10, members=2
    )
    with (path / "designs.npz").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_dataset(path)


def test_failed_experiment_records_error(tmp_path):
    with pytest.raises(RuntimeError):
        with Experiment(tmp_path, "test", {"seed": 1}, {}) as run:
            run.event("started")
            raise RuntimeError("injected")
    record = list_runs(tmp_path)[0]
    assert record["status"] == "failed" and "injected" in record["error"]
    assert "config_sha256" in record
