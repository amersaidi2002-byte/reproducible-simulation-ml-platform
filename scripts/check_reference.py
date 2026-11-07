#!/usr/bin/env python3
"""Verify committed dataset/model/evaluation links and frozen accuracy results."""

import json
from pathlib import Path

import numpy as np
import torch

from simml.datasets import verify_dataset
from simml.provenance import file_hash
from simml.training import load_checkpoint


def main():
    root = Path(__file__).resolve().parents[1]
    data, manifest = verify_dataset(root / "artifacts/dataset")
    model, metadata = load_checkpoint(root / "artifacts/checkpoint.pt")
    torch.set_num_threads(1)
    frozen = json.loads((root / "reports/evaluation.json").read_text())
    assert frozen["dataset_id"] == metadata["dataset_id"] == manifest["dataset_id"]
    assert frozen["checkpoint_sha256"] == file_hash(root / "artifacts/checkpoint.pt")
    indices = np.flatnonzero(data["split"] == 2)
    with torch.no_grad():
        prediction = model(
            torch.tensor(data["parameters"][indices], dtype=torch.float32)
        ).numpy()
    target = data["targets"][indices]
    delta = prediction - target
    actual = {
        "mae": np.mean(np.abs(delta), axis=0),
        "relative_mae": np.mean(np.abs(delta), axis=0) / np.mean(target, axis=0),
        "r2": 1
        - np.sum(delta**2, axis=0) / np.sum((target - target.mean(0)) ** 2, axis=0),
    }
    for name, values in actual.items():
        np.testing.assert_allclose(
            values, frozen["metrics"][name], rtol=1e-4, atol=1e-5
        )
    assert max(actual["relative_mae"]) < 0.1
    print("Dataset, checkpoint lineage and frozen held-out metrics verified.")


if __name__ == "__main__":
    main()
