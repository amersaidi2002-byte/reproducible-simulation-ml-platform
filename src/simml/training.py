"""Train a vibration-response surrogate; bind all weights to data/configs."""

import io
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .datasets import verify_dataset
from .provenance import atomic_bytes, atomic_json, digest, file_hash, source_fingerprint
from .tracking import Experiment

DEFAULT_TRAIN = {
    "epochs": 200,
    "seed": 2026,
    "width": 64,
    "batch_size": 128,
    "learning_rate": 0.003,
}
DEFAULT_EVALUATION = {
    "split": "test",
    "metrics": ["mae", "relative_mae", "r2"],
    "schema": "evaluation/1",
}


class Surrogate(nn.Module):
    """Physical float32 inputs → nonnegative physical RMS outputs."""

    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(
            nn.Linear(5, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 3),
        )
        self.register_buffer("x_mean", torch.zeros(5))
        self.register_buffer("x_std", torch.ones(5))
        self.register_buffer("y_mean", torch.zeros(3))
        self.register_buffer("y_std", torch.ones(3))

    def normalized(self, x):
        return self.net((x - self.x_mean) / self.x_std)

    def forward(self, x):
        return torch.expm1(self.normalized(x) * self.y_std + self.y_mean).clamp(min=0)


def load_checkpoint(path):
    path = Path(path)
    sidecar = json.loads(path.with_suffix(".json").read_text())
    if file_hash(path) != sidecar["checkpoint_sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["provenance"]
    if (
        digest(metadata) != sidecar["provenance_sha256"]
        or metadata != sidecar["provenance"]
    ):
        raise ValueError("checkpoint provenance mismatch")
    for name in ("training_config", "evaluation_config", "dataset_manifest"):
        if digest(metadata[name]) != metadata[name + "_sha256"]:
            raise ValueError("checkpoint configuration digest mismatch")
    model = Surrogate(metadata["training_config"]["width"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), metadata


def train(dataset, run_root, config=None, evaluation=None):
    config = {**DEFAULT_TRAIN, **(config or {})}
    evaluation = evaluation or DEFAULT_EVALUATION.copy()
    if (
        config["epochs"] < 1
        or config["batch_size"] < 1
        or config["width"] < 4
        or config["learning_rate"] <= 0
    ):
        raise ValueError("invalid training configuration")
    if evaluation != DEFAULT_EVALUATION:
        raise ValueError("unsupported evaluation configuration")
    arrays, manifest = verify_dataset(dataset)
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    x = torch.tensor(arrays["parameters"], dtype=torch.float32)
    y = torch.log1p(torch.tensor(arrays["targets"], dtype=torch.float32))
    train_idx = np.flatnonzero(arrays["split"] == 0)
    val_idx = np.flatnonzero(arrays["split"] == 1)
    model = Surrogate(config["width"])
    model.x_mean.copy_(x[train_idx].mean(0))
    model.x_std.copy_(x[train_idx].std(0).clamp(min=1e-6))
    model.y_mean.copy_(y[train_idx].mean(0))
    model.y_std.copy_(y[train_idx].std(0).clamp(min=1e-6))
    target = (y - model.y_mean) / model.y_std
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    with Experiment(
        run_root, "train", config, {"dataset_id": manifest["dataset_id"]}
    ) as run:
        provenance = {
            "schema": "checkpoint/1",
            "run_id": run.id,
            "dataset_id": manifest["dataset_id"],
            "dataset_manifest": manifest,
            "dataset_manifest_sha256": digest(manifest),
            "training_config": config,
            "training_config_sha256": digest(config),
            "evaluation_config": evaluation,
            "evaluation_config_sha256": digest(evaluation),
            "source_sha256": source_fingerprint(
                "training.py", "datasets.py", "simulator.py"
            ),
        }
        path = run.path / "checkpoint.pt"
        best = float("inf")
        best_epoch = 0
        for epoch in range(config["epochs"]):
            model.train()
            order = rng.permutation(train_idx)
            for start in range(0, len(order), config["batch_size"]):
                idx = order[start : start + config["batch_size"]]
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(model.normalized(x[idx]), target[idx])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val = float(
                    nn.functional.mse_loss(
                        model.normalized(x[val_idx]), target[val_idx]
                    )
                )
            run.event("epoch", epoch=epoch + 1, validation_loss=val)
            if val < best:
                best, best_epoch = val, epoch + 1
                buffer = io.BytesIO()
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "provenance": provenance,
                        "epoch": best_epoch,
                    },
                    buffer,
                )
                atomic_bytes(path, buffer.getvalue())
            if (epoch + 1) % 50 == 0:
                print(f"epoch {epoch + 1}: validation={val:.6f}", flush=True)
        sidecar = {
            "checkpoint_sha256": file_hash(path),
            "provenance_sha256": digest(provenance),
            "provenance": provenance,
            "best_epoch": best_epoch,
            "best_validation_loss": best,
        }
        atomic_json(path.with_suffix(".json"), sidecar)
        run.artifact("checkpoint", path)
        run.artifact("checkpoint_manifest", path.with_suffix(".json"))
    return path


def evaluate(checkpoint, dataset, run_root):
    model, metadata = load_checkpoint(checkpoint)
    arrays, manifest = verify_dataset(dataset)
    if (
        manifest["dataset_id"] != metadata["dataset_id"]
        or digest(manifest) != metadata["dataset_manifest_sha256"]
    ):
        raise ValueError("evaluation dataset differs from checkpoint provenance")
    config = metadata["evaluation_config"]
    idx = np.flatnonzero(arrays["split"] == 2)
    with Experiment(
        run_root,
        "evaluate",
        config,
        {
            "dataset_id": manifest["dataset_id"],
            "checkpoint_sha256": file_hash(checkpoint),
        },
    ) as run:
        with torch.no_grad():
            prediction = model(
                torch.tensor(arrays["parameters"][idx], dtype=torch.float32)
            ).numpy()
        target = arrays["targets"][idx]
        delta = prediction - target
        metrics = {
            "mae": np.mean(np.abs(delta), axis=0).tolist(),
            "relative_mae": (
                np.mean(np.abs(delta), axis=0) / np.mean(target, axis=0)
            ).tolist(),
            "r2": (
                1
                - np.sum(delta**2, axis=0)
                / np.sum((target - target.mean(0)) ** 2, axis=0)
            ).tolist(),
        }
        result = {
            "run_id": run.id,
            "rows": len(idx),
            "dataset_id": manifest["dataset_id"],
            "checkpoint_sha256": file_hash(checkpoint),
            "evaluation_config": config,
            "metrics": metrics,
        }
        atomic_json(run.path / "metrics.json", result)
        run.artifact("metrics", run.path / "metrics.json")
    return result
