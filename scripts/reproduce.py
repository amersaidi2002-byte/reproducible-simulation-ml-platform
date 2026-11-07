#!/usr/bin/env python3
"""Build the reference data/model/bundle and publish auditable local evidence."""

import argparse
import json
import shutil
import time
from pathlib import Path

from simml.datasets import build_dataset
from simml.export import export_onnx
from simml.provenance import atomic_json, file_hash
from simml.simulator import SimulationConfig
from simml.training import evaluate, train
from simml.validation import latency, parity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reference.json")
    parser.add_argument("--work", default="work")
    parser.add_argument("--output", default=".")
    parser.add_argument("--cpp-binary")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    work, output = Path(args.work), Path(args.output)
    artifact = output / "artifacts"
    report = output / "reports"
    artifact.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    simulation = SimulationConfig(**config["simulation"])
    timings = []
    for name in ("cold_or_resumed", "warm"):
        start = time.perf_counter()
        dataset, stats = build_dataset(
            work / "datasets",
            work / "jobs.sqlite",
            simulation=simulation,
            **config["dataset"],
        )
        timings.append(
            {"pass": name, "elapsed_seconds": time.perf_counter() - start, **stats}
        )
    checkpoint = train(dataset, work / "runs", config["training"], config["evaluation"])
    evaluation = evaluate(checkpoint, dataset, work / "runs")
    for source, target in [
        (checkpoint, artifact / "checkpoint.pt"),
        (checkpoint.with_suffix(".json"), artifact / "checkpoint.json"),
    ]:
        shutil.copyfile(source, target)
    shutil.copytree(dataset, artifact / "dataset", dirs_exist_ok=True)
    bundle = export_onnx(artifact / "checkpoint.pt", artifact / "bundle")
    numeric = parity(artifact / "checkpoint.pt", artifact / "bundle", args.cpp_binary)
    performance = latency(artifact / "checkpoint.pt", artifact / "bundle")
    atomic_json(
        report / "cache_benchmark.json", {"configuration": config, "passes": timings}
    )
    atomic_json(report / "evaluation.json", evaluation)
    atomic_json(report / "parity.json", numeric)
    atomic_json(report / "latency.json", performance)
    atomic_json(
        report / "lineage.json",
        {
            "dataset_id": stats["dataset_id"],
            "checkpoint_sha256": file_hash(artifact / "checkpoint.pt"),
            "bundle_id": bundle["bundle_id"],
            "onnx_sha256": bundle["onnx_sha256"],
            "training_run": checkpoint.parent.name,
            "evaluation_run": evaluation["run_id"],
            "configuration": config,
        },
    )
    for run in [checkpoint.parent, work / "runs" / evaluation["run_id"]]:
        destination = report / "experiments" / run.name
        destination.mkdir(parents=True, exist_ok=True)
        for file in run.glob("*"):
            if file.suffix in (".json", ".jsonl"):
                shutil.copyfile(file, destination / file.name)
    print(
        json.dumps(
            {"dataset": stats, "evaluation": evaluation["metrics"], "parity": numeric},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
