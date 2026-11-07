"""Measured numerical parity and latency (not a substitute for model accuracy)."""

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .inference import Predictor
from .provenance import environment, file_hash
from .simulator import BOUNDS, simulate
from .training import load_checkpoint


def cpp_predict(binary, model, values):
    with tempfile.TemporaryDirectory() as directory:
        input_path, output_path = (
            Path(directory) / "input.txt",
            Path(directory) / "output.txt",
        )
        with input_path.open("w") as stream:
            stream.write(str(len(values)) + "\n")
            np.savetxt(stream, values, fmt="%.9g")
        subprocess.run(
            [
                str(Path(binary).resolve()),
                str(Path(model).resolve()),
                str(input_path),
                str(output_path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return np.loadtxt(output_path, dtype=np.float32, ndmin=2)


def parity(checkpoint, bundle, cpp_binary=None):
    model, _ = load_checkpoint(checkpoint)
    predictor = Predictor(bundle)
    rng = np.random.default_rng(7026)
    reports = []
    for batch in (1, 7, 64, 256):
        x = (
            BOUNDS[:, 0] + rng.random((batch, 5)) * np.diff(BOUNDS, axis=1).ravel()
        ).astype(np.float32)
        if batch > 1:
            x[:2] = BOUNDS.T.astype(np.float32)
        with torch.no_grad():
            reference = model(torch.from_numpy(x)).numpy()
        backends = {"onnx_python": predictor.predict(x)}
        if cpp_binary:
            backends["onnx_cpp"] = cpp_predict(
                cpp_binary, Path(bundle) / "model.onnx", x
            )
        for name, result in backends.items():
            np.testing.assert_allclose(result, reference, rtol=2e-5, atol=2e-4)
            reports.append(
                {
                    "backend": name,
                    "batch": batch,
                    "max_absolute_error": float(np.max(np.abs(result - reference))),
                    "max_relative_error": float(
                        np.max(
                            np.abs(result - reference)
                            / np.maximum(np.abs(reference), 1e-4)
                        )
                    ),
                }
            )
    return {
        "tolerance": {"rtol": 2e-5, "atol": 2e-4},
        "cases": reports,
        "checkpoint_sha256": file_hash(checkpoint),
        "onnx_sha256": file_hash(Path(bundle) / "model.onnx"),
    }


def latency(checkpoint, bundle, repeats=200):
    torch.set_num_threads(1)
    model, _ = load_checkpoint(checkpoint)
    predictor = Predictor(bundle)
    x = np.tile([7.0, 0.12, 8.0, 1.0, 1.0], (32, 1)).astype(np.float32)
    tensor = torch.from_numpy(x)

    @torch.no_grad()
    def pytorch():
        return model(tensor)

    functions = {
        "pytorch_batch_32": pytorch,
        "onnx_batch_32": lambda: predictor.predict(x),
        "simulator_batch_32": lambda: [simulate(row.astype(float)) for row in x],
    }
    report = {}
    for name, fn in functions.items():
        for _ in range(20):
            fn()
        durations = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            fn()
            durations.append((time.perf_counter_ns() - start) / 1e6)
        report[name] = {
            "median_ms": float(np.median(durations)),
            "p95_ms": float(np.quantile(durations, 0.95)),
        }
    return {
        "repeats": repeats,
        "warmups": 20,
        "threads": 1,
        "batch": 32,
        "environment": environment(),
        "torch": str(torch.__version__),
        "measurements": report,
        "excludes": ["model loading", "HTTP", "C++ process startup", "filesystem I/O"],
    }
