"""Export a self-contained physical-input ONNX model with artifact lineage."""

from pathlib import Path

import onnx
import torch

from .provenance import atomic_json, digest, file_hash
from .simulator import BOUNDS, CONTRACT, OUTPUTS, PARAMETERS
from .training import load_checkpoint


def export_onnx(checkpoint, directory):
    model, metadata = load_checkpoint(checkpoint)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.onnx"
    example = torch.tensor(
        [[7.0, 0.1, 8.0, 1.0, 1.0], [5.0, 0.2, 4.0, 2.0, 2.0]], dtype=torch.float32
    )
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=["parameters"],
        output_names=["metrics"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=256)},),
    )
    graph = onnx.load(path)
    onnx.helper.set_model_props(
        graph,
        {
            "dataset_id": metadata["dataset_id"],
            "checkpoint_sha256": file_hash(checkpoint),
            "evaluation_config_sha256": metadata["evaluation_config_sha256"],
            "training_config_sha256": metadata["training_config_sha256"],
            "simulator_contract": CONTRACT,
        },
    )
    onnx.checker.check_model(graph)
    onnx.save(graph, path)
    manifest = {
        "schema": "simml-bundle/1",
        "onnx_sha256": file_hash(path),
        "checkpoint_sha256": file_hash(checkpoint),
        "dataset_id": metadata["dataset_id"],
        "training_config_sha256": metadata["training_config_sha256"],
        "evaluation_config": metadata["evaluation_config"],
        "evaluation_config_sha256": metadata["evaluation_config_sha256"],
        "simulator_contract": CONTRACT,
        "input": {
            "name": "parameters",
            "dtype": "float32",
            "shape": ["batch", 5],
            "parameters": list(PARAMETERS),
            "bounds": BOUNDS.tolist(),
            "max_batch": 256,
        },
        "output": {
            "name": "metrics",
            "dtype": "float32",
            "shape": ["batch", 3],
            "metrics": list(OUTPUTS),
        },
    }
    manifest["bundle_id"] = digest(manifest)
    atomic_json(directory / "manifest.json", manifest)
    return manifest
