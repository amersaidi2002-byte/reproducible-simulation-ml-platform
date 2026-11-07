"""ONNX Runtime inference with no PyTorch dependency."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .provenance import digest, file_hash
from .simulator import BOUNDS, CONTRACT, OUTPUTS, PARAMETERS


class Predictor:
    def __init__(self, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        payload = {k: v for k, v in manifest.items() if k != "bundle_id"}
        if (
            manifest.get("schema") != "simml-bundle/1"
            or digest(payload) != manifest["bundle_id"]
        ):
            raise ValueError("bundle manifest identity mismatch")
        if file_hash(directory / "model.onnx") != manifest["onnx_sha256"]:
            raise ValueError("ONNX artifact checksum mismatch")
        if (
            manifest["simulator_contract"] != CONTRACT
            or manifest["input"]["parameters"] != list(PARAMETERS)
            or manifest["input"]["bounds"] != BOUNDS.tolist()
            or manifest["output"]["metrics"] != list(OUTPUTS)
            or manifest["input"]["max_batch"] != 256
        ):
            raise ValueError("bundle does not match serving contract")
        if (
            digest(manifest["evaluation_config"])
            != manifest["evaluation_config_sha256"]
        ):
            raise ValueError("evaluation configuration checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(directory / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or inputs[0].name != "parameters"
            or outputs[0].name != "metrics"
        ):
            raise ValueError("unexpected ONNX input/output names")
        if (
            inputs[0].shape[1:] != [5]
            or outputs[0].shape[1:] != [3]
            or inputs[0].type != "tensor(float)"
            or outputs[0].type != "tensor(float)"
        ):
            raise ValueError("unexpected ONNX tensor contract")
        props = self.session.get_modelmeta().custom_metadata_map
        for key in (
            "checkpoint_sha256",
            "dataset_id",
            "training_config_sha256",
            "evaluation_config_sha256",
        ):
            if props.get(key) != manifest[key]:
                raise ValueError("ONNX embedded provenance mismatch")
        self.manifest = manifest

    def predict(self, values):
        x = np.asarray(values, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 5 or not 1 <= len(x) <= 256:
            raise ValueError("expected float32 parameters [batch,5], batch in [1,256]")
        if (
            not np.isfinite(x).all()
            or np.any(x < BOUNDS[:, 0].astype(np.float32))
            or np.any(x > BOUNDS[:, 1].astype(np.float32))
        ):
            raise ValueError("non-finite or out-of-bounds float32 parameters")
        result = self.session.run(["metrics"], {"parameters": np.ascontiguousarray(x)})[
            0
        ]
        if (
            result.shape != (len(x), 3)
            or not np.isfinite(result).all()
            or np.any(result < 0)
        ):
            raise RuntimeError("invalid model output")
        return result
