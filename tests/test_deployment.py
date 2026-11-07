import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from simml.api import create_app
from simml.inference import Predictor
from simml.provenance import digest
from simml.simulator import BOUNDS, PARAMETERS
from simml.training import load_checkpoint
from simml.validation import parity

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def test_python_onnx_and_cpp_parity():
    binary = os.environ.get("SIMML_CPP_BINARY")
    if not binary:
        if os.environ.get("CI"):
            pytest.fail("CI must compile C++ and set SIMML_CPP_BINARY")
        pytest.skip("set SIMML_CPP_BINARY after building the C++ client")
    report = parity(ARTIFACTS / "checkpoint.pt", ARTIFACTS / "bundle", binary)
    assert len(report["cases"]) == 8


def test_model_golden_regression():
    fixture = json.loads((ROOT / "tests/fixtures/inference_v1.json").read_text())
    predictor = Predictor(ARTIFACTS / "bundle")
    assert fixture["onnx_sha256"] == predictor.manifest["onnx_sha256"]
    np.testing.assert_allclose(
        predictor.predict(fixture["inputs"]), fixture["expected"], rtol=2e-5, atol=2e-4
    )


def test_http_contract_and_batch_boundaries():
    with TestClient(create_app(ARTIFACTS / "bundle")) as client:
        assert client.get("/health").json()["status"] == "ready"
        for count in (1, 256):
            parameters = [dict(zip(PARAMETERS, BOUNDS[:, 0].tolist()))] * count
            response = client.post("/predict", json={"parameters": parameters})
            assert (
                response.status_code == 200
                and len(response.json()["predictions"]) == count
            )
        for parameters in (
            [],
            [dict(zip(PARAMETERS, BOUNDS[:, 0].tolist()))] * 257,
            [{"unknown": 7}],
            [{**dict(zip(PARAMETERS, BOUNDS[:, 0].tolist())), "mass_kg": True}],
            [{**dict(zip(PARAMETERS, BOUNDS[:, 0].tolist())), "force_n": 100}],
        ):
            assert (
                client.post("/predict", json={"parameters": parameters}).status_code
                == 422
            )


def test_corrupt_bundle_and_manifest_refused(tmp_path):
    target = tmp_path / "bundle"
    shutil.copytree(ARTIFACTS / "bundle", target)
    model = target / "model.onnx"
    model.write_bytes(model.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        Predictor(target)
    shutil.copyfile(ARTIFACTS / "bundle/model.onnx", model)
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["dataset_id"] = "wrong"
    manifest["bundle_id"] = digest(
        {k: v for k, v in manifest.items() if k != "bundle_id"}
    )
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance"):
        Predictor(target)


def test_checkpoint_data_and_config_lineage():
    _, metadata = load_checkpoint(ARTIFACTS / "checkpoint.pt")
    manifest = json.loads((ARTIFACTS / "dataset/manifest.json").read_text())
    assert metadata["dataset_id"] == manifest["dataset_id"]
    assert metadata["dataset_manifest_sha256"] == digest(manifest)
    assert metadata["evaluation_config"]["split"] == "test"


def test_cpp_rejects_invalid_inputs(tmp_path):
    binary = os.environ.get("SIMML_CPP_BINARY")
    if not binary:
        pytest.skip("C++ client not configured")
    input_path = tmp_path / "input.txt"
    for text in ["0\n", "257\n", "1\n7 .1 8 1\n", "1\n7 .1 8 1 1 trailing\n"]:
        input_path.write_text(text)
        result = subprocess.run(
            [
                binary,
                str(ARTIFACTS / "bundle/model.onnx"),
                str(input_path),
                str(tmp_path / "out.txt"),
            ],
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 2


def test_nonfinite_http_input_is_a_json_validation_error():
    payload = '{"parameters":[{"natural_frequency_hz":7,"damping_ratio":NaN,"drive_frequency_hz":8,"force_n":1,"mass_kg":1}]}'
    with TestClient(create_app(ARTIFACTS / "bundle")) as client:
        response = client.post(
            "/predict", content=payload, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422 and "detail" in response.json()
