import json

import pytest

from simml.datasets import build_dataset
from simml.provenance import digest
from simml.training import evaluate, load_checkpoint, train


@pytest.mark.integration
def test_training_binds_data_configs_and_evaluation(tmp_path):
    data, _ = build_dataset(
        tmp_path / "data", tmp_path / "jobs.sqlite", families=10, members=5
    )
    checkpoint = train(data, tmp_path / "runs", {"epochs": 3, "width": 16})
    _, provenance = load_checkpoint(checkpoint)
    assert provenance["training_config_sha256"] == digest(provenance["training_config"])
    result = evaluate(checkpoint, data, tmp_path / "runs")
    assert result["dataset_id"] == provenance["dataset_id"]
    other, _ = build_dataset(
        tmp_path / "data", tmp_path / "jobs.sqlite", families=10, members=5, seed=7
    )
    with pytest.raises(ValueError, match="differs"):
        evaluate(checkpoint, other, tmp_path / "runs")
    sidecar = checkpoint.with_suffix(".json")
    metadata = json.loads(sidecar.read_text())
    metadata["provenance"]["evaluation_config"]["split"] = "train"
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="provenance"):
        load_checkpoint(checkpoint)
