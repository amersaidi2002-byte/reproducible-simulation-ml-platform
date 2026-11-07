import importlib.util
import io
from http.client import RemoteDisconnected
from pathlib import Path

import pytest


def smoke_module():
    path = Path(__file__).resolve().parents[1] / "scripts/container_smoke.py"
    spec = importlib.util.spec_from_file_location("container_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readiness_retries_connection_reset_and_incomplete_http(monkeypatch):
    module = smoke_module()
    calls = []

    def request(_url, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise ConnectionResetError("container not listening yet")
        if len(calls) == 2:
            raise RemoteDisconnected("startup closed connection")
        return io.BytesIO(b'{"status":"ready"}')

    monkeypatch.setattr(module, "urlopen", request)
    module.wait_for_ready("http://127.0.0.1:8000", attempts=3, interval=0)
    assert len(calls) == 3


def test_readiness_stops_after_bounded_failure(monkeypatch):
    module = smoke_module()
    calls = []

    def request(_url, timeout):
        calls.append(timeout)
        raise ConnectionRefusedError("not running")

    monkeypatch.setattr(module, "urlopen", request)
    with pytest.raises(RuntimeError, match="never became ready"):
        module.wait_for_ready("http://127.0.0.1:8000", attempts=3, interval=0)
    assert len(calls) == 3
