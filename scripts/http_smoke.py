#!/usr/bin/env python3
"""Launch Uvicorn and exercise real HTTP within one process environment."""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--require-no-torch", action="store_true")
    args = parser.parse_args()
    if args.require_no_torch:
        subprocess.run(
            [
                args.python,
                "-c",
                "import importlib.util; assert importlib.util.find_spec('torch') is None",
            ],
            check=True,
        )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "SIMML_BUNDLE": str(root / "artifacts/bundle")}
    process = subprocess.Popen(
        [
            args.python,
            "-m",
            "uvicorn",
            "simml.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("server exited during startup")
            try:
                with urlopen(url + "/health", timeout=0.5) as response:
                    assert response.status == 200
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("server did not become ready")
        subprocess.run(
            [sys.executable, str(root / "scripts/container_smoke.py"), "--url", url],
            check=True,
            timeout=10,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        print(process.stdout.read().decode())


if __name__ == "__main__":
    main()
