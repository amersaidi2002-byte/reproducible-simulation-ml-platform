#!/usr/bin/env python3
"""Check a running container through real HTTP; fail on contract changes."""

import argparse
import json
import time
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PAYLOAD = {
    "parameters": [
        {
            "natural_frequency_hz": 7.0,
            "damping_ratio": 0.12,
            "drive_frequency_hz": 8.0,
            "force_n": 1.0,
            "mass_kg": 1.0,
        }
    ]
}


def wait_for_ready(url, attempts=40, interval=0.5):
    """Bounded retries for startup refusal, reset, timeout or incomplete HTTP."""
    for _ in range(attempts):
        try:
            with urlopen(url + "/health", timeout=2) as r:
                if json.load(r)["status"] == "ready":
                    return
        except (OSError, HTTPException):
            pass
        time.sleep(interval)
    raise RuntimeError("API never became ready")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    args = p.parse_args()
    wait_for_ready(args.url)
    request = Request(
        args.url + "/predict",
        data=json.dumps(PAYLOAD).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as r:
        result = json.load(r)
    assert len(result["predictions"]) == 1 and len(result["bundle_id"]) == 64
    assert all(v >= 0 for v in result["predictions"][0].values())
    bad = Request(
        args.url + "/predict",
        data=b'{"parameters":[]}',
        headers={"Content-Type": "application/json"},
    )
    try:
        urlopen(bad, timeout=5)
    except HTTPError as error:
        assert error.code == 422
    else:
        raise AssertionError("invalid request accepted")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
