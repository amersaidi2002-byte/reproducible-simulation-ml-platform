#!/usr/bin/env python3
"""Check a running container through real HTTP; fail on contract changes."""

import argparse
import json
import time
from urllib.error import HTTPError, URLError
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    args = p.parse_args()
    ready = False
    for _ in range(40):
        try:
            with urlopen(args.url + "/health", timeout=2) as r:
                ready = json.load(r)["status"] == "ready"
            if ready:
                break
        except (URLError, TimeoutError):
            time.sleep(0.5)
    if not ready:
        raise RuntimeError("API never became ready")
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
