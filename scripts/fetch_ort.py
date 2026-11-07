#!/usr/bin/env python3
"""Prepare a Linux SDK from the pinned Python wheel + upstream tagged headers.

The shared library comes from the official onnxruntime==1.24.4 wheel.
Headers come from the same Microsoft release tag. Record checksums for audit.
This downloads build dependencies; nothing is vendored into the repository.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

import onnxruntime

VERSION = "1.24.4"
HEADERS = [
    "onnxruntime_c_api.h",
    "onnxruntime_cxx_api.h",
    "onnxruntime_cxx_inline.h",
    "onnxruntime_float16.h",
    "onnxruntime_ep_c_api.h",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="build/ort")
    args = parser.parse_args()
    if onnxruntime.__version__ != VERSION:
        raise RuntimeError("install onnxruntime==" + VERSION)
    root = Path(args.output).resolve()
    (root / "include").mkdir(parents=True, exist_ok=True)
    (root / "lib").mkdir(exist_ok=True)
    hashes = {}
    for name in HEADERS:
        url = f"https://raw.githubusercontent.com/microsoft/onnxruntime/v{VERSION}/include/onnxruntime/core/session/{name}"
        with urlopen(url, timeout=30) as response:
            content = response.read()
        (root / "include" / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    library = (
        Path(onnxruntime.__file__).parent / "capi" / f"libonnxruntime.so.{VERSION}"
    )
    target = root / "lib" / library.name
    shutil.copyfile(library, target)
    hashes[library.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    for alias in ("libonnxruntime.so", "libonnxruntime.so.1"):
        link = root / "lib" / alias
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.name)
    (root / "dependencies.json").write_text(
        json.dumps({"version": VERSION, "sha256": hashes}, indent=2) + "\n"
    )
    print(root)


if __name__ == "__main__":
    main()
