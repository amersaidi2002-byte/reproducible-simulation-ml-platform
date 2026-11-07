"""Canonical hashes, atomic local writes, and code/environment fingerprints."""

import dataclasses
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

import numpy as np


def canonical(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    atomic_bytes(
        path,
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n",
    )


def source_fingerprint(*names):
    root = Path(__file__).parent
    return digest({name: file_hash(root / name) for name in sorted(names)})


def environment():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def arrays_hash(arrays):
    """Stable logical identity independent of ZIP timestamps/compression."""
    h = hashlib.sha256()
    for name, arr in sorted(arrays.items()):
        arr = np.ascontiguousarray(arr)
        h.update(
            canonical({"name": name, "dtype": arr.dtype.str, "shape": list(arr.shape)})
        )
        h.update(arr.tobytes())
    return h.hexdigest()
