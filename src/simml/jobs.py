"""SQLite simulation cache with leases, owner fencing and bounded retries.

Exactly-once *publication*, not exactly-once execution: an expired worker may
still compute after another worker acquires its lease. Its output is rejected.
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .provenance import canonical, digest, source_fingerprint
from .simulator import (
    CONTRACT,
    SimulationConfig,
    simulate,
    validate_parameters,
    validate_result,
)


class JobBusy(RuntimeError):
    pass


class JobFailed(RuntimeError):
    pass


class LostLease(RuntimeError):
    pass


def simulator_identity():
    return {
        "contract": CONTRACT,
        "source_sha256": source_fingerprint("simulator.py"),
        "numpy": np.__version__,
    }


class JobCache:
    def __init__(self, path, lease_seconds=60, max_attempts=3):
        if lease_seconds <= 0 or max_attempts < 1:
            raise ValueError("positive lease and retry count required")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lease_seconds, self.max_attempts = lease_seconds, max_attempts
        self.identity = simulator_identity()
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                key TEXT PRIMARY KEY, request TEXT NOT NULL, status TEXT NOT NULL,
                attempts INTEGER NOT NULL, owner TEXT, lease_until REAL,
                result TEXT, result_hash TEXT, error TEXT, updated REAL NOT NULL)""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def request(self, parameters, config):
        request = {
            "parameters": validate_parameters(parameters),
            "simulation": config.as_dict(),
            "simulator": self.identity,
        }
        return digest(request), request

    def run(
        self,
        parameters,
        config=SimulationConfig(),
        _simulate=None,
        retry_failed=True,
    ):
        key, request = self.request(parameters, config)
        owner = uuid.uuid4().hex
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE key=?", (key,)).fetchone()
            if row and row["status"] == "succeeded":
                result = json.loads(row["result"])
                validate_result(result)
                if digest(result) != row["result_hash"]:
                    raise JobFailed(
                        "cached result checksum mismatch; delete corrupted cache entry explicitly"
                    )
                return result, {
                    "key": key,
                    "cache_hit": True,
                    "attempts": row["attempts"],
                }
            if row and row["status"] == "running" and row["lease_until"] > now:
                raise JobBusy(key)
            attempts = (row["attempts"] if row else 0) + 1
            if row and (
                attempts > self.max_attempts
                or (row["status"] == "failed" and not retry_failed)
            ):
                raise JobFailed(f"retry limit or disabled retry: {key}: {row['error']}")
            db.execute(
                """INSERT INTO jobs VALUES (?,?, 'running',?,?,?,NULL,NULL,NULL,?)
                ON CONFLICT(key) DO UPDATE SET status='running',attempts=excluded.attempts,
                owner=excluded.owner,lease_until=excluded.lease_until,error=NULL,updated=excluded.updated""",
                (
                    key,
                    canonical(request).decode(),
                    attempts,
                    owner,
                    now + self.lease_seconds,
                    now,
                ),
            )
        try:
            result = (_simulate or simulate)(request["parameters"], config)
            validate_result(result)
            encoded = canonical(result).decode()
        except Exception as exc:
            with self.connect() as db:
                db.execute(
                    "UPDATE jobs SET status='failed',error=?,lease_until=NULL,updated=? WHERE key=? AND owner=? AND status='running'",
                    (f"{type(exc).__name__}: {exc}", time.time(), key, owner),
                )
            raise
        with self.connect() as db:
            changed = db.execute(
                "UPDATE jobs SET status='succeeded',result=?,result_hash=?,lease_until=NULL,updated=? WHERE key=? AND owner=? AND status='running'",
                (encoded, digest(result), time.time(), key, owner),
            ).rowcount
        if changed != 1:
            raise LostLease(key)
        return result, {"key": key, "cache_hit": False, "attempts": attempts}

    def stats(self):
        with self.connect() as db:
            return {
                r["status"]: r["n"]
                for r in db.execute(
                    "SELECT status,COUNT(*) AS n FROM jobs GROUP BY status"
                )
            }
