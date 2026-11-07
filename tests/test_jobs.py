import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from simml.jobs import JobBusy, JobCache, JobFailed, LostLease
from simml.simulator import SimulationConfig, simulate

P = [7, 0.1, 8, 1, 1]


def test_parameter_hash_order_config_and_cache(tmp_path):
    cache = JobCache(tmp_path / "jobs.sqlite")
    result, first = cache.run(P)
    _, hit = cache.run(
        dict(
            reversed(
                list(
                    zip(
                        (
                            "natural_frequency_hz",
                            "damping_ratio",
                            "drive_frequency_hz",
                            "force_n",
                            "mass_kg",
                        ),
                        P,
                    )
                )
            )
        )
    )
    assert not first["cache_hit"] and hit["cache_hit"] and first["key"] == hit["key"]
    assert cache.request(P, SimulationConfig(128))[0] != first["key"]
    assert cache.run([7, 0.1, 8, 2, 1])[1]["key"] != first["key"]
    cache.identity = {**cache.identity, "source_sha256": "different-code"}
    assert cache.run(P)[1]["key"] != first["key"]
    assert result == simulate(P)


def test_failed_job_retry_and_limit(tmp_path):
    cache = JobCache(tmp_path / "jobs.sqlite", max_attempts=2)

    def failure(*_):
        raise RuntimeError("injected transient failure")

    with pytest.raises(RuntimeError):
        cache.run(P, _simulate=failure)
    result, metadata = cache.run(P)
    assert result == simulate(P) and metadata["attempts"] == 2
    other = [8, 0.1, 8, 1, 1]
    for _ in range(2):
        with pytest.raises(RuntimeError):
            cache.run(other, _simulate=failure)
    with pytest.raises(JobFailed):
        cache.run(other)


def test_expired_lease_recovery(tmp_path):
    cache = JobCache(tmp_path / "jobs.sqlite")

    def failure(*_):
        raise RuntimeError("fault")

    with pytest.raises(RuntimeError):
        cache.run(P, _simulate=failure)
    with cache.connect() as db:
        db.execute(
            "UPDATE jobs SET status='running',owner='dead-worker',lease_until=?",
            (time.time() - 1,),
        )
    assert cache.run(P)[1]["attempts"] == 2


def test_active_lease_and_owner_fencing(tmp_path):
    cache = JobCache(tmp_path / "jobs.sqlite")
    barrier = Barrier(2)

    def paused(*args):
        barrier.wait(timeout=5)
        barrier.wait(timeout=5)
        return simulate(*args)

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(cache.run, P, _simulate=paused)
        barrier.wait(timeout=5)
        with pytest.raises(JobBusy):
            cache.run(P)
        with cache.connect() as db:
            db.execute("UPDATE jobs SET lease_until=?", (time.time() - 1,))
        recovered = cache.run(P)
        barrier.wait(timeout=5)
        with pytest.raises(LostLease):
            future.result(timeout=5)
    assert recovered[1]["attempts"] == 2 and cache.run(P)[1]["cache_hit"]


def test_corrupted_cache_is_not_silently_reused(tmp_path):
    cache = JobCache(tmp_path / "jobs.sqlite")
    cache.run(P)
    with sqlite3.connect(cache.path) as db:
        db.execute("UPDATE jobs SET result_hash='bad'")
    with pytest.raises(JobFailed, match="checksum"):
        cache.run(P)
