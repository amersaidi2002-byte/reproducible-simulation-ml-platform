# Failure cases and recovery boundaries

| Case | Expected behavior | Automated evidence |
|---|---|---|
| Container startup resets or closes an HTTP connection | Retry readiness within a bounded timeout; prediction failures still fail the check | `test_readiness_retries_connection_reset_and_incomplete_http` |
| A job raises a transient exception | Record `failed` and the exception; the next request claims attempt 2 | `test_failed_job_retry_and_limit` |
| Repeated failures exceed three attempts | Raise `JobFailed`; no infinite retry loop | Same test (uses a two-attempt limit for speed) |
| Worker disappears while its job is `running` | Reclaim after its lease expires | `test_expired_lease_recovery` |
| Expired worker finishes after a replacement worker | Reject its result with `LostLease`; preserve replacement's result | `test_active_lease_and_owner_fencing` |
| Another worker holds a live lease | Raise `JobBusy`; caller can retry later | Same test |
| Cache result is corrupted | Refuse cached result; require explicit invalidation | `test_corrupted_cache_is_not_silently_reused` |
| Data ZIP or data/config identity is altered | Reject the dataset during verification | `test_dataset_tampering_detected` |
| A checkpoint is paired with another dataset | Reject evaluation instead of silently mixing versions | `test_training_binds_data_configs_and_evaluation` |
| Checkpoint manifest or model bytes are altered | Reject load; do not serve a partially updated bundle | Training/deployment tampering tests |
| HTTP request contains unknown fields, booleans, wrong bounds, empty/oversized batch | Return HTTP 422 with JSON error details | HTTP contract tests |
| HTTP request contains NaN/Infinity | Return a JSON validation error, not an error containing non-JSON numbers | `test_nonfinite_http_input_is_a_json_validation_error` |
| C++ input is incomplete, has a bad batch or trailing tokens | Exit with code 2 and an explanatory stderr message | `test_cpp_rejects_invalid_inputs` |
| Surrogate approximation is poor near resonance | Return the numerical prediction; use test-set residuals to inspect suitability | `reports/worst-cases.json` |

Recovery tests inject faults on purpose. They demonstrate state-machine behavior; they are not a claim of production outage experience.

The 60-second lease is not renewed. A computation that exceeds it may be executed twice; owner fencing prevents the older owner publishing after a replacement claim. This local example's simulator runs in milliseconds. For a long external solver, use a renewable heartbeat, a sufficiently long lease, and explicit resource cancellation before extending this design.

A crash after checkpoint bytes are replaced but before the sidecar is written produces a rejected checkpoint, which is preferable to silently serving mismatched metadata. Artifact packaging is not a distributed multi-file transaction; publish a complete versioned bundle and switch its directory as one deployment operation.

Dataset generation clips sampled physical values to exact contract bounds. This avoids a floating-point boundary error such as `0.03 + (0.30 - 0.03)` becoming slightly larger than 0.30. Serving bounds are checked in float32, matching ONNX/C++; simulation inputs are validated in float64.

The reference API has no authentication and is demonstrated with a loopback host port. Public hosting, rate limits, service authentication, multi-host queues and production monitoring are outside this project's measured scope.
