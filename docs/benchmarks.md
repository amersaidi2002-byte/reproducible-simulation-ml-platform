# Reproducible benchmark protocol

The fixed experiment is `configs/reference.json`. All metrics below derive from committed JSON evidence, not the proposed CV wording.

## Data and checkpoint selection

80 anchor families × 40 local parameter variants = 3,200 simulations. Families, rather than individual rows, are shuffled into 70% training / 15% validation / 15% test. This gives 2,240 training, 480 validation and 480 test examples. Sibling variants never cross the split.

The model has two 64-unit SiLU layers. Input means/stds and the mean/std of log1p(target) are fitted only to training examples. The inverse transform is inside the inference graph. Adam trains for 250 epochs with seed 2026 and batch size 128; the lowest validation MSE selects the checkpoint. The held-out set is evaluated only after that selection. There is one dataset seed and one training seed, so this is not a seed-robustness or out-of-domain study.

MAE uses the physical output units. `relative_mae` = mean absolute error divided by mean test target, per output; it is **not** mean absolute percentage error. R² is 1 − residual sum of squares / total sum of squares. See `reports/evaluation.json` for the exact values and `reports/worst-cases.json` for individual high-error cases.

## Cache measurement

A fresh local SQLite file is used for the first reference pass. The second pass has identical parameters, code/dependency identity and simulation settings, and reads the completed jobs. The measured second pass performs zero simulator executions. Timings include dataset assembly and verification, not just solver work; do not interpret the warm/cold ratio as a pure physics speedup.

Changing the simulation configuration, simulator source or NumPy version intentionally creates different job keys and dataset identities. The `cold_or_resumed` label is retained in the script because rerunning it against an existing work directory can produce a resumed first pass; inspect execution counts before calling that pass cold.

## Numerical parity

Reference is CPU float32 PyTorch inference on physical inputs. Compare the same inputs through ONNX Runtime 1.24.4 in Python and compiled C++. Test batches 1, 7, 64 and 256, including exact float32 parameter boundaries. Each output must satisfy:

    |candidate − reference| <= 0.0002 + 0.00002 * |reference|

Outputs have different physical units and scales; therefore the global maximum absolute error must not be read as a displacement error. Near-zero outputs may have a larger relative error while passing the absolute tolerance. `reports/parity.json` contains both extrema and the exact model/checkpoint hashes.

This checks implementation/export agreement, not surrogate-versus-physics accuracy. The latter is a separate held-out evaluation. The C++ executable is actually compiled and executed in these checks.

## Timing

Use one inference thread, batch size 32, 20 warmup calls and 200 measured calls. Record median and p95 monotonic-clock durations. The simulator baseline computes 32 independent signals; the surrogate approximates their RMS metrics. The simulator timing includes Python input validation, and ORT timing includes serving-side validation. Model loading, HTTP transport, C++ process startup and filesystem I/O are excluded. No end-to-end C++ latency advantage is claimed.

The reference environment and library versions are recorded with the timing results. The workload is small enough that scheduling noise and validation overhead matter; use the provided commands to measure your own environment.

## Reproduction and automated gates

1. Install the pinned serving dependencies plus training/test extras.
2. Fetch matching ORT headers, compile C++, and run `scripts/reproduce.py`.
3. Run the test suite with `SIMML_CPP_BINARY` set.
4. `scripts/check_reference.py` checks the frozen held-out values, data and model identities, and a maximum 10% mean-normalized MAE per output.
5. CI successfully builds the Docker image, starts a read-only non-root container, checks real HTTP behavior and verifies PyTorch is absent. All 26 tests and the C++ client pass. See `reports/deployment-validation.json` for the exact validated commit and workflow run. A separate isolated Python serving environment is also tested through actual HTTP.

The CI latency gate deliberately checks no wall-clock threshold: host contention would make that flaky. Correctness and artifact/data consistency are the automated regression gates.

## Implementation references

- [PyTorch ONNX export](https://docs.pytorch.org/docs/stable/onnx.html): dynamo export and dynamic input shapes.
- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/get-started/with-cpp.html): C++ sessions and tensor execution.
- [FastAPI Docker deployment](https://fastapi.tiangolo.com/deployment/docker/): application container and exec-form startup.
- [FastAPI lifespan testing](https://fastapi.tiangolo.com/advanced/testing-events/): startup verification through a context-managed test client.
