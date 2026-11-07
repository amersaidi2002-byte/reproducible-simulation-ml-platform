import json
from pathlib import Path

import numpy as np
import pytest

from simml.simulator import (
    BOUNDS,
    OUTPUTS,
    SimulationConfig,
    simulate,
    validate_parameters,
)


def test_force_linearity_and_inverse_mass():
    a = np.array(list(simulate([7, 0.12, 8, 1, 1]).values()))
    np.testing.assert_allclose(
        list(simulate([7, 0.12, 8, 2, 1]).values()), 2 * a, rtol=1e-12
    )
    np.testing.assert_allclose(
        list(simulate([7, 0.12, 8, 1, 2]).values()), a / 2, rtol=1e-12
    )


def test_analytic_signal_against_independent_runge_kutta():
    params = [7.0, 0.12, 8.0, 1.5, 1.2]
    config = SimulationConfig(4096, 0.5)
    metrics, t, signal = simulate(params, config, return_signal=True)
    fn, zeta, fd, force, mass = params
    wn, w = 2 * np.pi * fn, 2 * np.pi * fd
    state = np.zeros(2)
    numerical = []

    def rhs(time, state):
        return np.array(
            [
                state[1],
                force / mass * np.sin(w * time)
                - 2 * zeta * wn * state[1]
                - wn**2 * state[0],
            ]
        )

    h = 1 / config.sample_rate_hz
    for time in t:
        numerical.append(state.copy())
        k1 = rhs(time, state)
        k2 = rhs(time + h / 2, state + h * k1 / 2)
        k3 = rhs(time + h / 2, state + h * k2 / 2)
        k4 = rhs(time + h, state + h * k3)
        state += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    np.testing.assert_allclose(
        signal[:, :2], np.array(numerical) * 1000, atol=2e-6, rtol=2e-6
    )
    np.testing.assert_allclose(signal[0], 0, atol=1e-12)
    assert set(metrics) == set(OUTPUTS)


@pytest.mark.parametrize(
    "values",
    [
        [1, 0.1, 8, 1, 1],
        [7, float("nan"), 8, 1, 1],
        [7, 0.1, 8, 1],
        [True, 0.1, 8, 1, 1],
    ],
)
def test_invalid_parameters(values):
    with pytest.raises(ValueError):
        validate_parameters(values)


def test_boundaries_and_config_validation():
    for p in (BOUNDS[:, 0], BOUNDS[:, 1]):
        assert all(np.isfinite(list(simulate(p).values())))
    for fs, duration in [(32, 4), (256, 0), (256, float("inf"))]:
        with pytest.raises(ValueError):
            SimulationConfig(fs, duration)


def test_checked_in_numerical_regression():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/simulator_v1.json").read_text()
    )
    for case in fixture["cases"]:
        actual = simulate(case["parameters"], SimulationConfig(**case["configuration"]))
        for name in OUTPUTS:
            np.testing.assert_allclose(
                actual[name], case["expected"][name], rtol=1e-10, atol=1e-10
            )
