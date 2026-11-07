"""Contracted forced-oscillator simulator: SI inputs, millimetre-based outputs.

m x'' + 2*zeta*m*wn*x' + m*wn²*x = F*sin(w*t), x(0)=x'(0)=0.
The underdamped analytic solution includes the decaying transient; it is not
an equilibrium-only approximation. No external solver is required.
"""

from dataclasses import asdict, dataclass

import numpy as np

PARAMETERS = (
    "natural_frequency_hz",
    "damping_ratio",
    "drive_frequency_hz",
    "force_n",
    "mass_kg",
)
BOUNDS = np.array([[2.0, 15.0], [0.03, 0.3], [1.0, 20.0], [0.2, 3.0], [0.5, 3.0]])
OUTPUTS = ("displacement_rms_mm", "velocity_rms_mm_s", "acceleration_rms_mm_s2")
CONTRACT = "forced-oscillator/1"


@dataclass(frozen=True)
class SimulationConfig:
    sample_rate_hz: int = 256
    duration_s: float = 4.0

    def __post_init__(self):
        if (
            type(self.sample_rate_hz) is not int
            or not 64 <= self.sample_rate_hz <= 4096
        ):
            raise ValueError("sample_rate_hz must be an integer in [64,4096]")
        if not np.isfinite(self.duration_s) or not 0.5 <= self.duration_s <= 20:
            raise ValueError("duration_s must be in [0.5,20]")

    def as_dict(self):
        return asdict(self)


def validate_parameters(values):
    if isinstance(values, dict):
        if set(values) != set(PARAMETERS):
            raise ValueError("parameter names must exactly match simulator contract")
        values = [values[name] for name in PARAMETERS]
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError("expected a numeric parameter vector")
    if any(
        isinstance(v, (bool, np.bool_))
        or not isinstance(v, (int, float, np.integer, np.floating))
        for v in values
    ):
        raise ValueError("boolean parameters are not supported")
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (5,) or not np.isfinite(x).all():
        raise ValueError("expected five finite numeric parameters")
    if np.any(x < BOUNDS[:, 0]) or np.any(x > BOUNDS[:, 1]):
        raise ValueError("parameters outside documented simulator bounds")
    return {name: float(v) for name, v in zip(PARAMETERS, x)}


def simulate(values, config=SimulationConfig(), return_signal=False):
    p = validate_parameters(values)
    fn, zeta, fd, force, mass = (p[name] for name in PARAMETERS)
    wn, w = 2 * np.pi * fn, 2 * np.pi * fd
    a, wd = zeta * wn, wn * np.sqrt(1 - zeta * zeta)
    den = (wn * wn - w * w) ** 2 + (2 * a * w) ** 2
    s = force / mass * (wn * wn - w * w) / den
    c = -force / mass * 2 * a * w / den
    A = -c
    B = (a * A - s * w) / wd
    t = (
        np.arange(round(config.duration_s * config.sample_rate_hz))
        / config.sample_rate_hz
    )
    q = A * np.cos(wd * t) + B * np.sin(wd * t)
    x = s * np.sin(w * t) + c * np.cos(w * t) + np.exp(-a * t) * q
    v = (
        s * w * np.cos(w * t)
        - c * w * np.sin(w * t)
        + np.exp(-a * t) * (-a * q - A * wd * np.sin(wd * t) + B * wd * np.cos(wd * t))
    )
    acc = force / mass * np.sin(w * t) - 2 * a * v - wn * wn * x
    signals = np.stack([x, v, acc], axis=1) * 1000
    result = {
        name: float(value)
        for name, value in zip(OUTPUTS, np.sqrt(np.mean(signals**2, axis=0)))
    }
    validate_result(result)
    return (result, t, signals) if return_signal else result


def validate_result(result):
    if set(result) != set(OUTPUTS) or any(
        not np.isfinite(v) or v < 0 for v in result.values()
    ):
        raise ValueError(
            "simulator returned invalid metric names, non-finite or negative RMS"
        )
