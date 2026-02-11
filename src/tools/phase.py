from __future__ import annotations

import numpy as np
from typing import Optional


_TWO_PI = 2.0 * np.pi


def principal_value(angle: np.ndarray) -> np.ndarray:
    """
    Wrap angles to the principal range [-pi, pi).

    Works with any shape/dtype convertible to float.
    """
    angle = np.asarray(angle, dtype=np.float32)
    return (angle + np.pi) % _TWO_PI - np.pi


def phase_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Principal (wrapped) phase difference: principal_value(a - b).
    """
    return principal_value(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))


def unwrap_temporal(
    phase: np.ndarray,
    axis: int = 0,
    ref_index: int = 0,
) -> np.ndarray:
    """
    Temporal phase unwrapping via cumulative sum of principal differences.

    Given phase(t) in [-pi, pi), returns an unwrapped version over time:
      unwrapped[t] = unwrapped[t-1] + principal_value(phase[t] - phase[t-1])

    Parameters
    ----------
    phase:
        Array containing phase values (e.g., [T, H, W] or [H, W, T]).
    axis:
        Time axis.
    ref_index:
        Which frame to use as reference (typically 0). Unwrapped at ref_index equals original.

    Returns
    -------
    unwrapped:
        Same shape as phase, float32.
    """
    ph = np.asarray(phase, dtype=np.float32)
    axis = int(axis)

    if ph.shape[axis] < 1:
        return ph.copy()

    # Move time axis to 0 for simpler logic
    ph0 = np.moveaxis(ph, axis, 0)  # [T, ...]
    T = ph0.shape[0]

    # Compute wrapped temporal differences
    d = principal_value(ph0[1:] - ph0[:-1])  # [T-1, ...]

    # Cumulative sum to unwrap (relative to ph0[0])
    unwrapped0 = np.empty_like(ph0, dtype=np.float32)
    unwrapped0[0] = ph0[0]
    unwrapped0[1:] = ph0[0] + np.cumsum(d, axis=0)

    # If reference index isn't 0, shift so that unwrapped[ref] = phase[ref]
    if ref_index != 0:
        ref_index = int(ref_index)
        ref_index = max(0, min(ref_index, T - 1))
        shift = ph0[ref_index] - unwrapped0[ref_index]
        unwrapped0 = unwrapped0 + shift  # broadcast shift over time

    # Move axis back
    return np.moveaxis(unwrapped0, 0, axis)


def phase_to_unit_complex(phase: np.ndarray) -> np.ndarray:
    """
    Convert phase to unit complex representation exp(1j * phase).
    Useful for robust averaging / reference phase estimation.

    Returns complex64 array.
    """
    ph = np.asarray(phase, dtype=np.float32)
    return np.exp(1j * ph).astype(np.complex64)


def circular_mean(phase: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Circular mean of phase angles.

    mean = angle(mean(exp(1j*phase)))

    Returns float32 in [-pi, pi).
    """
    z = phase_to_unit_complex(phase)
    m = np.mean(z, axis=axis)
    return np.angle(m).astype(np.float32)


def temporal_phase_reference(
    phase: np.ndarray,
    axis: int = 0,
    mode: str = "first",
) -> np.ndarray:
    """
    Compute a reference phase along time axis.

    mode:
      - "first": uses phase at first frame
      - "mean": circular mean over time (robust to wrapping)

    Returns array with time axis removed (i.e., shape phase.shape without axis).
    """
    ph = np.asarray(phase, dtype=np.float32)
    axis = int(axis)

    if mode == "first":
        return np.take(ph, indices=0, axis=axis).astype(np.float32)
    if mode == "mean":
        return circular_mean(ph, axis=axis)
    raise ValueError(f"Unknown mode='{mode}'. Use 'first' or 'mean'.")


def amplify_phase(
    phase: np.ndarray,
    delta_phase: np.ndarray,
    alpha: float,
    wrap_output: bool = False,
) -> np.ndarray:
    """
    Compose amplified phase: phase' = phase + alpha * delta_phase.

    Typically:
      phase = original phase
      delta_phase = bandpassed (and often unwrapped) temporal phase variation.

    If wrap_output=True, wraps to [-pi, pi) (sometimes useful before reconstruction).
    """
    ph = np.asarray(phase, dtype=np.float32)
    dp = np.asarray(delta_phase, dtype=np.float32)
    out = ph + np.float32(alpha) * dp
    return principal_value(out) if wrap_output else out
