from __future__ import annotations

import numpy as np
from typing import Literal, Optional, Tuple

from scipy.signal import butter, sosfilt, sosfiltfilt


FilterMode = Literal["iir", "ideal_fft"]
IIRApply = Literal["sosfiltfilt", "sosfilt"]


def _validate_band(fl: float, fh: float, fps: float) -> Tuple[float, float, float]:
    fps = float(fps)
    if not (fps > 0 and np.isfinite(fps)):
        raise ValueError(f"fps must be positive and finite. Got fps={fps}")
    fl = float(fl)
    fh = float(fh)
    if fl < 0 or fh < 0:
        raise ValueError(f"fl/fh must be >= 0. Got fl={fl}, fh={fh}")
    if fh <= fl:
        raise ValueError(f"Need fh > fl. Got fl={fl}, fh={fh}")
    nyq = fps / 2.0
    if fh >= nyq:
        # clamp a bit under nyquist to avoid instability
        fh = 0.999 * nyq
    if fl <= 0:
        fl = 1e-6
    return fl, fh, fps


def design_butter_bandpass_sos(
    fl: float,
    fh: float,
    fps: float,
    order: int = 2,
) -> np.ndarray:
    """
    Design Butterworth bandpass in SOS form.

    fl, fh in Hz. fps in frames/sec.
    """
    fl, fh, fps = _validate_band(fl, fh, fps)
    nyq = fps / 2.0
    low = fl / nyq
    high = fh / nyq
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid normalized band: low={low}, high={high} (fps={fps})")
    sos = butter(int(order), [low, high], btype="bandpass", output="sos")
    return sos


def apply_iir_bandpass(
    x: np.ndarray,
    fl: float,
    fh: float,
    fps: float,
    axis: int = 0,
    order: int = 2,
    apply: IIRApply = "sosfiltfilt",
    zi: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply IIR bandpass to signal along `axis`.

    apply:
      - "sosfiltfilt": zero-phase (offline; needs full signal)
      - "sosfilt": causal (streaming-friendly; supports zi)

    Returns filtered array (float32).
    """
    x = np.asarray(x, dtype=np.float32)
    sos = design_butter_bandpass_sos(fl, fh, fps, order=order)

    axis = int(axis)
    if x.shape[axis] < 3 and apply == "sosfiltfilt":
        # not enough samples for filtfilt to be stable
        apply = "sosfilt"

    if apply == "sosfiltfilt":
        y = sosfiltfilt(sos, x, axis=axis).astype(np.float32)
        return y

    if apply == "sosfilt":
        # NOTE: scipy's sosfilt supports zi, but users may ignore streaming until later.
        y = sosfilt(sos, x, axis=axis, zi=zi)[0] if zi is not None else sosfilt(sos, x, axis=axis)
        return np.asarray(y, dtype=np.float32)

    raise ValueError(f"Unknown apply='{apply}'.")


def apply_ideal_bandpass_fft(
    x: np.ndarray,
    fl: float,
    fh: float,
    fps: float,
    axis: int = 0,
) -> np.ndarray:
    """
    Ideal (brick-wall) bandpass using FFT along `axis`.

    Pros: very direct and often used in EVM references.
    Cons: offline only; can ring; needs full time buffer.

    Returns filtered array (float32).
    """
    x = np.asarray(x, dtype=np.float32)
    fl, fh, fps = _validate_band(fl, fh, fps)
    axis = int(axis)

    n = x.shape[axis]
    if n < 2:
        return x.copy()

    # rFFT along time axis
    X = np.fft.rfft(x, axis=axis)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)

    # mask frequencies within [fl, fh]
    mask = (freqs >= fl) & (freqs <= fh)

    # reshape mask to broadcast along axis
    shape = [1] * x.ndim
    shape[axis] = mask.shape[0]
    mask_b = mask.reshape(shape)

    X_filtered = X * mask_b
    y = np.fft.irfft(X_filtered, n=n, axis=axis)
    return np.asarray(y, dtype=np.float32)


def bandpass(
    x: np.ndarray,
    fl: float,
    fh: float,
    fps: float,
    axis: int = 0,
    mode: FilterMode = "iir",
    order: int = 2,
    iir_apply: IIRApply = "sosfiltfilt",
) -> np.ndarray:
    """
    Convenience wrapper.

    mode:
      - "iir": Butterworth bandpass (default)
      - "ideal_fft": brick-wall FFT bandpass
    """
    if mode == "iir":
        return apply_iir_bandpass(x, fl, fh, fps, axis=axis, order=order, apply=iir_apply)
    if mode == "ideal_fft":
        return apply_ideal_bandpass_fft(x, fl, fh, fps, axis=axis)
    raise ValueError(f"Unknown mode='{mode}'. Use 'iir' or 'ideal_fft'.")


def gaussian_second_derivative_kernel(
    fps: float,
    w_hz: float,
    truncate: float = 3.0,
) -> np.ndarray:
    """
    Build a temporal second-derivative-of-Gaussian (LoG in 1D time) kernel.

    Zhang et al. select sigma from target frequency w:
      sigma = r / (4 w sqrt(2)), r=fps.  (paper)
    We implement kernel in discrete time (frames).
    """
    if w_hz <= 0:
        raise ValueError("w_hz must be > 0")

    sigma = float(fps) / (4.0 * float(w_hz) * np.sqrt(2.0))
    # radius in frames
    radius = int(np.ceil(truncate * sigma))
    t = np.arange(-radius, radius + 1, dtype=np.float32)

    # continuous-time 2nd derivative of Gaussian (up to scale):
    # G(t)=exp(-t^2/(2s^2)); G''(t)=((t^2 - s^2)/s^4)*exp(-t^2/(2s^2))
    s2 = sigma * sigma
    s4 = s2 * s2
    g = np.exp(-(t * t) / (2.0 * s2)).astype(np.float32)
    g2 = ((t * t - s2) / (s4 + 1e-12)).astype(np.float32) * g

    # zero-mean already; normalize L1 of abs to keep scale stable across sigma
    denom = float(np.sum(np.abs(g2)) + 1e-12)
    g2 = (g2 / denom).astype(np.float32)
    return g2


def acceleration_filter(
    x: np.ndarray,
    fps: float,
    w_hz: float,
    axis: int = 0,
    truncate: float = 3.0,
) -> np.ndarray:
    """
    Apply temporal acceleration filter (2nd deriv of Gaussian) along `axis`.
    Offline convolution with reflect padding (good quality).
    """
    k = gaussian_second_derivative_kernel(fps=fps, w_hz=w_hz, truncate=truncate)
    x = np.asarray(x, dtype=np.float32)

    # move axis to front
    x0 = np.moveaxis(x, axis, 0)
    T = x0.shape[0]
    pad = len(k) // 2

    # reflect pad
    xpad = np.pad(x0, [(pad, pad)] + [(0, 0)] * (x0.ndim - 1), mode="reflect")

    # convolve along time
    y = np.zeros_like(x0, dtype=np.float32)
    for i in range(T):
        window = xpad[i : i + len(k)]  # [K,...]
        y[i] = np.tensordot(k, window, axes=(0, 0))

    # move axis back
    y = np.moveaxis(y, 0, axis)
    return y.astype(np.float32)