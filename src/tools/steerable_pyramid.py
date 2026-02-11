from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np


@dataclass(frozen=True)
class PyramidSpec:
    """
    Paper-grade (Simoncelli-style) steerable pyramid spec (frequency domain).

    - n_scales: number of scales (levels) excluding residuals
    - n_orientations: number of orientation bands; in pyrtools this is order+1
    - twidth: radial transition width (in octaves)
    """
    n_scales: int = 3
    n_orientations: int = 4
    twidth: int = 1


KeyT = Tuple[Union[int, str], Union[int, str]]


def _require_pyrtools():
    try:
        import pyrtools as pt  # type: ignore
        return pt
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'pyrtools'. Install with: pip install pyrtools\n"
            f"Original import error: {e}"
        ) from e


def _split_coeffs(pyr_coeffs: Dict[Any, np.ndarray]) -> Tuple[Optional[Any], Optional[Any], List[int], List[int], Dict[Tuple[int, int], Any]]:
    """
    Robust key parser for pyrtools.pyr_coeffs across versions.

    Returns:
      hi_key, lo_key,
      levels (sorted),
      bands (sorted),
      key_map: (level, band) -> full_key_in_dict
    """
    hi_key = None
    lo_key = None
    levels: set[int] = set()
    bands: set[int] = set()
    key_map: Dict[Tuple[int, int], Any] = {}

    for key in pyr_coeffs.keys():
        # normalize: key may be tuple of varying length OR string OR int
        if isinstance(key, tuple):
            parts = key
        else:
            parts = (key,)

        # detect residual keys by any string part
        lowered = " ".join([p.lower() for p in parts if isinstance(p, str)])
        if lowered:
            if "high" in lowered:
                hi_key = key
                continue
            if "low" in lowered:
                lo_key = key
                continue

        # detect oriented bands by first two int parts (level, band)
        if len(parts) >= 2 and isinstance(parts[0], int) and isinstance(parts[1], int):
            lvl = int(parts[0])
            bnd = int(parts[1])
            levels.add(lvl)
            bands.add(bnd)
            # keep the first full key we see for this (lvl,bnd)
            key_map.setdefault((lvl, bnd), key)

    return hi_key, lo_key, sorted(levels), sorted(bands), key_map



class ComplexSteerablePyramid:
    """
    Paper-grade Complex Steerable Pyramid wrapper (Simoncelli / pyrtools).

    forward(x) -> dict with:
      - 'hi': residual highpass (H,W) float32/complex64 (depends on pyrtools key)
      - 'bands': List[List[band]]  bands[scale][ori] complex64
      - 'lo': residual lowpass (h_s,w_s) float32

    inverse(coeffs) -> reconstructed image float32 at full resolution.
    """

    def __init__(self, height: int, width: int, spec: PyramidSpec):
        self.H = int(height)
        self.W = int(width)
        self.spec = spec

        if self.spec.n_scales < 0:
            raise ValueError("n_scales must be >= 0")
        if self.spec.n_orientations < 1:
            raise ValueError("n_orientations must be >= 1")

        # pyrtools SteerablePyramidFreq uses `order` and yields (order+1) orientation bands.
        self.order = int(self.spec.n_orientations - 1)
        if self.order < 0:
            raise ValueError("n_orientations must be >= 1 (so order >= 0)")

    def forward(self, x: np.ndarray) -> Dict[str, Any]:
        pt = _require_pyrtools()

        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("forward expects a 2D array (grayscale/luminance).")
        if x.shape != (self.H, self.W):
            raise ValueError(f"Input shape {x.shape} != pyramid shape {(self.H, self.W)}")

        # NOTE: pyrtools uses circular boundary handling in frequency domain. :contentReference[oaicite:2]{index=2}
        pyr = pt.pyramids.SteerablePyramidFreq(
            x,
            height=int(self.spec.n_scales),
            order=int(self.order),
            twidth=int(self.spec.twidth),
            is_complex=True,
        )

        coeffs = pyr.pyr_coeffs  # dict: (level, band) -> array
        hi_key, lo_key, levels, bands, key_map = _split_coeffs(coeffs)


        # residuals
        hi = np.zeros((self.H, self.W), dtype=np.float32)
        if hi_key is not None and hi_key in coeffs:
            hi_arr = coeffs[hi_key]
            hi = np.asarray(hi_arr, dtype=np.float32)

        lo = None
        if lo_key is not None and lo_key in coeffs:
            lo = np.asarray(coeffs[lo_key], dtype=np.float32)

        # build oriented bands in [scale][ori] order
        # pyrtools says coefficients run from fine to coarse. :contentReference[oaicite:3]{index=3}
        bands_out: List[List[np.ndarray]] = []
        for s, lvl in enumerate(levels):
            if s >= self.spec.n_scales:
                break
            row: List[np.ndarray] = []
            for o, b in enumerate(bands):
                if o >= self.spec.n_orientations:
                    break
                full_key = key_map.get((lvl, b), None)
                if full_key is None or full_key not in coeffs:
                    raise RuntimeError(f"Missing pyramid coefficient for (level={lvl}, band={b})")
                row.append(np.asarray(coeffs[full_key], dtype=np.complex64))
            bands_out.append(row)

        if lo is None:
            # In some configs height=0 yields only residuals; handle gracefully.
            lo = np.zeros_like(x, dtype=np.float32)

        return {
            "hi": hi,
            "bands": bands_out,
            "lo": lo,
            "_pt_pyr_params": {  # used for inverse reconstruction
                "height": int(self.spec.n_scales),
                "order": int(self.order),
                "twidth": int(self.spec.twidth),
            },
            "_pt_keys": {
                "hi_key": hi_key,
                "lo_key": lo_key,
                "levels": levels[: int(self.spec.n_scales)],
                "bands": bands[: int(self.spec.n_orientations)],
                "key_map": key_map,
            },
        }

    def inverse(self, coeffs: Dict[str, Any]) -> np.ndarray:
        pt = _require_pyrtools()

        # Recreate a pyramid object and overwrite coefficients, then recon_pyr (exact recon). :contentReference[oaicite:4]{index=4}
        params = coeffs.get("_pt_pyr_params", None)
        keys = coeffs.get("_pt_keys", None)
        if params is None or keys is None:
            raise ValueError("Missing pyrtools metadata (_pt_pyr_params/_pt_keys) in coeffs dict.")

        dummy = np.zeros((self.H, self.W), dtype=np.float32)
        pyr = pt.pyramids.SteerablePyramidFreq(
            dummy,
            height=int(params["height"]),
            order=int(params["order"]),
            twidth=int(params["twidth"]),
            is_complex=True,
        )

        new_coeffs = dict(pyr.pyr_coeffs)  # start with correct key structure

        # residuals
        hi_key = keys.get("hi_key", None)
        lo_key = keys.get("lo_key", None)

        if hi_key is not None and "hi" in coeffs:
            new_coeffs[hi_key] = np.asarray(coeffs["hi"], dtype=np.float32)

        if lo_key is not None and "lo" in coeffs:
            new_coeffs[lo_key] = np.asarray(coeffs["lo"], dtype=np.float32)

        # oriented bands
        levels: List[int] = list(keys["levels"])
        bands_idx: List[int] = list(keys["bands"])
        bands_in: List[List[np.ndarray]] = coeffs["bands"]
        key_map = keys.get("key_map", {})

        for s, lvl in enumerate(levels):
            for o, b in enumerate(bands_idx):
                full_key = key_map.get((lvl, b), (lvl, b))
                new_coeffs[full_key] = np.asarray(bands_in[s][o], dtype=np.complex64)


        pyr.pyr_coeffs = new_coeffs
        recon = pyr.recon_pyr()
        return np.asarray(recon, dtype=np.float32)
