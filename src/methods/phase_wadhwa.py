from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

import numpy as np

from src.methods.base import MagnificationMethod, MethodResult
from src.registry import register_method
from src.tools.video_io import get_video_info, iter_frames, write_video
from src.tools.phase import unwrap_temporal, principal_value
from src.tools.temporal_filters import bandpass
from src.tools.steerable_pyramid import ComplexSteerablePyramid, PyramidSpec


@dataclass(frozen=True)
class PhaseWadhwaParams:
    # smoke-test knobs (PC pequeno)
    max_frames: Optional[int] = 60
    resize_factor: float = 0.5  # 1.0 = full-res

    # pyramid (paper-grade)
    n_scales: int = 3
    n_orientations: int = 4
    twidth: int = 1

    # temporal bandpass (Hz)
    fl: float = 0.5
    fh: float = 3.0
    filter_mode: str = "iir"  # "iir" or "ideal_fft"
    iir_order: int = 2
    iir_apply: str = "sosfiltfilt"  # offline best

    # magnification
    alpha: float = 8.0
    alpha_per_scale: Optional[Tuple[float, ...]] = None
    wrap_output_phase: bool = True


def _try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


def _resize_rgb_u8(frame: np.ndarray, factor: float) -> np.ndarray:
    if factor >= 0.999:
        return frame
    cv2 = _try_import_cv2()
    h, w = frame.shape[:2]
    nh = max(2, int(h * factor))
    nw = max(2, int(w * factor))
    if cv2 is not None:
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    step = max(1, int(round(1.0 / factor)))
    return frame[::step, ::step]


def _rgb_to_yiq(rgb01: np.ndarray) -> np.ndarray:
    M = np.array(
        [
            [0.299, 0.587, 0.114],
            [0.596, -0.274, -0.322],
            [0.211, -0.523, 0.312],
        ],
        dtype=np.float32,
    )
    return rgb01 @ M.T


def _yiq_to_rgb(yiq: np.ndarray) -> np.ndarray:
    Mi = np.array(
        [
            [1.0, 0.956, 0.621],
            [1.0, -0.272, -0.647],
            [1.0, -1.106, 1.703],
        ],
        dtype=np.float32,
    )
    return yiq @ Mi.T


@register_method("phase_wadhwa")
class PhaseWadhwaMethod(MagnificationMethod):
    @property
    def name(self) -> str:
        return "phase_wadhwa"

    def run(
        self,
        video_path: Path,
        out_root: Path,
        params: PhaseWadhwaParams,
        save_interim: bool = False,
        run_id: Optional[str] = None,
    ) -> MethodResult:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        run_id = run_id or self._default_run_id()
        out_dir = self._build_output_dir(out_root=out_root, video_path=video_path, run_id=run_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        info = get_video_info(video_path)
        fps = float(info.fps)

        # 1) Load frames (RGB uint8) -> resize -> float32 [0,1]
        frames_rgb: List[np.ndarray] = []
        for i, fr in enumerate(iter_frames(video_path, as_rgb=True)):
            fr = _resize_rgb_u8(fr, float(params.resize_factor))
            frames_rgb.append(fr)
            if params.max_frames is not None and (i + 1) >= params.max_frames:
                break
        if len(frames_rgb) < 2:
            raise RuntimeError("Need at least 2 frames.")

        rgb01 = np.stack(frames_rgb, axis=0).astype(np.float32) / 255.0  # [T,H,W,3]
        T, H, W, _ = rgb01.shape

        yiq = _rgb_to_yiq(rgb01)
        Y = yiq[..., 0]
        I = yiq[..., 1]
        Q = yiq[..., 2]

        # 2) Paper-grade complex steerable pyramid (Simoncelli / pyrtools)
        spec = PyramidSpec(
            n_scales=int(params.n_scales),
            n_orientations=int(params.n_orientations),
            twidth=int(params.twidth),
        )
        pyr = ComplexSteerablePyramid(height=H, width=W, spec=spec)

        # 3) Forward pyramid for all frames (store complex oriented bands)
        coeff0 = pyr.forward(Y[0])
        bands0 = coeff0["bands"]

        bands_by_scale_ori: List[List[np.ndarray]] = []
        for s in range(spec.n_scales):
            bands_by_scale_ori.append([])
            for o in range(spec.n_orientations):
                hs, ws = bands0[s][o].shape
                bands_by_scale_ori[s].append(np.zeros((T, hs, ws), dtype=np.complex64))

        hi_all = np.zeros((T, H, W), dtype=np.float32)
        lo_all: List[np.ndarray] = []

        for t in range(T):
            coeff = pyr.forward(Y[t])
            hi_all[t] = np.asarray(coeff["hi"], dtype=np.float32)
            for s in range(spec.n_scales):
                for o in range(spec.n_orientations):
                    bands_by_scale_ori[s][o][t] = coeff["bands"][s][o]
            lo_all.append(np.asarray(coeff["lo"], dtype=np.float32))

        # alpha per scale
        if params.alpha_per_scale is not None:
            if len(params.alpha_per_scale) != spec.n_scales:
                raise ValueError("alpha_per_scale must have length n_scales")
            alpha_scales = list(params.alpha_per_scale)
        else:
            alpha_scales = [float(params.alpha)] * spec.n_scales

        # 4) Phase pipeline (Wadhwa): unwrap(t), bandpass(t), amplify, reconstruct
        bands_mod: List[List[np.ndarray]] = []
        for s in range(spec.n_scales):
            bands_mod.append([])
            alpha_s = float(alpha_scales[s])

            for o in range(spec.n_orientations):
                B = bands_by_scale_ori[s][o]  # [T,hs,ws] complex
                amp = np.abs(B).astype(np.float32)
                ph = np.angle(B).astype(np.float32)

                # unwrap in time
                ph_u = unwrap_temporal(ph, axis=0, ref_index=0)

                # delta phase relative to first frame (stable)
                delta = ph_u - ph_u[0]

                # temporal bandpass on delta
                delta_f = bandpass(
                    delta,
                    fl=float(params.fl),
                    fh=float(params.fh),
                    fps=float(fps),
                    axis=0,
                    mode=params.filter_mode,
                    order=int(params.iir_order),
                    iir_apply=params.iir_apply,  # type: ignore[arg-type]
                )

                ph_amp = ph + alpha_s * delta_f
                if params.wrap_output_phase:
                    ph_amp = principal_value(ph_amp)

                B_mod = (amp * np.exp(1j * ph_amp)).astype(np.complex64)
                bands_mod[s].append(B_mod)

        # 5) Reconstruct each frame and write output
        out_frames_u8: List[np.ndarray] = []
        for t in range(T):
            coeff_t: Dict[str, Any] = coeff0.copy()  # carries pyrtools metadata keys
            coeff_t["hi"] = hi_all[t]
            coeff_t["lo"] = lo_all[t]
            coeff_t["bands"] = [[bands_mod[s][o][t] for o in range(spec.n_orientations)] for s in range(spec.n_scales)]

            Y_rec = pyr.inverse(coeff_t)

            yiq_t = np.stack([Y_rec, I[t], Q[t]], axis=-1).astype(np.float32)
            rgb_rec = _yiq_to_rgb(yiq_t)
            rgb_rec = np.clip(rgb_rec, 0.0, 1.0)
            out_frames_u8.append((rgb_rec * 255.0).astype(np.uint8))

        out_video = out_dir / "output.mp4"
        meta_path = out_dir / "meta.json"
        write_video(out_video, frames=iter(out_frames_u8), fps=fps, input_is_rgb=True)

        meta = {
            "method": self.name,
            "run_id": run_id,
            "input_path": str(video_path),
            "output_video_path": str(out_video),
            "video_info": {
                "fps": fps,
                "size": {"width": int(W), "height": int(H)},
                "frame_count_processed": int(T),
            },
            "params": self._params_to_dict(params),
            "save_interim": bool(save_interim),
            "notes": "Paper-grade steerable pyramid via pyrtools (Simoncelli-style), phase-based Wadhwa pipeline.",
        }
        self._write_meta(meta_path, meta)

        return MethodResult(
            method_name=self.name,
            input_path=str(video_path),
            output_dir=str(out_dir),
            output_video_path=str(out_video),
            meta_path=str(meta_path),
            run_id=run_id,
            extra={},
        )
