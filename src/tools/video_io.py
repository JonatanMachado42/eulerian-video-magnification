from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    size: Tuple[int, int]  # (width, height)
    frame_count: Optional[int]
    codec: Optional[str] = None


def _try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


def _try_import_imageio():
    try:
        import imageio  # type: ignore
        import imageio.v3 as iio  # type: ignore
        return imageio, iio
    except Exception:
        return None, None


def get_video_info(video_path: Path) -> VideoInfo:
    cv2 = _try_import_cv2()
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return VideoInfo(fps=fps if fps > 0 else 30.0, size=(w, h), frame_count=n if n > 0 else None)

    imageio, iio = _try_import_imageio()
    if iio is None:
        raise RuntimeError(
            "No video backend found. Install one of:\n"
            "- opencv-python\n"
            "- imageio + imageio-ffmpeg\n"
        )
    # imageio v3 doesn't always expose fps reliably; best-effort.
    meta = {}
    try:
        meta = iio.immeta(str(video_path))
    except Exception:
        meta = {}
    fps = float(meta.get("fps") or 30.0)
    size = meta.get("size")
    if isinstance(size, (tuple, list)) and len(size) == 2:
        w, h = int(size[0]), int(size[1])
    else:
        # fallback: read first frame
        it = iio.imiter(str(video_path))
        first = next(iter(it))
        h, w = first.shape[:2]
    return VideoInfo(fps=fps, size=(w, h), frame_count=meta.get("nframes"))


def iter_frames(video_path: Path, as_rgb: bool = True) -> Iterator[np.ndarray]:
    """
    Yields frames as uint8 arrays.
    - If using cv2, reads BGR and converts to RGB if as_rgb=True.
    - If using imageio, usually yields RGB already (but we keep it consistent).
    """
    cv2 = _try_import_cv2()
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if as_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame
        finally:
            cap.release()
        return

    imageio, iio = _try_import_imageio()
    if iio is None:
        raise RuntimeError(
            "No video backend found. Install one of:\n"
            "- opencv-python\n"
            "- imageio + imageio-ffmpeg\n"
        )
    for frame in iio.imiter(str(video_path)):
        # Ensure uint8
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        yield frame


def write_video(
    out_path: Path,
    frames: Iterator[np.ndarray],
    fps: float,
    input_is_rgb: bool = True,
) -> None:
    """
    Writes frames to out_path as MP4.
    Tries cv2 first; if cv2 fails (codec/timebase/etc.), falls back to imageio.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # sanitize fps (avoid weird/zero values)
    try:
        fps = float(fps)
    except Exception:
        fps = 30.0
    if not (fps > 0 and np.isfinite(fps)):
        fps = 30.0
    # clamp to reasonable range (helps some writers)
    fps = max(1.0, min(fps, 240.0))

    cv2 = _try_import_cv2()
    if cv2 is not None:
        frames = iter(frames)
        try:
            first = next(frames)
        except StopIteration:
            raise ValueError("No frames to write.")

        h, w = first.shape[:2]

        def _to_bgr(f: np.ndarray) -> np.ndarray:
            if input_is_rgb:
                return cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            return f

        # Try a few common codecs; if all fail, fall back to imageio
        fourcc_candidates = ["mp4v", "avc1", "H264", "XVID"]
        writer = None
        for fourcc_str in fourcc_candidates:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            wtr = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
            if wtr.isOpened():
                writer = wtr
                break
            else:
                try:
                    wtr.release()
                except Exception:
                    pass

        if writer is not None:
            try:
                writer.write(_to_bgr(first))
                for f in frames:
                    writer.write(_to_bgr(f))
                return
            finally:
                try:
                    writer.release()
                except Exception:
                    pass
        # If cv2 path failed, continue to imageio fallback using first+rest frames
        def _rechain():
            yield first
            for f in frames:
                yield f
        frames = _rechain()

    # ---- imageio fallback (robust) ----
    imageio, iio = _try_import_imageio()
    if imageio is None:
        raise RuntimeError(
            "No video backend found. Install one of:\n"
            "- opencv-python\n"
            "- imageio + imageio-ffmpeg\n"
        )

    writer = imageio.get_writer(str(out_path), fps=float(fps))
    try:
        for f in frames:
            if f.dtype != np.uint8:
                f = np.clip(f, 0, 255).astype(np.uint8)
            writer.append_data(f)  # RGB is fine here
    finally:
        writer.close()
