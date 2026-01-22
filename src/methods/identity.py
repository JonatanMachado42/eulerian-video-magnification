from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.config import IdentityParams
from src.methods.base import MagnificationMethod, MethodResult
from src.registry import register_method
from src.tools.video_io import get_video_info, iter_frames, write_video


@register_method("identity")
class IdentityMethod(MagnificationMethod):
    @property
    def name(self) -> str:
        return "identity"

    def run(
        self,
        video_path: Path,
        out_root: Path,
        params: IdentityParams,
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
        out_video = out_dir / "output.mp4"
        meta_path = out_dir / "meta.json"

        # Stream frames input -> output
        frames = iter_frames(video_path, as_rgb=True)
        write_video(out_video, frames=frames, fps=info.fps, input_is_rgb=True)

        meta = {
            "method": self.name,
            "run_id": run_id,
            "input_path": str(video_path),
            "output_video_path": str(out_video),
            "video_info": {
                "fps": info.fps,
                "size": {"width": info.size[0], "height": info.size[1]},
                "frame_count": info.frame_count,
            },
            "params": self._params_to_dict(params),
            "save_interim": bool(save_interim),
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
