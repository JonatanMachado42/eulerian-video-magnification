from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RunConfig, IdentityParams  # noqa: E402
from src.registry import get_method, list_methods  # noqa: E402


def _collect_videos(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        # common video extensions
        exts = (".mp4", ".mov", ".mkv", ".avi")
        vids = []
        for p in input_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                vids.append(p)
        return sorted(vids)
    raise FileNotFoundError(f"Input path not found: {input_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Eulerian Video Magnification - Runner")
    parser.add_argument("--method", type=str, default="identity", help=f"Method name. Options: {', '.join(list_methods())}")
    parser.add_argument("--input", type=str, required=True, help="Path to a video file OR a directory containing videos")
    parser.add_argument("--out-root", type=str, default="data/processed", help="Output root directory")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id. Default: timestamp")
    parser.add_argument("--save-interim", action="store_true", help="Save intermediate artifacts when supported")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_root = Path(args.out_root)

    MethodCls = get_method(args.method)
    method = MethodCls()

    # For now we only have IdentityParams. Later we parse per-method params.
    params = IdentityParams()
    run_cfg = RunConfig(out_root=out_root, save_interim=bool(args.save_interim), run_id=args.run_id)

    videos = _collect_videos(input_path)
    if not videos:
        print("No videos found.", file=sys.stderr)
        return 2

    print(f"Method: {method.name}")
    print(f"Found {len(videos)} video(s). Output root: {run_cfg.out_root}")

    for vp in videos:
        print(f"\nRunning on: {vp}")
        result = method.run(
            video_path=vp,
            out_root=run_cfg.out_root,
            params=params,
            save_interim=run_cfg.save_interim,
            run_id=run_cfg.run_id,
        )
        print(f"  Output: {result.output_video_path}")
        print(f"  Meta:   {result.meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
