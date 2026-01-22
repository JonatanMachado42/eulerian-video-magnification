from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json
import time


@dataclass(frozen=True)
class MethodResult:
    method_name: str
    input_path: str
    output_dir: str
    output_video_path: str
    meta_path: str
    run_id: str
    extra: Dict[str, Any]


class MagnificationMethod(ABC):
    """
    Base contract for any method (identity, Linear EVM, phase-based, learning-based...).
    The class should be thin: validate params, orchestrate tools, save outputs/meta.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique method name used by registry/CLI (e.g. 'identity', 'linear_evm')."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        video_path: Path,
        out_root: Path,
        params: Any,
        save_interim: bool = False,
        run_id: Optional[str] = None,
    ) -> MethodResult:
        raise NotImplementedError

    # ---------- helpers ----------
    def _default_run_id(self) -> str:
        # time-based id (human readable enough). You can switch to hash(params) later.
        return time.strftime("%Y%m%d-%H%M%S")

    def _build_output_dir(self, out_root: Path, video_path: Path, run_id: str) -> Path:
        video_stem = video_path.stem
        return out_root / self.name / video_stem / run_id

    def _write_meta(
        self,
        meta_path: Path,
        meta: Dict[str, Any],
    ) -> None:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _params_to_dict(self, params: Any) -> Dict[str, Any]:
        # Supports dataclasses or plain dict-like params
        if params is None:
            return {}
        if hasattr(params, "__dataclass_fields__"):
            return asdict(params)  # type: ignore[arg-type]
        if isinstance(params, dict):
            return dict(params)
        # fallback: best effort
        return {"value": str(params)}
