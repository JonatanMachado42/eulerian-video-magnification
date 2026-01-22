from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunConfig:
    """
    CLI/run-level config (not method-specific).
    """
    out_root: Path = Path("data/processed")
    save_interim: bool = False
    run_id: Optional[str] = None


# ---- Method-specific params ----

@dataclass(frozen=True)
class IdentityParams:
    """
    Identity method: passes video through unchanged.
    Kept for consistency with other methods.
    """
    # Example future knobs:
    # force_fps: Optional[float] = None
    pass
