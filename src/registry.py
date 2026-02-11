from __future__ import annotations

from typing import Callable, Dict, List, Type
from src.methods.base import MagnificationMethod

_METHODS: Dict[str, Type[MagnificationMethod]] = {}
_IMPORTED = False


def _ensure_registered() -> None:
    """
    Lazily import method modules so they can register themselves.
    This avoids circular imports (methods import registry to use @register_method).
    """
    global _IMPORTED
    if _IMPORTED:
        return

    # Import modules (not classes) so decorators run
    try:
        import src.methods.identity  # noqa: F401
    except Exception:
        pass

    try:
        import src.methods.phase_wadhwa  # noqa: F401
    except Exception:
        pass

    _IMPORTED = True


def register_method(name: str) -> Callable[[Type[MagnificationMethod]], Type[MagnificationMethod]]:
    """
    Decorator to register a method class by name.
    """
    def _decorator(cls: Type[MagnificationMethod]) -> Type[MagnificationMethod]:
        key = name.strip().lower()
        if key in _METHODS:
            raise ValueError(f"Method '{key}' already registered with {_METHODS[key]}")
        _METHODS[key] = cls
        return cls
    return _decorator


def list_methods() -> List[str]:
    _ensure_registered()
    return sorted(_METHODS.keys())


def get_method(name: str) -> Type[MagnificationMethod]:
    _ensure_registered()
    key = name.strip().lower()
    if key not in _METHODS:
        available = ", ".join(list_methods()) or "(none)"
        raise KeyError(f"Unknown method '{key}'. Available: {available}")
    return _METHODS[key]
