from __future__ import annotations

from typing import Callable, Dict, List, Type
from src.methods.base import MagnificationMethod

_METHODS: Dict[str, Type[MagnificationMethod]] = {}


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
    return sorted(_METHODS.keys())


def get_method(name: str) -> Type[MagnificationMethod]:
    key = name.strip().lower()
    if key not in _METHODS:
        available = ", ".join(list_methods()) or "(none)"
        raise KeyError(f"Unknown method '{key}'. Available: {available}")
    return _METHODS[key]


# Import methods here so they register themselves via decorator.
# Keep this at bottom to avoid circular imports.
from src.methods.identity import IdentityMethod  # noqa: E402,F401
