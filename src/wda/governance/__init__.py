"""Governance: seal, blinding and the re-run policy (sections 11 and 16). Submodules load lazily (PEP 562) so that python -m on a submodule does not re-enter it."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from wda.governance import blind, rerun_policy, seal

__all__ = ["blind", "rerun_policy", "seal"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
