"""ASTRA-E Inference Package."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astra.inference.realvideo import (
        ACTIONS,
        ASTRARealVideoModel,
        ASTRARealVideoNet,
        sample_frame_indices,
    )

__all__ = [
    "ACTIONS",
    "ASTRARealVideoModel",
    "ASTRARealVideoNet",
    "sample_frame_indices",
]


def __getattr__(name: str):
    if name in __all__:
        import astra.inference.realvideo as rv
        return getattr(rv, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
