"""PyTorch backend (Metal/CUDA-accelerated ops + training)."""

from .modules import CandidateUpdater, SADRenderFunction, SADRenderer

__all__ = [
    "CandidateUpdater",
    "SADRenderFunction",
    "SADRenderer",
]
