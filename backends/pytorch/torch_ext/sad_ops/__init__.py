import importlib
import torch

# Load the compiled extension (registers torch.ops.sad)
try:
    importlib.import_module("sad_ops._sad_ext")
except ModuleNotFoundError:
    try:
        from . import _sad_ext  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch SAD extension not found. Build it with:\n"
            "  python -m pip install -e backends/pytorch/torch_ext --no-build-isolation"
        ) from exc

ops = torch.ops.sad

render_sad = ops.render_sad
render_sad_padded = ops.render_sad_padded
render_sad_backward = ops.render_sad_backward
init_candidates = ops.init_candidates
vpt_pass = ops.vpt_pass

__all__ = [
    "ops",
    "render_sad",
    "render_sad_padded",
    "render_sad_backward",
    "init_candidates",
    "vpt_pass",
]
