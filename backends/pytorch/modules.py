from __future__ import annotations

import torch

try:
    from .torch_ext import sad_ops as _vops
except ImportError:  # Allow running from source without package context.
    from torch_ext import sad_ops as _vops  # type: ignore


class SADRenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sites: torch.Tensor, cand0: torch.Tensor, cand1: torch.Tensor,
                inv_scale_sq: float, width: int, height: int) -> torch.Tensor:
        device_type = sites.device.type
        if device_type not in ("mps", "cuda"):
            raise RuntimeError("sites must be on MPS or CUDA")
        if sites.dtype != torch.float32:
            raise RuntimeError("sites must be float32")
        if cand0.dtype != torch.int32 or cand1.dtype != torch.int32:
            raise RuntimeError("candidates must be int32")
        if cand0.device != sites.device or cand1.device != sites.device:
            raise RuntimeError("candidates must be on the same device as sites")

        width = int(width)
        height = int(height)
        inv_scale_sq = float(inv_scale_sq)

        cand0 = cand0.contiguous()
        cand1 = cand1.contiguous()
        sites = sites.contiguous()

        out = _vops.render_sad(cand0, cand1, sites, inv_scale_sq, width, height)
        ctx.save_for_backward(sites, cand0, cand1)
        ctx.inv_scale_sq = inv_scale_sq
        ctx.width = width
        ctx.height = height
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        sites, cand0, cand1 = ctx.saved_tensors
        if grad_output.device != sites.device:
            raise RuntimeError("grad_output must be on the same device as sites")
        grad_output = grad_output.contiguous()
        grad_sites = _vops.render_sad_backward(
            cand0,
            cand1,
            sites,
            grad_output,
            float(ctx.inv_scale_sq),
            int(ctx.width),
            int(ctx.height),
        )
        return grad_sites, None, None, None, None, None


class SADRenderer(torch.nn.Module):
    def __init__(self, inv_scale_sq: float):
        super().__init__()
        self.inv_scale_sq = float(inv_scale_sq)

    def forward(self, sites: torch.Tensor, cand0: torch.Tensor, cand1: torch.Tensor,
                width: int, height: int, inv_scale_sq: float | None = None) -> torch.Tensor:
        inv_scale = self.inv_scale_sq if inv_scale_sq is None else float(inv_scale_sq)
        return SADRenderFunction.apply(sites, cand0, cand1, inv_scale, width, height)


class CandidateUpdater:
    def __init__(self, width: int, height: int, site_count: int, seed: int = 1,
                 device: torch.device | None = None):
        self.width = int(width)
        self.height = int(height)
        self.site_count = int(site_count)
        self.seed = int(seed)
        if device is not None and not isinstance(device, torch.device):
            device = torch.device(device)
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                raise RuntimeError("MPS/CUDA device required")
        self.device = device
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self.cand0, self.cand1 = _vops.init_candidates(self.width, self.height, self.site_count, self.seed)
        else:
            self.cand0, self.cand1 = _vops.init_candidates(self.width, self.height, self.site_count, self.seed)

    def reset(self, site_count: int | None = None, seed: int | None = None):
        if site_count is not None:
            self.site_count = int(site_count)
        if seed is not None:
            self.seed = int(seed)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self.cand0, self.cand1 = _vops.init_candidates(self.width, self.height, self.site_count, self.seed)
        else:
            self.cand0, self.cand1 = _vops.init_candidates(self.width, self.height, self.site_count, self.seed)
        return self.cand0, self.cand1

    @torch.no_grad()
    def update(self,
               sites: torch.Tensor,
               inv_scale_sq: float,
               jump: int,
               radius_probes: int = 1,
               radius_scale: float = 1.0,
               inject_count: int = 0,
               seed: int | None = None):
        if seed is None:
            seed = self.seed
        if sites.device != self.cand0.device:
            raise RuntimeError("sites must be on the same device as candidates")
        self.cand0, self.cand1 = _vops.vpt_pass(
            self.cand0,
            self.cand1,
            sites,
            float(inv_scale_sq),
            self.width,
            self.height,
            int(jump),
            int(radius_probes),
            float(radius_scale),
            int(inject_count),
            int(seed),
        )
        return self.cand0, self.cand1

    def get(self):
        return self.cand0, self.cand1


__all__ = [
    "SADRenderFunction",
    "SADRenderer",
    "CandidateUpdater",
]
