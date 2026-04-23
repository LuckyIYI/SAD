import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules import CandidateUpdater, SADRenderer  # noqa: E402


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        raise RuntimeError("MPS/CUDA is not available")
    width = 64
    height = 64
    n_sites = 256
    inv_scale_sq = 1.0

    # Dummy target image
    target = torch.rand((height, width, 3), device=device, dtype=torch.float32)

    # Initialize sites (leaf tensor with gradients)
    sites = torch.zeros((n_sites, 10), device=device, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        sites[:, 0] = torch.rand((n_sites,), device=device) * (width - 1)
        sites[:, 1] = torch.rand((n_sites,), device=device) * (height - 1)
        sites[:, 2] = -2.0  # log_tau
        sites[:, 3] = 1.0   # radius
        sites[:, 4:7] = torch.rand((n_sites, 3), device=device)
        sites[:, 7] = 1.0   # aniso_dir.x
        sites[:, 8] = 0.0   # aniso_dir.y
        sites[:, 9] = 0.0   # log_aniso

    updater = CandidateUpdater(width, height, n_sites, seed=1, device=device)
    cand0, cand1 = updater.update(
        sites,
        inv_scale_sq=inv_scale_sq,
        jump=1,
        radius_probes=1,
        radius_scale=1.0,
        inject_count=0,
    )

    renderer = SADRenderer(inv_scale_sq)
    out = renderer(sites, cand0, cand1, width, height)
    loss = torch.mean((out - target) ** 2)
    loss.backward()

    grad_norm = sites.grad.norm().item() if sites.grad is not None else 0.0
    print(f"loss={loss.item():.6f} grad_norm={grad_norm:.6f}")


if __name__ == "__main__":
    main()
