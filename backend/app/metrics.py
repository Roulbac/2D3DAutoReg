"""Image similarity metrics for 2D/3D registration.

Provides both NumPy (for scipy optimizers) and PyTorch (for gradient-based
optimizers) implementations. All functions return negative values / loss values
suitable for minimization.
"""

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# NumPy metrics (for scipy.optimize.minimize)
# ---------------------------------------------------------------------------

def neg_normalized_cross_correlation(x, y, mask=slice(None)):
    u = x[mask] - x[mask].mean()
    v = y[mask] - y[mask].mean()
    denom = np.sqrt(np.sum(u**2)) * np.sqrt(np.sum(v**2))
    if denom == 0:
        return 0.0
    return -np.sum(u * v) / denom


def neg_gradient_corr(x, y, mask=slice(None)):
    dx1, dx2 = np.gradient(x[mask])
    dy1, dy2 = np.gradient(y[mask])
    return 0.5 * (
        neg_normalized_cross_correlation(dx1, dy1)
        + neg_normalized_cross_correlation(dx2, dy2)
    )


def mean_recipr_sqdiff(x, y, mask=slice(None)):
    x_min_y_sq = (x[mask] - y[mask]) ** 2
    return np.mean(x_min_y_sq / (1 + x_min_y_sq))


def neg_mutual_information(x, y, bins=32, mask=slice(None)):
    x, y = x[mask].flatten(), y[mask].flatten()
    m, M = min(x.min(), y.min()), max(x.max(), y.max())
    px, _ = np.histogram(x, bins=bins, range=(m, M), density=True)
    py, _ = np.histogram(y, bins=bins, range=(m, M), density=True)
    pxpy = np.outer(px, py)
    pxy, _, _ = np.histogram2d(
        x, y, bins=bins, range=((m, M), (m, M)), density=True
    )
    nzero_ids = pxy != 0
    return -np.sum(pxy[nzero_ids] * np.log(pxy[nzero_ids] / pxpy[nzero_ids]))


METRIC_REGISTRY = {
    "ncc": neg_normalized_cross_correlation,
    "gradient_corr": neg_gradient_corr,
    "mean_recipr_sqdiff": mean_recipr_sqdiff,
    "mutual_info": neg_mutual_information,
}

# ---------------------------------------------------------------------------
# PyTorch differentiable metrics (for gradient-based optimizers)
# ---------------------------------------------------------------------------

def torch_neg_ncc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Differentiable negative normalized cross-correlation."""
    u = x - x.mean()
    v = y - y.mean()
    denom = torch.sqrt(torch.sum(u ** 2)) * torch.sqrt(torch.sum(v ** 2))
    return -torch.sum(u * v) / denom.clamp(min=1e-8)


def torch_neg_gradient_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Differentiable negative gradient correlation.

    Uses conv2d with central-difference kernels to match np.gradient behaviour.
    """
    def _grad_2d(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img4d = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        ky = torch.tensor(
            [[-0.5], [0.0], [0.5]], device=img.device, dtype=img.dtype,
        ).reshape(1, 1, 3, 1)
        kx = torch.tensor(
            [[-0.5, 0.0, 0.5]], device=img.device, dtype=img.dtype,
        ).reshape(1, 1, 1, 3)
        gy = F.conv2d(img4d, ky, padding=(1, 0)).squeeze(0).squeeze(0)
        gx = F.conv2d(img4d, kx, padding=(0, 1)).squeeze(0).squeeze(0)
        return gy, gx

    dx1, dx2 = _grad_2d(x)
    dy1, dy2 = _grad_2d(y)
    return 0.5 * (torch_neg_ncc(dx1, dy1) + torch_neg_ncc(dx2, dy2))


def torch_mean_recipr_sqdiff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Differentiable mean reciprocal squared difference."""
    diff_sq = (x - y) ** 2
    return torch.mean(diff_sq / (1.0 + diff_sq))


TORCH_METRIC_REGISTRY = {
    "ncc": torch_neg_ncc,
    "gradient_corr": torch_neg_gradient_corr,
    "mean_recipr_sqdiff": torch_mean_recipr_sqdiff,
}
