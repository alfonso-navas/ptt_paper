import torch
from torch import Tensor


def sample_rcm_bernoulli(
    p_m: Tensor,
    mu: Tensor,
    U: Tensor,
    num_samples: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    num_visibles = U.shape[1]
    cdf = torch.cumsum(p_m, 0)
    x = torch.rand(num_samples, device=device, dtype=dtype)
    idx = torch.searchsorted(sorted_sequence=cdf, input=x) - 1
    mu_full = (mu[idx] @ U) * num_visibles**0.5  # n_samples x Nv
    x = torch.rand((num_samples, num_visibles), device=device, dtype=dtype)
    p = 1 / (1 + torch.exp(-2 * mu_full))  # n_samples x Nv
    # We want {0,1} samples
    s_gen = (x < p).to(dtype)
    return s_gen


def sample_rcm_potts(
    p_m: Tensor,
    mu: Tensor,
    U: Tensor,
    num_samples: int,
    num_colors: int,
    device: torch.device | str,
    dtype: torch.dtype,
):
    num_visibles = U.shape[1]
    num_sites = num_visibles // num_colors
    num_points = mu.shape[0]
    cdf = torch.zeros(num_points + 1, device=device, dtype=dtype)
    cdf[1:] = torch.cumsum(p_m, 0)
    x = torch.rand(num_samples, device=device, dtype=dtype)
    idx = torch.searchsorted(sorted_sequence=cdf, input=x) - 1
    idx = torch.min(idx, torch.ones_like(idx) * (mu.shape[0] - 1))
    mu_full = (mu[idx] @ U) * num_visibles**0.5  # n_samples x Nv
    p = torch.nn.functional.softmax(2 * mu_full.reshape(-1, num_colors), dim=-1)
    s_gen = torch.multinomial(p.reshape(-1, num_colors), 1).reshape(
        num_samples, num_sites
    )
    return s_gen
