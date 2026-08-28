import numpy as np
import torch
from rbms.classes import EBM
from rbms.partition_function.ais import compute_partition_function_ais
from torch import Tensor

from ptt_paper.implement import _init_sampling, _ptt_sampling
from ptt_paper.pre_sampler import PreSampler


def ptt_sampling(
    list_params: list[EBM],
    chains: list[dict[str, Tensor]],
    index: Tensor | None,
    it_mcmc: int,
    pre_sampler: PreSampler | None = None,
    increment: int = 10,
    show_pbar: bool = True,
    show_acc_rate: bool = True,
) -> tuple[list[dict[str, Tensor]], Tensor, Tensor | None]:
    assert len(list_params) == len(chains), (
        f"list_params and chains must have the same length, but got {len(list_params)} and {len(chains)}"
    )
    chains, acc_rate, index = _ptt_sampling(
        list_params=list_params,
        chains=chains,
        it_mcmc=it_mcmc,
        increment=increment,
        pre_sampler=pre_sampler,
        show_pbar=show_pbar,
        show_acc_rate=show_acc_rate,
        index=index,
    )
    return chains, acc_rate, index


def init_sampling(
    n_gen: int,
    list_params: list[EBM],
    start_v: Tensor | None = None,
    it_mcmc: int = 1000,
    pre_sampler: PreSampler | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    show_pbar: bool = True,
):
    return _init_sampling(
        n_gen=n_gen,
        list_params=list_params,
        start_v=start_v,
        it_mcmc=it_mcmc,
        pre_sampler=pre_sampler,
        device=device,
        dtype=dtype,
        show_pbar=show_pbar,
    )


def compute_partition_function(
    list_params: list[EBM],
    list_chains: list[dict[str, Tensor]],
    log_z_init: float | None = None,
) -> Tensor:
    if log_z_init is None:
        # Estimate the first log Z using AIS
        # Should work if this distribution is not multimodal
        log_z_init = compute_partition_function_ais(
            num_chains=5000,
            num_beta=1000,
            params=list_params[0],
        )

    logz = log_z_init
    logZ = torch.zeros(len(list_params))
    logZ[0] = log_z_init
    for idx in range(len(list_params) - 1):
        E0 = list_params[idx].compute_energy_visibles(list_chains[idx]["visible"])
        E1 = list_params[idx + 1].compute_energy_visibles(list_chains[idx]["visible"])
        c0 = torch.logsumexp(-E1 + E0, dim=0) - np.log(
            list_chains[idx]["visible"].shape[0]
        )
        logz += c0
        logZ[idx + 1] = logz
    return logZ
