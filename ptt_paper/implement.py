import matplotlib.pyplot as plt
import numpy as np
import torch
from rbms.classes import EBM
from rbms.custom_fn import clone_dict
from scipy.optimize import curve_fit
from torch import Tensor
from tqdm.autonotebook import tqdm

from ptt_paper.custom_fn import swap_chains
from ptt_paper.pre_sampler import PreSampler


def _init_sampling(
    n_gen: int,
    list_params: list[EBM],
    start_v: Tensor | None = None,
    it_mcmc: int = 1000,
    pre_sampler: PreSampler | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    show_pbar: bool = True,
) -> list[dict[str, Tensor]]:
    all_chains = []
    if show_pbar:
        pbar = tqdm(total=len(list_params), leave=False)
        pbar.set_description("Initializing PTT chains")

    init_v = list_params[0].init_chains(n_gen)["visible"]
    # init_v = torch.bernoulli(
    #     torch.ones(n_gen, list_params[0].vbias.shape[0], device=device, dtype=dtype)
    # )

    # Initialize the starting chains with the pre_sampler configurations
    if (pre_sampler is not None) and not (start_v is not None):
        pre_sampler.sample(n_gen)
        # Swap all configurations
        init_v = pre_sampler.perform_swap(
            visible_conf=init_v,
            swap_mask=torch.ones(n_gen, dtype=torch.bool, device=init_v.device),
        )

    # Start every model from random permutations of the input dataset
    if start_v is not None:
        perm_index = torch.randperm(start_v.shape[0])
        init_v = start_v[perm_index][:n_gen]

    for i, params in enumerate(list_params):
        chains = params.init_chains(num_samples=n_gen, start_v=init_v)

        # Iterate over the chains for some time
        chains = params.sample_state(n_steps=it_mcmc, chains=chains)

        init_v = chains["visible"]

        all_chains.append(clone_dict(chains))
        if show_pbar:
            pbar.update(1)
    return all_chains


def _swap_config_multi(
    params: list[EBM],
    chains: list[dict[str, Tensor]],
    index: Tensor | None = None,
    perform_swap: bool = True,
) -> tuple[list[dict[str, Tensor]], Tensor, Tensor | None]:
    n_chains, L = chains[0]["visible"].shape
    n_rbms = len(params)
    acc_rate = torch.zeros(n_rbms - 1, device=chains[0]["visible"].device)
    for idx in range(n_rbms - 1):
        delta_energy = (
            -params[idx].compute_energy_visibles(chains[idx + 1]["visible"])
            + params[idx].compute_energy_visibles(chains[idx]["visible"])
            + params[idx + 1].compute_energy_visibles(chains[idx + 1]["visible"])
            - params[idx + 1].compute_energy_visibles(chains[idx]["visible"])
        )

        swap = torch.exp(delta_energy) > torch.rand(
            size=(n_chains,), device=delta_energy.device
        )

        if index is not None:
            swapped_index_0 = torch.where(swap, index[idx + 1], index[idx])
            swapped_index_1 = torch.where(swap, index[idx], index[idx + 1])
            index[idx] = swapped_index_0
            index[idx + 1] = swapped_index_1

        acc_rate[idx] = swap.sum() / n_chains
        if perform_swap:
            chains[idx], chains[idx + 1] = swap_chains(chains[idx], chains[idx + 1], swap)
            perm_1 = torch.randperm(chains[idx]["weights"].shape[0])
            perm_2 = torch.randperm(chains[idx]["weights"].shape[0])
            chains[idx]["visible"] = chains[idx]["visible"][perm_1]
            chains[idx + 1]["visible"] = chains[idx + 1]["visible"][perm_2]
            if index is not None:
                index[idx] = index[idx][perm_1]
                index[idx + 1] = index[idx + 1][perm_2]

    return chains, acc_rate, index


def _sampling_step(
    list_params: list[EBM],
    chains: list[dict[str, Tensor]],
    pre_sampler: PreSampler | None,
    it_mcmc: int,
) -> list[dict[str, Tensor]]:
    """Performs it_mcmc sampling steps with all the models.

    Args:
        list_params list[RBM]: Saved models parameters.
        chains (list[Chain]): Previous configuration of the chains.
        it_mcmc (int): Number of steps to perform.

    Returns:
        list[Chain]: Updated chains.
    """
    # Sample from rbm
    n_chains = chains[0]["visible"].shape[0]
    for idx, params in enumerate(list_params):
        chains[idx] = params.sample_state(chains=chains[idx], n_steps=it_mcmc)
    if pre_sampler is not None:
        pre_sampler.sample(num_samples=n_chains)
        swap_mask = pre_sampler.compute_swap_acc(visible_conf=chains[0]["visible"])
        # print(swap_mask.sum())
        chains[0]["visible"] = pre_sampler.perform_swap(
            chains[0]["visible"], swap_mask=swap_mask
        )
    return chains


def _ptt_sampling(
    list_params: list[EBM],
    chains: list[dict[str, Tensor]],
    it_mcmc: int,
    increment: int = 10,
    pre_sampler: PreSampler | None = None,
    show_pbar: bool = True,
    show_acc_rate: bool = True,
    perform_swap: bool = True,
    index: Tensor | None = None,
) -> tuple[list[dict[str, Tensor]], Tensor, Tensor | None]:
    if show_pbar:
        pbar = tqdm(total=it_mcmc, leave=False)
    for steps in range(0, it_mcmc):
        if show_pbar:
            pbar.update(1)
        chains, acc_rate, index = _swap_config_multi(
            chains=chains, params=list_params, index=index, perform_swap=perform_swap
        )
        chains = _sampling_step(
            list_params=list_params,
            chains=chains,
            pre_sampler=pre_sampler,
            it_mcmc=increment,
        )
    if show_pbar:
        pbar.close()
    if show_acc_rate:
        print("acc_rate: ", acc_rate)
    return chains, acc_rate, index


def exponential_decay(t, C0, tauexp):
    return C0 * np.exp(-t / tauexp)


@torch.jit.script
def _tau_int(C: torch.Tensor) -> float:
    C[0] = 0.5
    t = torch.arange(C.shape[0], device=C.device)
    t_int = torch.cumsum(C, 0)
    t = torch.where(t >= 6 * t_int)[0]
    if len(t) == 0:
        t = -torch.ones(1, dtype=torch.long, device=C.device)
    return t_int[t[0]].item()


def _C_ftt(x):
    n = x.shape[0]

    # Step 1: Subtract mean across time dimension for each sample
    x = x

    # Step 2: Compute the FFT for each sample along the time dimension
    n_fft = 2 ** (int(np.log2(n)) + 1)  # Padding to next power of 2 for efficiency
    x_fft = torch.fft.fft(x, n_fft, dim=0)

    # Step 3: Compute the power spectrum for each sample
    power_spectrum = torch.abs(x_fft) ** 2

    # Step 4: Compute the inverse FFT for each sample to get the autocorrelation
    autocorr = torch.fft.ifft(power_spectrum, dim=0).real

    c = autocorr[: n // 2, :].mean(1)
    c = c / c[0]

    return c


def _process_experiment(swaps: Tensor, n_therm: int = 0, plot: bool = True):
    """
    swaps: Tensor of shape (n_steps, n_models, n_chains) with the index saved at every sampling steps.
    n_therm: int = 0 number of steps of thermalization which will be discarded
    """
    device = swaps.device
    num_samples = swaps.shape[2]
    num_models = swaps.shape[1]
    num_steps = swaps.shape[0]
    swaps = swaps[n_therm:]

    DT = 1
    delta_t = 1
    sw = swaps.reshape(num_steps - n_therm, -1)
    x = sw - (num_models - 1) / 2 * torch.ones(sw.shape, device=device)

    if len(x) == 0:
        print("Problem with")

    C = _C_ftt(x)

    ids = torch.isnan(C)
    C = C[~ids]
    times = torch.arange(len(C), device=device)[~ids]

    ids = torch.where(C < 0)[0]

    if len(ids) > 0:
        max_C = ids[0].cpu().item()
    else:
        max_C = len(C) // 2

    i0 = int(max_C / 3)

    xt = torch.arange(len(C), device=device)[~ids].float()

    tau_int = _tau_int(C)

    # print(i0, max_C * 2, C[i0].cpu())
    my_x = np.array(times[i0 : max_C * 2].cpu())
    my_y = np.array(C[i0 : max_C * 2].cpu())

    tau_optimized = max_C / 20
    c0_optimized = C[i0].cpu()

    popt, pcov = curve_fit(
        exponential_decay, my_x, my_y, p0=[c0_optimized, tau_optimized]
    )
    c0_final, tau_exp = popt

    C_all = C.cpu()
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=100)

        ax.plot(
            xt.cpu() * delta_t * DT,
            exponential_decay(xt.cpu(), popt[0], tau_exp),
            "--",
            color="grey",
            label=r"fit $\tau_\mathrm{exp}$",
        )

        x = np.arange(len(C)) * DT
        y = C_all

        ax.plot(x, y, color="C0")

        ax.set_xlabel("MCMC steps")
        ax.set_ylabel(r"$C(t)$")
        ax.set_yscale("log")

        ax.set_ylim(1e-5, 1)
        ax.set_xlim(0, max_C)

        fig.legend(loc="upper right")
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()

        plt.show()
    return tau_int, tau_exp, C
