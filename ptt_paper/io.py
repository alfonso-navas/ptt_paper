import h5py
import numpy as np
import torch
from rbms.classes import EBM
from rbms.custom_fn import load_string
from rbms.io import load_params, save_model
from rbms.map_model import map_model
from rbms.utils import get_flagged_updates, get_saved_updates
from torch import Tensor

from ptt_paper.map_sampler import map_sampler
from ptt_paper.pre_sampler import map_pre_sampler
from ptt_paper.pre_sampler.classes import PreSampler
from ptt_paper.sampler import PTT


def load_pre_sampler(
    filename: str,
    ebm: EBM,
    device: torch.device | str,
    dtype: torch.dtype,
) -> PreSampler | None:
    named_params: dict[str, np.ndarray] = {}
    with h5py.File(filename, "r") as f:
        if "sampler" in f.keys():
            for k in f["sampler"].keys():
                named_params[k] = np.asarray(f["sampler"][k][()])
    pre_sampler_type = load_string(f["sampler"], k)
    # pre_sampler_type = str(named_params.pop("pre_sampler_type").astype(str))
    if pre_sampler_type == "none":
        return None
    return map_pre_sampler[pre_sampler_type].set_named_parameters(
        ebm=ebm, named_params=named_params, device=device, dtype=dtype
    )


def load_full_sampler_from_filename(
    filename: str,
    num_chains: int,
    append_last_model: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    ptt_updates: np.ndarray | None,
    increment: int,
    num_steps: int,
    map_model: dict[str, type[EBM]] = map_model,
    map_sampler: dict[str, type[PTT]] = map_sampler,
) -> PTT:
    list_up = get_saved_updates(filename)
    if ptt_updates is None:
        ptt_updates = get_flagged_updates(filename, "ptt")

    assert len(ptt_updates) > 0, "No ptt updates found !"

    if ptt_updates[-1] != list_up[-1]:
        print(
            f"The last ptt update '{ptt_updates[-1]}' does not correspond to the last saved update '{list_up[-1]}'"
        )
        if append_last_model:
            ptt_updates = np.append(ptt_updates, list_up[-1:])
            print(
                f"Appending the last update to the list of updates : {list_up[-1].item()}"
            )

    print(f"Selected updates for PTT: {ptt_updates}")
    log_z_init = None
    with h5py.File(filename, "r") as f:
        model_type = f["model_type"][()].decode()
        if "log_z" in f[f"update_{ptt_updates[0]}"].keys():
            log_z_init = f[f"update_{ptt_updates[0]}"]["log_z"][()].item()
    if log_z_init is None:
        params = load_params(filename, ptt_updates[0], device, dtype)
        from rbms.partition_function.ais import compute_partition_function_ais

        log_z_init = compute_partition_function_ais(
            num_chains=1000, num_beta=5000, params=params
        )

    list_model = []
    for upd in ptt_updates:
        list_model.append(
            load_params(
                filename=filename,
                index=upd.item(),
                device=device,
                dtype=dtype,
                map_model=map_model,
            )
        )

    match model_type:
        case "PBRBM":
            pass
        case "BBRBM" | "IIRBM":
            model_type = "default"
            pass
        case _:
            model_type = "default"
    return map_sampler[model_type](
        list_model=list_model,
        num_chains=num_chains,
        increment=increment,
        num_swaps=num_steps,
        target_acc_rate=0.25,
        max_n_model=2,
        target_n_model=2,
        full_sampler=False,
        reservoir_size=10 * num_chains,
        n_sample_steps=100,
        log_z_init=log_z_init,
        device=device,
        dtype=dtype,
    )


def save_model_ptt(
    filename: str,
    params: EBM,
    chains: dict[str, Tensor],
    num_updates: int,
    time: float,
    log_z: float,
    learning_rate: float,
    flags: list[str] = [],
):
    save_model(
        filename=filename,
        params=params,
        chains=chains,
        num_updates=num_updates,
        learning_rate=torch.tensor(learning_rate),
        time=time,
        flags=flags,
    )
    with h5py.File(filename, "a") as f:
        f[f"update_{num_updates}"]["log_z"] = log_z
        # f[f"update_{num_updates}"]["learning_rate"] = learning_rate


def load_params_ptt(
    filename: str,
    map_model: dict[str, type[EBM]],
    device: torch.device,
    dtype: torch.dtype,
) -> list[EBM]:
    ptt_updates = get_flagged_updates(filename, "ptt")
    list_params = []
    for upd in ptt_updates:
        params = load_params(
            filename, index=upd, device=device, dtype=dtype, map_model=map_model
        )
        list_params.append(params)
    return list_params
