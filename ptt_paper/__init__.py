import h5py
import numpy as np
import torch
from rbms.classes import EBM
from rbms.dataset import load_dataset
from rbms.io import load_params
from rbms.map_model import map_model
from rbms.utils import compute_log_likelihood, get_flagged_updates, get_saved_updates
from torch import Tensor

from ptt_paper.functional import ptt_sampling
from ptt_paper.io import load_full_sampler_from_filename, load_pre_sampler
from ptt_paper.sampler import PTT


def sample_from_file(
    filename: str,
    n_gen: int,
    device: torch.device | str,
    dtype: torch.dtype,
    n_steps: int,
    increment: int,
    n_steps_init: int,
    use_all_updates=False,
    optimal_acc=0.3,
    append_last_model: bool = True,
    map_model: dict[str, type[EBM]] = map_model,
    remove_high_acc=False,
) -> tuple[PTT, np.ndarray]:
    if use_all_updates:
        ptt_updates = get_saved_updates(filename)
    else:
        ptt_updates = get_flagged_updates(filename, "ptt")
        last_update = get_saved_updates(filename)[-1]
        first_update = get_saved_updates(filename)[0]
        if first_update not in ptt_updates:
            ptt_updates = np.append([first_update], ptt_updates)
        if last_update not in ptt_updates:
            ptt_updates = np.append(ptt_updates, [last_update])

    sampler = load_full_sampler_from_filename(
        filename=filename,
        num_chains=n_gen,
        append_last_model=append_last_model,
        increment=increment,
        num_steps=n_steps,
        device=device,
        dtype=dtype,
        ptt_updates=ptt_updates,
        map_model=map_model,
    )

    pre_sampler = None
    """pre_sampler = load_pre_sampler(
        filename=filename, device=device, dtype=dtype, ebm=sampler.get_model(0)
    )"""
    sampler.set_pre_sampler(pre_sampler)

    sampler.init_annealing_chains(num_chains=n_gen, num_steps=n_steps_init)
    sampler.sample(num_swaps=n_steps, show_pbar=True)

    if min(sampler.acc_rates) < optimal_acc - 0.2:
        print(
            f"The minimal acceptance rate {min(sampler.acc_rates)} is below the threshold {optimal_acc - 0.2}."
        )

        if len(ptt_updates) > 0:
            sampler, ptt_updates = find_good_times(
                filename,
                n_gen,
                device,
                dtype,
                n_steps,
                increment,
                n_steps_init,
                use_all_updates,
                optimal_acc,
                remove_high_acc=remove_high_acc,
                # map_model=map_model,
            )
            print("I ended find_good_times and we've got", len(sampler), "models")
        else:
            print(ptt_updates)

    return sampler, ptt_updates


def find_good_times(
    filename: str,
    n_gen: int,
    device: torch.device | str,
    dtype: torch.dtype,
    n_steps: int,
    increment: int,
    n_steps_init: int,
    use_all_updates=False,
    optimal_acc=0.3,
    remove_high_acc=True,
):
    min_acc = max(optimal_acc - 0.2, 0.1)
    max_acc = min(optimal_acc + 0.2, 0.9)
    # We start by reading the updates in the ptt list and in the all_updates
    ptt_updates = get_flagged_updates(filename, "ptt")
    all_updates = get_saved_updates(filename)
    last_update = get_saved_updates(filename)[-1]
    first_update = get_saved_updates(filename)[0]
    if first_update not in ptt_updates:
        ptt_updates = np.append([first_update], ptt_updates)
    if last_update not in ptt_updates:
        ptt_updates = np.append(ptt_updates, [last_update])
    sampler = load_full_sampler_from_filename(
        filename=filename,
        num_chains=n_gen,
        append_last_model=True,
        increment=increment,
        num_steps=100,
        device=device,
        dtype=dtype,
        ptt_updates=ptt_updates,
        map_model=map_model,
    )
    """pre_sampler = load_pre_sampler(
        filename=filename, device=device, dtype=dtype, ebm=sampler.get_model(0)
    )"""
    pre_sampler = None
    sampler.set_pre_sampler(pre_sampler)
    sampler.init_annealing_chains(num_chains=n_gen, num_steps=n_steps_init)
    sampler.sample(num_swaps=n_steps, show_pbar=True)

    if torch.min(sampler.acc_rates).item() < min_acc + 0.05 or (
        torch.max(sampler.acc_rates).item() > max_acc and remove_high_acc
    ):
        print("We start searching for a better ladder")
        # If some accuracies are below min_acc, we begin to add updates in between the previous ones
        num_ups = len(ptt_updates)
        num_ups_before = 0
        while torch.min(sampler.acc_rates).item() < min_acc and num_ups > num_ups_before:
            num_ups_before = num_ups
            indices = np.concatenate(
                [np.where(all_updates == up)[0] for up in ptt_updates]
            )
            insert = np.array((sampler.acc_rates < min_acc).cpu())

            mid = ((indices[:-1] + indices[1:]) * 0.5).astype(int)

            new_ids = np.unique(np.concatenate([indices, mid[insert]]))

            # print(new_ids)
            ptt_updates = all_updates[new_ids]
            sampler = load_full_sampler_from_filename(
                filename=filename,
                num_chains=n_gen,
                append_last_model=True,
                increment=increment,
                num_steps=1000,
                device=device,
                dtype=dtype,
                ptt_updates=ptt_updates,
                map_model=map_model,
            )
            """pre_sampler = load_pre_sampler(
                filename=filename, device=device, dtype=dtype, ebm=sampler.get_model(0)
            )"""
            pre_sampler = None
            sampler.set_pre_sampler(pre_sampler)
            sampler.init_annealing_chains(num_chains=n_gen, num_steps=n_steps_init)
            sampler.sample(num_swaps=n_steps, show_pbar=True)
            num_ups = len(ptt_updates)
            # print(sampler.acc_rates)
        if torch.min(sampler.acc_rates).item() < min_acc:
            print(
                "There are no enough updates in all_updates to keep the ptt acceptance within the margins!!!!!"
            )
            print(sampler.acc_rates)

        if remove_high_acc:
            # Now we remove the too high acceptance rates
            acc_rates = sampler.acc_rates
            already_checked = []
            cont = True

            while torch.max(acc_rates).item() > max_acc and cont:
                # print(already_checked)
                old_ptt_updates = ptt_updates.copy()
                old_acc = acc_rates.clone()
                indices = np.concatenate(
                    [np.where(all_updates == up)[0] for up in ptt_updates]
                )
                # print("Before removing we have ", len(indices))
                remove = (acc_rates > max_acc).cpu()
                # print(indices, remove)
                idx = torch.where(remove)[
                    0
                ]  # we take the first index in which a remove flag appear
                i = idx[0] if idx.numel() > 0 else None
                if i is None or indices[min(i + 1, len(indices) - 1)] in already_checked:
                    # if indices[i + 1] in already_checked or i == None:
                    cont = False  # we have already checked this one
                else:
                    cont = True

                    if i + 1 == len(
                        indices
                    ):  # This is the last update that we cannot remove
                        i -= 1
                    already_checked.append(indices[i + 1].item())
                    new_ids = np.concatenate([indices[: i + 1], indices[i + 2 :]])
                    print("Removing", indices[i + 1], new_ids, "!!!")
                    ptt_updates = all_updates[new_ids]
                    indices = np.concatenate(
                        [np.where(all_updates == up)[0] for up in ptt_updates]
                    )

                    sampler = load_full_sampler_from_filename(
                        filename=filename,
                        num_chains=n_gen,
                        append_last_model=True,
                        increment=increment,
                        num_steps=100,
                        device=device,
                        dtype=dtype,
                        ptt_updates=ptt_updates,
                        map_model=map_model,
                    )
                    """pre_sampler = load_pre_sampler(
                        filename=filename, device=device, dtype=dtype, ebm=sampler.get_model(0)
                    )"""
                    pre_sampler = None
                    sampler.set_pre_sampler(pre_sampler)
                    sampler.init_annealing_chains(
                        num_chains=n_gen, num_steps=n_steps_init
                    )
                    sampler.sample(num_swaps=n_steps, show_pbar=True)
                    acc_rates = sampler.acc_rates
                    print("After removing", sampler.acc_rates)
                    print("We have ", len(ptt_updates))
                    if torch.min(acc_rates).item() < min_acc:
                        print("We reverse the delete!!!!!!!!!!!!!!!")
                        ptt_updates = old_ptt_updates.copy()
                        acc_rates = old_acc.clone()
                    old_ptt_updates = ptt_updates.copy()
                    old_acc = acc_rates.clone()
        # we redo a final sampling with all the updates
        sampler = load_full_sampler_from_filename(
            filename=filename,
            num_chains=n_gen,
            append_last_model=True,
            increment=increment,
            num_steps=100,
            device=device,
            dtype=dtype,
            ptt_updates=ptt_updates,
            map_model=map_model,
        )
        """pre_sampler = load_pre_sampler(
            filename=filename, device=device, dtype=dtype, ebm=sampler.get_model(0)
        )"""
        pre_sampler = None
        sampler.set_pre_sampler(pre_sampler)
        sampler.init_annealing_chains(num_chains=n_gen, num_steps=n_steps_init)
        sampler.sample(num_swaps=n_steps, show_pbar=True)
        print("Final acceptance list", sampler.acc_rates)
        prev_all_updates = get_saved_updates(filename)
        with h5py.File(filename, "a") as f:
            for upd in prev_all_updates:
                if "ptt" in f[f"update_{upd}"]["flags"].keys():
                    del f[f"update_{upd}"]["flags"]["ptt"]
            for upd in ptt_updates:
                if "ptt" not in f[f"update_{upd}"]["flags"].keys():
                    f[f"update_{upd}"]["flags"]["ptt"] = True
    return sampler, ptt_updates


def retrieve_ll(
    filename,
    dataset_name,
    test_dataset_name=None,
    use_weights=False,
    alphabet="protein",
    device: torch.device | str = torch.device("cpu"),
    dtype=torch.float32,
    optimal_acc=0.3,
    recompute=False,
    use_all_updates=False,
    map_model: dict[str, type[EBM]] = map_model,
):
    train_dataset, test_dataset = load_dataset(
        dataset_name,
        test_dataset_name=test_dataset_name,
        device=device,
        alphabet=alphabet,
        use_weights=use_weights,
    )
    if test_dataset is None:
        with h5py.File(filename, "r") as f:
            seed = int(f["hyperparameters"]["seed"][()])
            train_size = float(f["hyperparameters"]["train_size"][()])
        rng = np.random.default_rng(seed)
        train_dataset, test_dataset = train_dataset.split_train_test(rng, train_size)

    ptt_updates = get_flagged_updates(filename=filename, flag="ptt")
    if len(ptt_updates) <= 1 or use_all_updates:
        ptt_updates = get_saved_updates(filename=filename)
    ptt_updates = get_saved_updates(filename=filename)

    compute_log_z = False
    with h5py.File(filename, "r") as f:
        if "log_z" not in f[f"update_{ptt_updates[-1]}"].keys():
            compute_log_z = True
    sampler = None
    if compute_log_z or recompute:
        sampler, ptt_updates = sample_from_file(
            filename=filename,
            n_gen=1000,
            device=device,
            dtype=dtype,
            n_steps=1000,
            increment=1,
            n_steps_init=1000,
            use_all_updates=use_all_updates,
            optimal_acc=optimal_acc,
            map_model=map_model,
        )
        log_z = sampler.compute_partition_function()
        print(log_z.shape)
        print(ptt_updates.shape)
        for i, upd in enumerate(ptt_updates):
            with h5py.File(filename, "a") as f:
                if "log_z" in f[f"update_{upd}"].keys():
                    del f[f"update_{upd}"]["log_z"]
                f[f"update_{upd}"]["log_z"] = log_z[i].item()
        ret_ptt_updates = ptt_updates

    else:
        log_z = []
        ret_ptt_updates = []
        with h5py.File(filename, "r") as f:
            for upd in ptt_updates:
                if "log_z" in f[f"update_{upd}"].keys():
                    ret_ptt_updates.append(upd)
                    log_z.append(f[f"update_{upd}"]["log_z"][()])

    visible_type = load_params(
        filename, ret_ptt_updates[0], device, dtype, map_model=map_model
    ).visible_type
    train_dataset.match_model_variable_type(visible_type)
    if test_dataset is not None:
        test_dataset.match_model_variable_type(visible_type)
    train_ll = []
    test_ll = []
    for i, upd in enumerate(ret_ptt_updates):
        params = load_params(filename, upd, device, dtype, map_model=map_model)
        train_ll.append(
            compute_log_likelihood(
                train_dataset.data.to(device),
                train_dataset.weights,
                params,
                log_z[i].item(),
            )
        )
        test_ll.append(
            compute_log_likelihood(
                test_dataset.data.to(device),
                test_dataset.weights,
                params,
                log_z[i].item(),
            )
        )
    return np.asarray(ret_ptt_updates), train_ll, test_ll, sampler
