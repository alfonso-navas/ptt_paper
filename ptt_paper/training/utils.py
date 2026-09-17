import h5py
import numpy as np
import torch
from rbms.classes import EBM
from rbms.dataset.dataset_class import RBMDataset
from rbms.io import load_params, save_sampler
from rbms.map_model import map_model
from rbms.parser import default_args, set_args_default
from rbms.partition_function.ais import compute_partition_function_ais
from rbms.training.implement import _init_training
from rbms.utils import get_flagged_updates, get_saved_updates
from tqdm.autonotebook import tqdm

from ptt_paper.io import load_full_sampler_from_filename
from ptt_paper.map_sampler import map_sampler
from ptt_paper.pre_sampler.reservoir import Reservoir
from ptt_paper.sampler import PTT


def init_training_ptt(
    args: dict,
    train_dataset: RBMDataset,
    flags: list[str] = ["checkpoint"],
    map_model: dict[str, type[EBM]] = map_model,
) -> None:
    args = set_args_default(args, default_args=default_args)
    if args["reservoir_size"] is None:
        args["reservoir_size"] = 10 * args["num_chains"]
    _init_training(
        train_dataset=train_dataset,
        seed=args["seed"],
        train_size=args["train_size"],
        test_size=1 - args["train_size"],
        num_hiddens=args["num_hiddens"],
        num_chains=args["num_chains"],
        model_type=args["model_type"],
        filename=args["filename"],
        n_save=args["n_save"],
        spacing=args["spacing"],
        batch_size=args["batch_size"],
        optim=args["optim"],
        mult_optim=args["mult_optim"],
        training_type=args["training_type"],
        learning_rate=args["learning_rate"],
        max_lr=args["max_lr"],
        gibbs_steps=args["num_swaps"],
        beta=1.0,
        centered=not (args["no_center"]),
        L1=args["L1"],
        L2=args["L2"],
        effL2=args.get("effL2", 0.0) or 0.0,
        normalize_grad=args["normalize_grad"],
        max_norm_grad=args["max_norm_grad"],
        subset_labels=args["subset_labels"],
        use_weights=args["use_weights"],
        alphabet=args["alphabet"],
        remove_duplicates=args["remove_duplicates"],
        dtype=args["dtype"],
        device=args["device"],
        flags=flags,
        map_model=map_model,
        init_vbias=args["init_vbias"],
    )
    if args["model_type"] is None:
        match train_dataset.variable_type:
            case "bernoulli":
                args["model_type"] = "BBRBM"
            case "categorical":
                args["model_type"] = "PBRBM"
            case "ising":
                args["model_type"] = "IIRBM"
            case _:
                raise NotImplementedError()
    args["update"] = 1
    with h5py.File(args["filename"], "a") as f:
        ptt_args = f.create_group("ptt_args")
        ptt_args["num_swaps"] = args["num_swaps"]
        ptt_args["increment"] = args["increment"]
        ptt_args["reservoir_size"] = args["reservoir_size"]
        ptt_args["max_n_model"] = args["max_n_model"]
        ptt_args["target_n_model"] = args["target_n_model"]
        ptt_args["num_steps_annealing"] = args["num_steps_annealing"]
        ptt_args["n_sample_steps"] = args["n_sample_steps"]
        ptt_args["target_acc_rate"] = args["target_acc_rate"]
        ptt_args["full_sampler"] = args["full_sampler"]
        ptt_args["patience"] = args["patience"]

    params = load_params(args["filename"], 1, args["device"], args["dtype"])

    # Create Sampler
    list_model = [params, params.clone()]
    log_z_init = compute_partition_function_ais(1000, 5000, params)
    if args["model_type"] not in map_sampler.keys():
        args["model_type"] = "generic"
    sampler = PTT(
        list_model=list_model,
        num_chains=args["num_chains"],
        increment=args["increment"],
        num_swaps=args["num_swaps"],
        log_z_init=log_z_init,
        target_acc_rate=args["target_acc_rate"],
        max_n_model=args["max_n_model"],
        target_n_model=args["target_n_model"],
        full_sampler=args["full_sampler"],
        reservoir_size=args["reservoir_size"],
        n_sample_steps=args["n_sample_steps"],
        device=args["device"],
        dtype=args["dtype"],
    )
    sampler.init_annealing_chains(
        num_chains=args["num_chains"],
        num_steps=args["num_steps_annealing"],
    )
    pre_sampler = sampler.update_pre_sampler(sampler=None)
    # init reservoir
    print("Initializing Reservoir")
    sampler.set_pre_sampler(pre_sampler)
    save_sampler(sampler=sampler, filename=args["filename"], update=1)
    torch.cuda.empty_cache()


def get_good_acc_rate(sampler: PTT, ptt_updates: np.ndarray, max_n_steps: int = 20_000):
    while True:
        if len(sampler) == 1:
            break
        tau_int, tau_exp, _ = sampler.trwa(
            num_chains=100,
            force_recompute=True,
            plot=False,
            max_total_steps=max_n_steps,
        )
        sampler.index = None
        if tau_int > 0:  # Means that trwa converged
            idx_cut = torch.where(sampler.acc_rates < sampler.min_acc_rate)[0]
            if len(idx_cut) == 0:  # All acc_rates are valid
                print(f"GOOD ACC RATES: {sampler.acc_rates}")
                break
            print(f"BAD ACC RATES: {sampler.acc_rates}")
            idx_cut = int(idx_cut[0].item() + 1)
        else:  # trwa did not converge, we remove the last ptt model
            idx_cut = len(sampler) - 1
        print(f"IDX CUT: {idx_cut}")
        print(f"PTT UPDATES BEFORE CUT: {ptt_updates}")
        ptt_updates = ptt_updates[:idx_cut]
        print(f"PTT UPDATES AFTER CUT: {ptt_updates}")
        sampler.set_list_model([sampler.get_model(i) for i in range(idx_cut)])
        sampler.set_chains([sampler.get_chains(i) for i in range(idx_cut)])
    return sampler, ptt_updates


def check_acc_rate(sampler: PTT, max_n_steps: int = 20_000) -> bool:
    tau_int, tau_exp, _ = sampler.trwa(
        num_chains=20,
        force_recompute=True,
        plot=False,
        max_total_steps=max_n_steps,
    )
    sampler.index = None
    if tau_int > 0:
        print(torch.where(sampler.acc_rates < sampler.min_acc_rate)[0])
        return len(torch.where(sampler.acc_rates < sampler.min_acc_rate)[0]) == 0
    return False


def line_search_last_good_model(
    sampler: PTT,
    ptt_updates: np.ndarray,
    saved_updates: np.ndarray,
    filename: str,
    reservoir_size: int,
    full_sampler: bool,
    device,
    dtype,
) -> tuple[PTT, int]:
    saved_updates = saved_updates[saved_updates > ptt_updates[-1]]
    # We can only keep the last good model in PTT and try the ones after if we setup a Reservoir
    if not full_sampler:
        if len(sampler) > 1:
            reservoir = Reservoir(
                sampler.get_model(-1),
                sampler.sample_large(
                    reservoir_size, 10, 1, sampler._increment, -2, device, True
                )["visible"],
                True,
            )
        else:
            model = sampler.get_model(-1)
            chains = model.init_chains(reservoir_size)
            print(chains["visible"].shape)
            chains = model.sample_state(chains, 1)
            reservoir = Reservoir(model, chains["visible"], True)
    curr_sampler = sampler.clone()
    while True:
        if len(saved_updates) == 0:
            return sampler, ptt_updates[-1]
        print(f" Loading update {saved_updates[-1]}")
        params = load_params(filename, saved_updates[-1], device, dtype)
        if full_sampler:
            curr_sampler.set_list_model(
                [*[sampler.get_model(i) for i in range(len(sampler))], params]
            )
        else:
            curr_sampler.set_list_model([sampler.get_model(-1), params])
            curr_sampler.set_pre_sampler(reservoir)
        curr_sampler.index = None
        curr_sampler.init_random_chains(20)
        if check_acc_rate(curr_sampler, 20_000):
            return curr_sampler, saved_updates[-1]
        # Remove non valid model
        curr_sampler.pop_model(-1)
        saved_updates = saved_updates[:-1]


def remove_all_model_after_update_included(filename, update):
    saved_updates = get_saved_updates(filename)
    updates_to_remove = saved_updates[saved_updates >= update]

    pbar = tqdm(range(len(updates_to_remove)))
    for i in pbar:
        upd = updates_to_remove[i]
        pbar.set_description(f"Removing update {upd}")
        with h5py.File(filename, "a") as f:
            del f[f"update_{upd}"]


def reset_training(args) -> tuple[PTT, EBM, dict, int]:
    print("=================== RESET TRAINING ===================")

    print("---- Loading full sampler: ----")
    sampler = load_full_sampler_from_filename(
        args["filename"],
        args["num_chains"],
        False,
        args["device"],
        args["dtype"],
        None,
        args["increment"],
        1,
        map_model,
        map_sampler,
    )
    ptt_updates = get_flagged_updates(args["filename"], "ptt")
    assert len(ptt_updates) == len(sampler)

    print("---- Looking for a ladder with valid acceptance rates ----")
    # Here we isolate the last model with valid acceptance rate
    sampler, new_ptt_updates = get_good_acc_rate(sampler, ptt_updates, max_n_steps=20_000)
    # if a ptt update was removed, delete all subsequent updates from the archive
    if new_ptt_updates[-1] != ptt_updates[-1]:
        removed_updates = np.sort(ptt_updates[ptt_updates > new_ptt_updates[-1]])
        print(f"--- Removing updates after {removed_updates[0]} ----")
        remove_all_model_after_update_included(args["filename"], removed_updates[0])
    ptt_updates = new_ptt_updates
    sampler.index = None

    # Now we do a line search between the last saved model and the last valid model to find the last one
    # with decent acceptance rate
    print("---- Line search for last good model ----")
    saved_updates = get_saved_updates(args["filename"])
    print(f"PTT UPDATES: {ptt_updates}")
    print(f"SAVED UPDATES: {saved_updates}")
    if saved_updates[-1] != ptt_updates[-1]:
        sampler, last_good_update = line_search_last_good_model(
            sampler,
            ptt_updates,
            saved_updates,
            args["filename"],
            10 * args["num_chains"],
            True,
            args["device"],
            args["dtype"],
        )
    else:
        last_good_update = ptt_updates[-1]

    if last_good_update != saved_updates[-1]:
        print(f"--- Removing updates after {last_good_update} ----")
        remove_all_model_after_update_included(
            args["filename"], np.sort(saved_updates[saved_updates > last_good_update])[0]
        )

    sampler.index = None
    if len(sampler) == 1:
        sampler.set_list_model([sampler.get_model(0), sampler.get_model(0)])
    sampler.init_random_chains(args["num_chains"])
    sampler.sample(1000)

    print(f"---- Updating learning rate for update {last_good_update} ----")
    with h5py.File(args["filename"], "a") as f:
        new_lr = np.asarray(f[f"update_{last_good_update}"]["learning_rate"][()]) / 2
        args["learning_rate"] = torch.from_numpy(new_lr)
        f[f"update_{last_good_update}"]["learning_rate"][...] = new_lr
        if "ptt" not in f[f"update_{last_good_update}"]["flags"].keys():
            f[f"update_{last_good_update}"]["flags"]["ptt"] = True
        print(f"new learning rate for update {last_good_update}: {new_lr}")
    # init reservoir
    if sampler._pre_sampler is None:
        print("---- Initializing Reservoir ----")
        pre_sampler = sampler.update_pre_sampler(sampler=None)
        sampler.set_pre_sampler(pre_sampler)

    print("---- Reducing number of models ----")
    print(f" current number of models: {len(sampler)}")
    print(f" max number of models: {sampler.max_n_model}")
    sampler.reduce_number_model(sampler.max_n_model, sampler.reservoir_size, None)
    print(f" new number of models: {len(sampler)}")
    print(f" Current acceptance rates: {sampler.acc_rates}")
    params = sampler.get_model(-1)
    params.init_grad()

    ret_sampler = PTT(
        [sampler.get_model(i) for i in range(len(sampler))],
        sampler.num_chains,
        sampler._increment,
        sampler._num_swaps,
        sampler.target_acc_rate,
        sampler.max_n_model,
        sampler.target_n_model,
        False,
        sampler.reservoir_size,
        sampler.n_sample_steps,
        sampler._log_z_init,
        sampler.device,
        sampler.dtype,
    )
    ret_sampler.set_pre_sampler(sampler._pre_sampler)
    assert ret_sampler.full_sampler is False

    ret_sampler.set_chains([sampler.get_chains(i) for i in range(len(sampler))])
    return ret_sampler, params, args, last_good_update
