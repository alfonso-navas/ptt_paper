from __future__ import annotations

import h5py
import numpy as np
import torch
from h5py import Group
from rbms import compute_log_likelihood
from rbms.classes import EBM, Sampler
from rbms.custom_fn import check_keys_dict, clone_dict
from rbms.partition_function.ais import compute_partition_function_ais
from torch import Tensor
from tqdm.autonotebook import tqdm

from ptt_paper.functional import (
    compute_partition_function,
)
from ptt_paper.implement import (
    _init_sampling,
    _process_experiment,
    _ptt_sampling,
)
from ptt_paper.pre_sampler import get_pre_sampler
from ptt_paper.pre_sampler.classes import PreSampler
from ptt_paper.pre_sampler.reservoir import Reservoir


class AcceptanceRateException(Exception):
    def __init__(self, message, errors):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


class PTT(Sampler):
    def __init__(
        self,
        list_model: list[EBM],
        num_chains: int,
        increment: int,
        num_swaps: int,
        target_acc_rate: float,
        max_n_model: int,
        target_n_model: int,
        full_sampler: bool,
        reservoir_size: int,
        n_sample_steps: int,
        log_z_init: float,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        self.device = device
        self.dtype = dtype
        self.set_list_model(list_model)
        self.set_increment(increment)
        self.set_num_swaps(num_swaps)
        self.set_pre_sampler(None)
        self._log_z_init = log_z_init
        self._hold_model = None
        self._hold_chains = None
        self._tmp_model = None
        self._tmp_chains = None
        self.acc_rates = torch.ones(len(self._list_model) - 1, device=device, dtype=dtype)
        self.init_random_chains(num_chains=num_chains)
        self.index = None
        self.log_z: Tensor | None = None
        self.target_acc_rate = target_acc_rate
        self.max_n_model = max_n_model
        self.target_n_model = target_n_model
        self.reservoir_size = reservoir_size
        self.n_sample_steps = n_sample_steps
        self.full_sampler = full_sampler
        self.name = "PTT"
        self.flags = []
        self._hold_model: EBM | None = None
        self._tmp_model: EBM | None = None
        self._hold_chains: dict[str, Tensor] | None = None
        self._tmp_chains: dict[str, Tensor] | None = None
        self.min_acc_rate = 0.1
        if self.full_sampler:
            self._full_sampler = PTT(
                list_model,
                num_chains,
                increment,
                num_swaps,
                target_acc_rate,
                max_n_model,
                target_n_model,
                False,
                reservoir_size,
                n_sample_steps,
                log_z_init,
                device,
                dtype,
            )

    def __len__(self):
        return len(self._list_model)

    @property
    def num_chains(self):
        return self._chains[0]["visible"].shape[0]

    def clone(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> PTT:
        named_params = self.named_parameters()
        return self.set_named_parameters(
            named_params,
            map_model={
                str(named_params["model_type"].astype("<U8")): type(self.get_model(0))
            },
            device=self.device,
            dtype=self.dtype,
        )

    # =================================== LOAD AND SAVE ===================================

    @torch.compiler.disable
    def named_parameters(self) -> dict[str, np.ndarray]:
        num_models = len(self)
        named_params: dict[str, np.ndarray] = {}
        named_params["sampler_type"] = np.asarray(self.name)

        if self._pre_sampler is None:
            named_params["pre_sampler_type"] = np.asarray("none", dtype="T")
        else:
            for k, v in self._pre_sampler.named_parameters().items():
                named_params[k] = v

        for i in range(num_models):
            named_params_model = self.get_model(i).named_parameters()
            for k, v in named_params_model.items():
                named_params[k + f"__{i}"] = v
        named_params["num_models"] = np.asarray(num_models)
        named_params["parallel_chains"] = self.get_chains(-1)["visible"].cpu().numpy()
        named_params["model_type"] = np.asarray(self.get_model(-1).name, dtype="T")
        named_params["increment"] = np.asarray(self._increment)
        named_params["num_swaps"] = np.asarray(self._num_swaps)
        named_params["log_z_init"] = np.asarray(self._log_z_init)
        named_params["sampler_type"] = np.asarray(self.name, dtype="T")
        named_params["num_chains"] = np.asarray(self.num_chains)

        named_params["target_acc_rate"] = np.asarray(self.target_acc_rate)
        named_params["max_n_model"] = np.asarray(self.max_n_model)
        named_params["target_n_model"] = np.asarray(self.target_n_model)
        named_params["reservoir_size"] = np.asarray(self.reservoir_size)
        named_params["n_sample_steps"] = np.asarray(self.n_sample_steps)
        named_params["full_sampler"] = np.asarray(self.full_sampler)
        return named_params

    @staticmethod
    def set_named_parameters(
        named_params: dict[str, np.ndarray],
        map_model: dict[str, type[EBM]],
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        names = [
            "num_models",
            # "parallels_chains",
            "model_type",
            "increment",
            "num_swaps",
            "log_z_init",
            "target_acc_rate",
            "max_n_model",
            "target_n_model",
            "reservoir_size",
            "n_sample_steps",
            "full_sampler",
        ]
        check_keys_dict(d=named_params, names=names)
        num_models = int(named_params.pop("num_models"))
        model_type = str(named_params.pop("model_type").astype("<U8"))
        increment = int(named_params.pop("increment"))
        num_swaps = int(named_params.pop("num_swaps"))
        log_z_init = float(named_params.pop("log_z_init"))
        num_chains = int(named_params.pop("num_chains"))
        target_acc_rate = float(named_params.pop("target_acc_rate"))
        max_n_model = int(named_params.pop("max_n_model"))
        target_n_model = int(named_params.pop("target_n_model"))
        reservoir_size = int(named_params.pop("reservoir_size"))
        n_sample_steps = int(named_params.pop("n_sample_steps"))
        full_sampler = bool(named_params.pop("full_sampler"))
        list_model: list[EBM] = []
        for i in range(num_models):
            keys_to_use: list[str] = []
            for k in named_params.keys():
                if "__" in k:
                    if int(k.split("__")[-1]) == i:
                        keys_to_use.append(k)
            named_params_model = {}
            for k in keys_to_use:
                named_params_model["_".join(k.split("__")[:-1])] = named_params.pop(k)
            list_model.append(
                map_model[model_type].set_named_parameters(
                    named_params=named_params_model, device=device, dtype=dtype
                )
            )
        pre_sampler = get_pre_sampler(
            named_params,
            ebm=list_model[0],
            device=device,
            dtype=dtype,
        )

        sampler = PTT(
            list_model=list_model,
            num_chains=num_chains,
            increment=increment,
            num_swaps=num_swaps,
            log_z_init=log_z_init,
            target_acc_rate=target_acc_rate,
            max_n_model=max_n_model,
            target_n_model=target_n_model,
            reservoir_size=reservoir_size,
            n_sample_steps=n_sample_steps,
            full_sampler=full_sampler,
            device=device,
            dtype=dtype,
        )
        sampler.set_pre_sampler(pre_sampler)
        return sampler

    @staticmethod
    def from_filename(
        filename: str,
        device: torch.device | str,
        dtype: torch.dtype,
        map_model: dict[str, type[EBM]],
    ):
        with h5py.File(filename, "r") as f:
            if "sampler" not in f.keys():
                raise ValueError(f"No PTT sampler found in {filename}")
            named_params: dict[str, np.ndarray] = {}
            sampler_archive: Group = f["sampler"]
            for k in sampler_archive.keys():
                named_params[k] = np.asarray(sampler_archive[k])
            print(named_params)
        return PTT.set_named_parameters(
            named_params=named_params, map_model=map_model, device=device, dtype=dtype
        )

    # =================================== TRAIN ===================================

    def get_conf_grad(self, batch) -> dict[str, Tensor]:
        self.sample()
        if self.acc_rates.min().cpu().item() < self.min_acc_rate:
            raise AcceptanceRateException(
                f"Acceptance rate too low: {self.acc_rates.cpu().numpy()}, min_acc_rate: {self.min_acc_rate}",
                self.acc_rates.cpu().numpy(),
            )
        return self.get_chains(-1)

    def post_grad_update(self, params):
        self.set_last_model(model=params)

    def pre_grad_update(self):
        self.update_replica_chain(
            sampler=None,
        )

    @torch.compiler.disable
    def get_metrics_display(self, metrics, **kwargs):
        train_batch = kwargs["train_dataset"].batch(self.num_chains)
        test_batch = kwargs["test_dataset"].batch(self.num_chains)
        log_z = self.compute_partition_function()[-1].item()
        params = self.get_model(-1)
        metrics["LL train"] = compute_log_likelihood(
            v_data=train_batch["data"],
            w_data=train_batch["weights"],
            params=params,
            log_z=log_z,
        )
        metrics["LL test"] = compute_log_likelihood(
            v_data=test_batch["data"],
            w_data=test_batch["weights"],
            params=params,
            log_z=log_z,
        )
        metrics["acc"] = " ".join([f"{elt.item():.2f}" for elt in self.acc_rates])
        return metrics

    def get_metrics_save(self):
        log_z = self.compute_partition_function()[-1]
        return {"log_z": log_z.cpu().numpy()}

    # =================================== SETTER ===================================

    def set_increment(self, increment: int):
        self._increment = increment

    def set_pre_sampler(self, pre_sampler: PreSampler | None):
        self._pre_sampler = pre_sampler

    def set_list_model(self, list_model: list[EBM]):
        self._list_model = [
            m.to(device=self.device, dtype=self.dtype) for m in list_model
        ]

    def set_num_swaps(self, num_swaps: int):
        self._num_swaps = num_swaps

    def set_last_model(self, model):
        self._list_model[-1] = model.to(device=self.device, dtype=self.dtype)

    # =================================== GETTER ===================================

    def get_chains(self, index: int):
        return self._chains[index]

    def get_curr_conf(self):
        return self.get_chains(-1)

    def set_chains(self, chains: list[dict[str, Tensor]]):
        assert len(chains) == len(self._list_model)
        self._chains = chains

    def get_model(self, index: int) -> EBM:
        """
        Returns the replica at a given index.

        Args:
            index (int): index of the replica

        Returns:
            EBM: The replica
        """
        return self._list_model[index]

    # =================================== INITIALIZATION ===================================

    def init_random_chains(self, num_chains: int):
        self._chains = []
        for m in self._list_model:
            self._chains.append(m.init_chains(num_samples=num_chains))

    def init_annealing_chains(self, num_chains, num_steps, start_v=None):
        self._chains = _init_sampling(
            n_gen=num_chains,
            list_params=self._list_model,
            start_v=start_v,
            it_mcmc=num_steps,
            pre_sampler=self._pre_sampler,
            device=self.device,
            dtype=self.dtype,
            show_pbar=True,
        )

    def init_index(self) -> Tensor:
        self.index = (
            torch.arange(len(self), device=self.device)
            .repeat_interleave(self.num_chains)
            .reshape(-1, self.num_chains)
        )

        return self.index

    # =================================== SAMPLING ===================================

    def sample(
        self,
        num_steps: int | None = None,
        **kwargs,
    ):
        """Samples the internal chains using PTT sampling.

        Args:
            num_swaps (int): Number of swap steps.
            increment (Optional[int]): Number of sampling steps between two swaps. Defaults to None.
        """
        if "increment" in kwargs.keys():
            increment = kwargs["increment"]
        else:
            increment = self._increment
        if "show_pbar" in kwargs.keys():
            show_pbar = kwargs["show_pbar"]
        else:
            show_pbar = False
        if "perform_swap" in kwargs.keys():
            perform_swap = kwargs["perform_swap"]
        else:
            perform_swap = True
        if "show_acc_rate" in kwargs.keys():
            show_acc_rate = kwargs["show_acc_rate"]
        else:
            show_acc_rate = False
        if num_steps is None:
            num_swaps = int(self._num_swaps * np.sqrt(len(self._list_model)))
        else:
            num_swaps = num_steps
        if increment is None:
            increment = self._increment
        self._chains, self.acc_rates, self.index = _ptt_sampling(
            list_params=self._list_model,
            chains=self._chains,
            it_mcmc=num_swaps,
            increment=increment,
            pre_sampler=self._pre_sampler,
            index=self.index,
            show_pbar=show_pbar,
            show_acc_rate=show_acc_rate,
            perform_swap=perform_swap,
        )

    def compute_partition_function(self) -> Tensor:
        """
        Compute the log partition function for all the replicas in use.

        Returns:
            Tensor: Log partition function
        Notes:
            If log_z_init was not provided when initializing the class it will be initialized here with AIS estimation.
        """
        if self._log_z_init is None:
            self._log_z_init = compute_partition_function_ais(
                self._chains[0]["visible"].shape[0], 5000, self._list_model[0]
            )
        self.log_z = compute_partition_function(
            list_params=self._list_model,
            list_chains=self._chains,
            log_z_init=self._log_z_init,
        )
        return self.log_z

    def get_ll(self, data: Tensor, w_data: Tensor) -> Tensor:
        if self.log_z is None:
            self.log_z = self.compute_partition_function()

        ll = torch.zeros(len(self))
        for i in range(len(self)):
            model = self.get_model(i)
            ll[i] = compute_log_likelihood(data, w_data, model, self.log_z[i].item())
        return ll

    def _hold_model_step(self):
        """Remove the temporary model from the replica chain and insert the hold model instead."""
        assert self._hold_model is not None
        assert self._hold_chains is not None

        self.pop_model(-2)
        self.insert_model(-2, self._hold_model, self._hold_chains)
        self._hold_model = None
        self._hold_chains = None

    def _tmp_model_step(self):
        """Insert a temporary model in the replica chain."""
        assert self._tmp_model is not None
        assert self._tmp_chains is not None
        self.insert_model(-2, self._tmp_model, self._tmp_chains)
        self._tmp_model = None
        self._tmp_chains = None

    def _store_hold_model(self):
        """Store a model as to use later as intermediate model."""
        self._hold_model = self._list_model[-1].clone()
        self._hold_chains = clone_dict(self.get_chains(-1))

    def _store_tmp_model(self):
        """Store a model as to use later as intermediate model."""
        self._tmp_model = self._list_model[-1].clone()
        self._tmp_chains = clone_dict(self.get_chains(-1))

    def reduce_number_model(
        self, n_models: int, reservoir_size: int, sampler: PTT | None
    ):
        self.log_z = self.compute_partition_function()
        self._log_z_init = self.log_z[-n_models].item()
        self.update_pre_sampler(
            sampler=sampler,
        )
        self.cut_sampler(-n_models)

    def update_pre_sampler(
        self,
        sampler: PTT | None,
        num_chains_init: int = 1000,
    ):
        if sampler is None:
            sampler = self
        reservoir_sampler = sampler.clone()
        reservoir_sampler.init_annealing_chains(
            num_chains_init,
            1,
            start_v=sampler.get_chains(-1)["visible"][:num_chains_init],
        )
        tau_int, tau_exp, _ = reservoir_sampler.trwa(
            100,
            increment=reservoir_sampler._increment,
            force_recompute=True,
            plot=False,
            max_total_steps=20_000,
        )
        if tau_int == -1:
            raise AcceptanceRateException(
                f"tau_int too large; acc rates: {self.acc_rates.cpu().numpy()}, min_acc_rate: {reservoir_sampler.min_acc_rate}",
                self.acc_rates.cpu().numpy(),
            )
        if (
            reservoir_sampler.acc_rates.min().cpu().item()
            < reservoir_sampler.min_acc_rate
        ):
            raise AcceptanceRateException(
                f"Acceptance rate too low: {self.acc_rates.cpu().numpy()}, min_acc_rate: {reservoir_sampler.min_acc_rate}",
                self.acc_rates.cpu().numpy(),
            )
        reservoir_sampler.index = None
        reservoir = reservoir_sampler.sample_large(
            num_samples=self.reservoir_size,
            idx_model=-self.target_n_model,
            num_steps_warmup=20 * int(tau_exp),
            num_steps_between=2 * int(tau_int),
            increment=reservoir_sampler._increment,
            out_device=reservoir_sampler.device,
            show_pbar=False,
        )["visible"]
        return Reservoir(
            ebm=sampler._list_model[-self.target_n_model], reservoir=reservoir, use=True
        )

    def update_replica_chain(
        self,
        sampler: PTT | None,
    ) -> None:
        """
        Update the replica chain according to the paper's algorithm.

        Args:
            target_acc_rate (float): The target acceptance rate between two replicas in the final chain.
            target_n_model (int): The target number of models when reducing the chain with a pre sampler.
            n_sample_steps (int): The number of sampling steps performed when modifying the chain to ensure equilibrium.
        """

        acc_rate = self.acc_rates[-1].to("cpu").item()

        if acc_rate > 2 * self.target_acc_rate:
            return
        if self._hold_model is None:
            if self._tmp_model is None and acc_rate < 2 * self.target_acc_rate:
                self._store_tmp_model()
            elif self._tmp_model is not None and acc_rate < self.target_acc_rate:
                self.flags.append("ptt")
                self._tmp_model_step()
                self.sample(self.n_sample_steps)
                self._store_hold_model()
        elif acc_rate < self.target_acc_rate:
            print(
                f"curr acc rates: {self.acc_rates}",
            )
            self._hold_model_step()
            self.sample(self.n_sample_steps)
            if self.full_sampler:
                self._full_sampler.insert_model(
                    -1, self.get_model(-1), self.get_chains(-1)
                )
            if len(self) > self.max_n_model and self._pre_sampler is not None:
                if sampler is None and self.full_sampler:
                    sampler = self._full_sampler
                self.reduce_number_model(
                    n_models=self.target_n_model,
                    reservoir_size=self.reservoir_size,
                    sampler=sampler,
                )

    def insert_model(self, i: int, model: EBM, chains: dict[str, Tensor]):
        """Insert a model in the PTT scheme at a given index

        Args:
            i (int): Index where to add the model.
            model (EBM): Model to add.
            chains (dict[str, Tensor]): Chains associated to the model.
        """
        if i == -1:
            self._list_model.append(model)
            self._chains.append(chains)
        if i < len(self._list_model):
            self._list_model.insert(i + 1, model)
            self._chains.insert(i + 1, chains)

    def pop_model(self, i: int):
        """Remove the replica at a given index

        Args:
            i (int): Index of the replica to remove
        """
        self._list_model.pop(i)
        self._chains.pop(i)

    def cut_sampler(self, index: int):
        self._list_model = self._list_model[index:]
        if self._chains is not None:
            self._chains = self._chains[index:]

    def sample_large(
        self,
        num_samples: int,
        num_steps_between: int,
        num_steps_warmup: int,
        increment: int,
        idx_model: int | None = None,
        out_device: torch.device | str = "cpu",
        show_pbar: bool = True,
    ):
        self.sample(num_steps=num_steps_warmup, increment=increment, show_pbar=True)
        output = dict()
        c = self.get_chains(-1)

        for k, v in c.items():
            shape = (
                (len(self), num_samples, *v.shape[1:])
                if idx_model is None
                else (num_samples, *v.shape[1:])
            )
            output[k] = torch.zeros(shape, device=out_device, dtype=v.dtype)
        idx_start = 0
        if show_pbar:
            n_pass = int(num_samples / self.num_chains) + int(
                num_samples / self.num_chains > 0
            )
            pbar = tqdm(total=n_pass, desc="Sample large", leave=False)
        while idx_start < num_samples:
            last_index = min(idx_start + v.shape[0], num_samples)
            if idx_model is None:
                for i in range(len(self)):
                    c = self.get_chains(i)
                    for k, v in c.items():
                        output[k][i, idx_start:last_index] = (
                            v[: (last_index - idx_start)].clone().to(device=out_device)
                        )
            else:
                c = self.get_chains(idx_model)
                for k, v in c.items():
                    output[k][idx_start:last_index] = (
                        v[: (last_index - idx_start)].clone().to(device=out_device)
                    )
            idx_start = last_index
            if show_pbar:
                pbar.update(1)
            if idx_start == num_samples:
                break
            self.sample(num_steps=num_steps_between, increment=increment, show_pbar=True)
        return output

    def trwa(
        self,
        num_chains: int | None = None,
        increment: int | None = None,
        num_swaps: int = 100,
        n_steps_therm: int = 1000,
        filename: str | None = None,
        force_recompute: bool = False,
        plot: bool = True,
        max_total_steps: int | None = None,
    ) -> tuple[float, float, np.ndarray]:
        compute = False
        if force_recompute:
            compute = True
        if filename is None:
            compute = True
        else:
            with h5py.File(filename, "r") as f:
                if "trwa" not in f.keys():
                    compute = True
        if compute:
            if increment is None:
                increment = self._increment
            if num_chains is not None and self.num_chains != num_chains:
                self.init_random_chains(num_chains)
            self.index_evolution = torch.zeros(
                num_swaps, len(self), self.num_chains, dtype=torch.int
            )
            # Thermalisation
            self.sample(num_steps=n_steps_therm, increment=increment, show_pbar=True)
            self.index = self.init_index()
            total_steps = 0
            start_idx = 0
            tau_exp = num_swaps // 2
            target = num_swaps
            while True:
                for i in tqdm(range(total_steps, target), leave=False):
                    self.sample(num_steps=1, increment=increment)
                    self.index_evolution[i] = self.index.cpu()
                    total_steps += 1
                tau_int, tau_exp, C = _process_experiment(
                    swaps=self.index_evolution,
                    plot=False,
                    # n_therm=self.index_evolution.shape[0] // 2,
                    n_therm=0,
                )
                print(
                    f"total_steps: {total_steps}; 20x tau_int: {20 * tau_int}, 20x tau_exp: {20 * tau_exp}"
                )
                target = max(int(20 * tau_int), int(20 * tau_exp))
                if total_steps >= target:
                    break
                num_swaps = target - total_steps
                print(f"New number of sampling steps: {num_swaps}")
                if max_total_steps is not None and target > max_total_steps:
                    return -1, -1, np.zeros(1)
                self.index_evolution = torch.cat(
                    [
                        self.index_evolution,
                        torch.zeros(
                            num_swaps, len(self), self.num_chains, dtype=torch.int
                        ),
                    ]
                )
            tau_int, tau_exp, C = _process_experiment(
                swaps=self.index_evolution, plot=plot, n_therm=0
            )
            if filename is not None:
                with h5py.File(filename, "a") as f:
                    if "trwa" not in f.keys():
                        f.create_group("trwa")
                    if "C" in f["trwa"]:
                        del f["trwa"]["C"]
                    if "tau_int" in f["trwa"]:
                        del f["trwa"]["tau_int"]
                    if "tau_exp" in f["trwa"]:
                        del f["trwa"]["tau_exp"]
                    f["trwa"]["C"] = C
                    f["trwa"]["tau_int"] = tau_int
                    f["trwa"]["tau_exp"] = tau_exp
        else:
            with h5py.File(filename, "r") as f:
                C = f["trwa"]["C"][()]
                tau_int = f["trwa"]["tau_int"][()]
                tau_exp = f["trwa"]["tau_exp"][()]
        return max(tau_int, 1), max(tau_exp, 1), C
