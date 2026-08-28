from __future__ import annotations

from typing import override

import numpy as np
import torch
from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.classes import EBM
from rbms.custom_fn import one_hot
from rbms.ising_ising.classes import IIRBM
from rbms.potts_bernoulli.classes import PBRBM
from torch import Tensor

from ptt_paper.pre_sampler.classes import PreSampler
from ptt_paper.pre_sampler.implement import sample_rcm_bernoulli, sample_rcm_potts


class BBRCM(PreSampler):
    def __init__(
        self,
        ebm: BBRBM,
        p_m: Tensor,
        mu: Tensor,
        U: Tensor,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ):
        super().__init__(ebm=ebm)
        if device is None:
            device = U.device
        if dtype is None:
            dtype = U.dtype
        self.device = device
        self.dtype = dtype
        self.p_m = p_m.to(device=self.device, dtype=dtype)
        self.mu = mu.to(device=self.device, dtype=dtype)
        self.U = U.to(device=self.device, dtype=dtype)
        self.swap_acc = torch.ones(2, device=self.device, dtype=self.dtype)
        self.name = "BBRCM"

    def sample(self, num_samples):
        self.v = sample_rcm_bernoulli(
            p_m=self.p_m,
            mu=self.mu,
            U=self.U,
            num_samples=num_samples,
            device=self.device,
            dtype=self.dtype,
        )

    @override
    def compute_swap_acc(self, visible_conf):
        return self.swap_acc

    def perform_swap(self, visible_conf, swap_mask):
        return self.v

    def clone(self, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        new_ebm = None
        if self.ebm is not None:
            new_ebm = self.ebm.clone().to(device=device, dtype=dtype)
        return BBRCM(
            ebm=new_ebm,  # type: ignore
            p_m=self.p_m.clone().to(device=device, dtype=dtype),
            mu=self.mu.clone().to(device=device, dtype=dtype),
            U=self.U.clone().to(device=device, dtype=dtype),
            device=device,
            dtype=dtype,
        )

    def named_parameters(self) -> dict[str, np.ndarray]:
        return {
            "p_m": self.p_m.cpu().numpy(),
            "mu": self.mu.cpu().numpy(),
            "U": self.U.cpu().numpy(),
            "pre_sampler_type": np.asarray(self.name, dtype="T"),
        }

    @staticmethod
    def set_named_parameters(
        ebm: EBM,
        named_params: dict[str, np.ndarray],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> BBRCM:
        p_m = torch.from_numpy(named_params.pop("p_m")).to(device=device, dtype=dtype)
        mu = torch.from_numpy(named_params.pop("mu")).to(device=device, dtype=dtype)
        U = torch.from_numpy(named_params.pop("U")).to(device=device, dtype=dtype)
        return BBRCM(ebm=ebm, p_m=p_m, mu=mu, U=U, device=device, dtype=dtype)  # type: ignore


class PBRCM(PreSampler):
    def __init__(
        self,
        ebm: PBRBM,
        p_m: Tensor,
        mu: Tensor,
        U: Tensor,
        num_colors: int,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ):
        super().__init__(ebm=ebm)
        if device is None:
            device = U.device
        if dtype is None:
            dtype = U.dtype
        self.device = device
        self.dtype = dtype
        self.p_m = p_m.to(device=self.device, dtype=dtype)
        self.mu = mu.to(device=self.device, dtype=dtype)
        self.U = U.to(device=self.device, dtype=dtype)
        self.swap_acc = torch.ones(2, device=self.device, dtype=self.dtype)
        self.num_colors = num_colors

    def sample(self, num_samples):
        self.v = sample_rcm_potts(
            p_m=self.p_m,
            mu=self.mu,
            U=self.U,
            num_samples=num_samples,
            num_colors=self.num_colors,
            device=self.device,
            dtype=self.dtype,
        )

    @override
    def compute_swap_acc(self, visible_conf):
        return self.swap_acc

    def perform_swap(self, visible_conf, swap_mask):
        return (
            one_hot(self.v.int(), num_classes=self.num_colors)
            .reshape(self.v.shape[0], -1)
            .to(self.dtype)
        )

    def clone(self, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return PBRCM(
            ebm=self.ebm.clone(device=device, dtype=dtype),  # type: ignore
            p_m=self.p_m.clone().to(device=device, dtype=dtype),
            mu=self.mu.clone().to(device=device, dtype=dtype),
            U=self.U.clone().to(device=device, dtype=dtype),
            num_colors=self.num_colors,
            device=device,
            dtype=dtype,
        )

    def named_parameters(self) -> dict[str, np.ndarray]:
        return {
            "p_m": self.p_m.cpu().numpy(),
            "mu": self.mu.cpu().numpy(),
            "U": self.U.cpu().numpy(),
            "num_colors": np.asarray(self.num_colors),
            "pre_sampler_type": np.asarray(self.name, dtype="T"),
        }

    @staticmethod
    def set_named_parameters(
        ebm: EBM,
        named_params: dict[str, np.ndarray],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> PBRCM:
        p_m = torch.from_numpy(named_params.pop("p_m")).to(device=device, dtype=dtype)
        mu = torch.from_numpy(named_params.pop("mu")).to(device=device, dtype=dtype)
        U = torch.from_numpy(named_params.pop("U")).to(device=device, dtype=dtype)
        num_colors = int(named_params.pop("num_colors"))
        return PBRCM(
            ebm=ebm,  # type: ignore
            p_m=p_m,
            mu=mu,
            U=U,
            num_colors=num_colors,
            device=device,
            dtype=dtype,
        )


class IIRCM(PreSampler):
    def __init__(
        self,
        ebm: IIRBM,
        p_m: Tensor,
        mu: Tensor,
        U: Tensor,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ):
        super().__init__(ebm=ebm)
        if device is None:
            device = U.device
        if dtype is None:
            dtype = U.dtype
        self.device = device
        self.dtype = dtype
        self.p_m = p_m.to(device=self.device, dtype=dtype)
        self.mu = mu.to(device=self.device, dtype=dtype)
        self.U = U.to(device=self.device, dtype=dtype)
        self.swap_acc = torch.ones(2, device=self.device, dtype=self.dtype)
        self.name = "IIRCM"

    def sample(self, num_samples):
        self.v = (
            2
            * sample_rcm_bernoulli(
                p_m=self.p_m,
                mu=self.mu,
                U=self.U,
                num_samples=num_samples,
                device=self.device,
                dtype=self.dtype,
            )
            - 1
        )

    @override
    def compute_swap_acc(self, visible_conf):
        return self.swap_acc

    def perform_swap(self, visible_conf, swap_mask):
        return self.v

    def clone(self, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        new_ebm = None
        if self.ebm is not None:
            new_ebm = self.ebm.clone().to(device=device, dtype=dtype)
        return IIRCM(
            ebm=new_ebm,  # type: ignore
            p_m=self.p_m.clone().to(device=device, dtype=dtype),
            mu=self.mu.clone().to(device=device, dtype=dtype),
            U=self.U.clone().to(device=device, dtype=dtype),
            device=device,
            dtype=dtype,
        )

    def named_parameters(self) -> dict[str, np.ndarray]:
        return {
            "p_m": self.p_m.cpu().numpy(),
            "mu": self.mu.cpu().numpy(),
            "U": self.U.cpu().numpy(),
            "pre_sampler_type": np.asarray(self.name, dtype="T"),
        }

    @staticmethod
    def set_named_parameters(
        ebm: EBM,
        named_params: dict[str, np.ndarray],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> IIRCM:
        p_m = torch.from_numpy(named_params.pop("p_m")).to(device=device, dtype=dtype)
        mu = torch.from_numpy(named_params.pop("mu")).to(device=device, dtype=dtype)
        U = torch.from_numpy(named_params.pop("U")).to(device=device, dtype=dtype)
        return IIRCM(ebm=ebm, p_m=p_m, mu=mu, U=U, device=device, dtype=dtype)  # type: ignore
