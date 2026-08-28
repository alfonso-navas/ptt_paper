from __future__ import annotations

import numpy as np
import torch
from rbms.classes import EBM
from torch import Tensor

from ptt_paper.pre_sampler.classes import PreSampler


class Reservoir(PreSampler):
    def __init__(self, ebm: EBM, reservoir: Tensor, use: bool):
        super().__init__(ebm=ebm)
        self.device = reservoir.device
        self.dtype = reservoir.dtype
        self.reservoir = reservoir
        self.use = use
        self.name = "Reservoir"

    @property
    def num_samples(self):
        return self.reservoir.shape[0]

    def sample(self, num_samples: int) -> None:
        self.index = torch.randperm(self.reservoir.shape[0], device=self.device)[
            :num_samples
        ]
        self.v = self.reservoir[self.index]

    def compute_swap_acc(self, visible_conf):
        return torch.ones(self.v.shape[0], device=self.device, dtype=torch.bool)

    def perform_swap(self, visible_conf, swap_mask):
        if self.use:
            tmp = visible_conf.reshape(
                visible_conf.shape[0],
                self.reservoir.shape[-1],
                -1,
            )
            if tmp.shape[-1] < 2:
                tmp = visible_conf
            else:
                tmp = tmp.argmax(-1).to(self.reservoir.dtype)
            self.sample(tmp.shape[0])

            self.reservoir[self.index] = tmp
            return self.v
        else:
            return visible_conf

    def clone(self, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return Reservoir(
            ebm=self.ebm.clone().to(device=device, dtype=dtype),
            reservoir=self.reservoir.clone().to(device=device, dtype=dtype),
            use=self.use,
        )

    def named_parameters(self) -> dict[str, np.ndarray]:
        return {
            "reservoir": self.reservoir.cpu().numpy(),
            "use": np.asarray(self.use),
            "pre_sampler_type": np.asarray(self.name, dtype="T"),
        }

    @staticmethod
    def set_named_parameters(
        ebm: EBM,
        named_params: dict[str, np.ndarray],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Reservoir:
        reservoir = torch.from_numpy(named_params.pop("reservoir")).to(
            device=device, dtype=dtype
        )
        use = bool(named_params.pop("use"))
        return Reservoir(ebm=ebm, reservoir=reservoir, use=use)
