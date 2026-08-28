from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from rbms.classes import EBM
from rbms.custom_fn import swap_tensor
from torch import Tensor


class PreSampler(ABC):
    v: Tensor
    swap_idx: Tensor
    ebm: EBM
    name: str

    def __init__(self, ebm: EBM):
        self.ebm = ebm

    @abstractmethod
    def sample(self, num_samples: int) -> None:
        """
        Sample the PreSampler and store them internally

        Args:
            num_samples (int): Number of generated samples.
        """
        ...

    @abstractmethod
    def compute_swap_acc(self, visible_conf: Tensor) -> Tensor:
        """Compute the MH acceptance rate between the internal configurations and the provided ones

        Args:
            visible_conf (Tensor): Provided configurations from the equilibrium distribution of the internal EBM.

        Returns:
            Tensor: Boolean tensor corresponding to the swap to perform.
        """
        ...

    def perform_swap(self, visible_conf: Tensor, swap_mask: Tensor) -> Tensor:
        """Perform the swap between the provided configurations and the internal ones.

        Args:
            visible_conf (Tensor): Provided configurations from the equilibrium distribution of the internal EBM.
            swap_mask (Tensor): Boolean tensor corresponding to the swap to perform.

        Returns:
            Tensor: Tensor with the updated visible_conf.
        """
        visible_conf, self.v = swap_tensor(
            v1=visible_conf, v2=self.v, swap_mask=swap_mask, swap_only_v1=False
        )
        return visible_conf

    @abstractmethod
    def clone(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> PreSampler: ...

    @abstractmethod
    def named_parameters(self) -> dict[str, np.ndarray]: ...

    @staticmethod
    @abstractmethod
    def set_named_parameters(
        ebm: EBM,
        named_params: dict[str, np.ndarray],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> PreSampler: ...
