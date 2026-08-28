import numpy as np
import torch
from rbms.classes import EBM

from ptt_paper.pre_sampler.classes import PreSampler
from ptt_paper.pre_sampler.rcm import BBRCM, IIRCM, PBRCM
from ptt_paper.pre_sampler.reservoir import Reservoir

map_pre_sampler: dict[str, type[PreSampler]] = {
    "BBRCM": BBRCM,
    "PBRCM": PBRCM,
    "IIRCM": IIRCM,
    "Reservoir": Reservoir,
}


def get_pre_sampler(
    named_params: dict[str, np.ndarray],
    ebm: EBM,
    device: torch.device | str,
    dtype: torch.dtype,
) -> PreSampler | None:
    pre_sampler_type = named_params.pop("pre_sampler_type")
    if pre_sampler_type.dtype != np.dtypes.StringDType():
        pre_sampler_type = pre_sampler_type.astype(str)
    pre_sampler_type = str(pre_sampler_type)
    if pre_sampler_type == "none":
        return None
    return map_pre_sampler[pre_sampler_type].set_named_parameters(
        ebm=ebm, named_params=named_params, device=device, dtype=dtype
    )
