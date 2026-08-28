import torch
from torch import Tensor
from torch.nn.functional import softmax


def compute_ess(logit_weights: Tensor) -> Tensor:
    """Computes the Effective Sample Size of the chains.

    Args:
        logit_weights: minus log-weights of the chains.
    """
    lwc = logit_weights - logit_weights.min()
    numerator = torch.square(torch.mean(torch.exp(-lwc)))
    denominator = torch.mean(torch.exp(-2.0 * lwc))

    return numerator / denominator


@torch.jit.script
def systematic_resampling(
    chains: dict[str, Tensor], log_weights: Tensor
) -> dict[str, Tensor]:
    """Performs the systematic resampling of the chains according to their relative weight and
    sets the logit_weights back to zero.

    Args:
        chains (Chain): Chains.

    Returns:
        Chain: Resampled chains.
    """
    num_chains = chains["visible"].shape[0]
    device = chains["visible"].device
    weights = softmax(-log_weights, -1)
    weights_span = torch.cumsum(weights.double(), dim=0).float()
    rand_unif = torch.rand(size=(1,), device=device)
    arrow_span = (torch.arange(num_chains, device=device) + rand_unif) / num_chains
    mask = (weights_span.reshape(num_chains, 1) >= arrow_span).sum(1)
    counts = torch.diff(mask, prepend=torch.tensor([0], device=device))
    chains["visible"] = torch.repeat_interleave(chains["visible"], counts, dim=0)
    chains["hidden"] = torch.repeat_interleave(chains["hidden"], counts, dim=0)

    return chains
