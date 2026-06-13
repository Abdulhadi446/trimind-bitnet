import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def ternary_weights(weight: torch.Tensor) -> torch.Tensor:
    scale = weight.abs().mean()
    threshold = 0.7 * scale
    return torch.where(
        weight > threshold,
        scale,
        torch.where(weight < -threshold, -scale, torch.tensor(0.0, device=weight.device, dtype=weight.dtype)),
    )


def quantize_inplace(model: nn.Module, exclude_names: set[str] | None = None) -> int:
    """Quantize all Linear layer weights to ternary {-1,0,+1} in-place.

    Modifies weight.data directly — no extra memory is allocated beyond the
    model itself. Original full-precision weights are lost (the model becomes
    the quantized version).

    Args:
        model: Any PyTorch module.
        exclude_names: Module names to skip (e.g. {'lm_head'}).

    Returns:
        Number of layers quantized.
    """
    if exclude_names is None:
        exclude_names = set()

    count = 0
    for name, child in model.named_children():
        if name in exclude_names:
            continue
        if isinstance(child, nn.Linear):
            child.weight.data = ternary_weights(child.weight.data)
            count += 1
        else:
            count += quantize_inplace(child, exclude_names)
    return count


def replace_linear_with_ternary(model: nn.Module, exclude_names: set[str] | None = None) -> int:
    """Alias for quantize_inplace. Kept for backward compatibility."""
    return quantize_inplace(model, exclude_names)
