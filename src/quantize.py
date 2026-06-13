import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def ternary_weights(weight: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Quantize a weight tensor to ternary {-1, 0, +1} using absmean scaling.

    Args:
        weight: Full-precision weight tensor.
        scale: Scaling factor (typically group/global absmean).

    Returns:
        Ternary weight tensor of same shape with values in {-scale, 0, +scale}.
    """
    threshold = 0.7 * scale
    return torch.where(
        weight > threshold,
        scale,
        torch.where(weight < -threshold, -scale, torch.tensor(0.0, device=weight.device)),
    )


class TernaryLinear(nn.Module):
    """Wrapper that applies ternary quantization to a parent nn.Linear layer.

    Stores full-precision weights internally for optional fine-tuning,
    but uses ternary-quantized weights during forward.
    """

    def __init__(self, parent: nn.Linear):
        super().__init__()
        self.in_features = parent.in_features
        self.out_features = parent.out_features
        self.weight = nn.Parameter(parent.weight.data.clone().detach())
        if parent.bias is not None:
            self.bias = nn.Parameter(parent.bias.data.clone().detach())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_ternary = ternary_weights(self.weight.data, scale=self.weight.abs().mean())
        return nn.functional.linear(x, w_ternary, self.bias)


def replace_linear_with_ternary(model: nn.Module, exclude_names: set | None = None):
    """Recursively replace nn.Linear layers in a model with TernaryLinear.

    Args:
        model: Any PyTorch module (transformers model, etc.).
        exclude_names: Set of module names to skip (e.g. {'lm_head', 'embed_tokens'}).

    Returns:
        Model with Linear layers replaced in-place.
    """
    if exclude_names is None:
        exclude_names = set()

    count = 0
    for name, child in model.named_children():
        if name in exclude_names:
            continue
        if isinstance(child, nn.Linear):
            setattr(model, name, TernaryLinear(child))
            count += 1
        else:
            count += replace_linear_with_ternary(child, exclude_names)
    return count


def revert_ternary_to_linear(model: nn.Module):
    """Reverse ternary quantization — restore original nn.Linear from TernaryLinear.

    Useful for saving in a standard format.
    """
    for name, child in model.named_children():
        if isinstance(child, TernaryLinear):
            new_linear = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
            new_linear.weight.data = child.weight.data.clone().detach()
            if child.bias is not None:
                new_linear.bias.data = child.bias.data.clone().detach()
            setattr(model, name, new_linear)
        else:
            revert_ternary_to_linear(child)
