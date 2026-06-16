import json
import logging
import os
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save_file
from huggingface_hub import try_to_load_from_cache

logger = logging.getLogger(__name__)


def materialize_meta_tensors(model: nn.Module):
    meta_params = [(n, p) for n, p in model.named_parameters() if p.device.type == "meta"]
    if not meta_params:
        return

    model_id = getattr(model.config, "_name_or_path", None)
    if model_id is None:
        logger.error("Cannot determine model ID — skipping meta materialization.")
        return

    base_path = None
    for filename in ("model.safetensors.index.json", "model.safetensors"):
        result = try_to_load_from_cache(model_id, filename)
        if result is not None and not result.startswith("https://"):
            base_path = os.path.dirname(result)
            break
    if base_path is None:
        logger.error("Cannot find HF cache for %s.", model_id)
        return

    logger.info("Materializing %d meta parameters ...", len(meta_params))
    index_path = os.path.join(base_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
    else:
        weight_map = {name: "model.safetensors" for name, _ in meta_params}

    file_to_params: dict[str, list[str]] = {}
    for name, _ in meta_params:
        fn = weight_map.get(name, "model.safetensors")
        file_to_params.setdefault(fn, []).append(name)

    for filename, param_names in file_to_params.items():
        fp = os.path.join(base_path, filename)
        if not os.path.exists(fp):
            continue
        with safe_open(fp, framework="pt") as sf:
            for name in param_names:
                tensor = sf.get_tensor(name)
                parts = name.split(".")
                mod = model
                for part in parts[:-1]:
                    mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
                if parts[-1] in mod._parameters:
                    mod._parameters[parts[-1]] = nn.Parameter(tensor)


def ternary_scale(weight: torch.Tensor) -> torch.Tensor:
    return weight.abs().mean()


def quantize_inplace(model: nn.Module, exclude_names: set[str] | None = None) -> int:
    if exclude_names is None:
        exclude_names = set()
    materialize_meta_tensors(model)
    count = 0
    for name, child in model.named_children():
        if name in exclude_names:
            continue
        if isinstance(child, nn.Linear):
            w = child.weight.data
            if w.device.type == "meta":
                continue
            scale = ternary_scale(w)
            threshold = 0.7 * scale
            child.weight.data = torch.where(
                w > threshold, scale,
                torch.where(w < -threshold, -scale, torch.tensor(0.0, device=w.device, dtype=w.dtype)),
            )
            count += 1
        else:
            count += quantize_inplace(child, exclude_names)
    return count


def _pack_ternary_tensor(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Pack a ternary weight into uint8 (4 values per byte). Returns (packed_uint8, scale)."""
    w_flat = w.view(-1).cpu()
    scale = w_flat.abs().mean().item()
    if scale == 0:
        scale = 1.0
    idx = (w_flat / scale).to(torch.int8).add_(1).clamp_(0, 2)
    n = idx.numel()
    pad = (4 - n % 4) % 4
    if pad:
        idx = torch.cat([idx, torch.zeros(pad, dtype=torch.int8)])
    idx = idx.view(-1, 4).to(torch.uint8)
    packed = (idx[:, 0] << 6) | (idx[:, 1] << 4) | (idx[:, 2] << 2) | idx[:, 3]
    return packed, scale


def _unpack_ternary_tensor(packed: torch.Tensor, shape: tuple, scale: float) -> torch.Tensor:
    """Unpack a packed uint8 tensor back to a float weight tensor."""
    idx = torch.stack([
        (packed >> 6) & 3, (packed >> 4) & 3, (packed >> 2) & 3, packed & 3,
    ], dim=1).view(-1)
    n = shape[0] * shape[1]
    idx = idx[:n]
    w = (idx.to(torch.int8).sub_(1)).to(torch.bfloat16) * scale
    return w.view(shape)


def save_quantized(model: nn.Module, save_dir: str):
    """Save quantized model as safetensors with 2-bit packed ternary weights (~2 GB)."""
    os.makedirs(save_dir, exist_ok=True)
    materialize_meta_tensors(model)

    model.config.save_pretrained(save_dir)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.save_pretrained(save_dir)

    state_dict = {}
    packed_info = {}

    # Collect all parameters: quantize linear weights, save others as-is
    all_params = dict(model.named_parameters())
    linear_param_keys = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            key = name + ".weight"
            if key in all_params:
                linear_param_keys.add(key)

    for key, param in all_params.items():
        if param.device.type == "meta":
            continue
        if key in linear_param_keys:
            packed_tensor, scale = _pack_ternary_tensor(param.data)
            state_dict[key + ".ternary_packed"] = packed_tensor
            state_dict[key + ".ternary_scale"] = torch.tensor([scale], dtype=torch.float16)
            packed_info[key] = {
                "shape": list(param.shape),
                "dtype": str(param.dtype).split(".")[-1],
            }
        else:
            state_dict[key] = param.data.cpu().to(torch.bfloat16)

    safetensors_save_file(state_dict, os.path.join(save_dir, "model.safetensors"))
    with open(os.path.join(save_dir, "ternary_packed_info.json"), "w") as f:
        json.dump({"packed_layers": packed_info, "version": 1}, f)

    packed_gb = sum(v.numel() for v in state_dict.values()) / (1024**3)
    n_linear = len(packed_info)
    n_total = len(state_dict)
    logger.info("Saved ternary safetensors: %.2f GB (%d layers packed, %d total tensors)",
                packed_gb, n_linear, n_total)


def load_quantized(model: nn.Module, model_dir: str, device="cpu"):
    """Load packed ternary weights from safetensors into a model."""
    with open(os.path.join(model_dir, "ternary_packed_info.json")) as f:
        packed_info = json.load(f).get("packed_layers", {})

    state_dict = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as sf:
        for key in sf.keys():
            tensor = sf.get_tensor(key)
            if key.endswith(".ternary_packed"):
                base_key = key[: -len(".ternary_packed")]
                meta = packed_info.get(base_key)
                if meta is None:
                    continue
                scale_key = base_key + ".ternary_scale"
                scale_t = sf.get_tensor(scale_key)
                scale = scale_t.item()
                shape = meta["shape"]
                idx = torch.stack([
                    (tensor >> 6) & 3, (tensor >> 4) & 3,
                    (tensor >> 2) & 3, tensor & 3,
                ], dim=1).view(-1)
                n = shape[0] * shape[1]
                idx = idx[:n]
                w = (idx.to(torch.int8).sub_(1)).to(torch.bfloat16) * scale
                state_dict[base_key] = w.view(shape)
            elif not key.endswith(".ternary_scale"):
                state_dict[key] = tensor.to(torch.bfloat16)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys: %d", len(unexpected))
    model.to(device)
    logger.info("Loaded %d tensors (%d packed).", len(state_dict), len(packed_info))
    return model
