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
    """Load meta-device parameters from original safetensors in HF cache."""
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
        logger.error("Cannot find HF cache for %s — skipping meta materialization.", model_id)
        return

    logger.info("Materializing %d meta parameters from %s ...", len(meta_params), base_path)

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
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            logger.warning("Missing safetensors: %s", filepath)
            continue
        with safe_open(filepath, framework="pt") as sf:
            for name in param_names:
                tensor = sf.get_tensor(name)
                parts = name.split(".")
                mod = model
                for part in parts[:-1]:
                    if part.isdigit():
                        mod = mod[int(part)]
                    else:
                        mod = getattr(mod, part)
                param_slot = mod._parameters.get(parts[-1])
                if param_slot is not None:
                    mod._parameters[parts[-1]] = nn.Parameter(tensor)
                else:
                    logger.warning("Cannot find param slot: %s", name)

    logger.info("Materialized %d meta parameters.", len(meta_params))


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
                logger.warning("Skipping meta tensor: %s.", name)
                continue
            scale = ternary_scale(w)
            threshold = 0.7 * scale
            child.weight.data = torch.where(
                w > threshold,
                scale,
                torch.where(w < -threshold, -scale, torch.tensor(0.0, device=w.device, dtype=w.dtype)),
            )
            count += 1
        else:
            count += quantize_inplace(child, exclude_names)
    return count


def save_quantized(model: nn.Module, save_dir: str):
    """Save quantized model as standard safetensors.

    After quantization, weights are ternary {-scale, 0, scale} stored in BF16.
    This saves them as model-*.safetensors — directly loadable with
    ``AutoModelForCausalLM.from_pretrained()``.
    """
    os.makedirs(save_dir, exist_ok=True)
    materialize_meta_tensors(model)

    model.config.save_pretrained(save_dir)
    if hasattr(model, "generation_config"):
        model.generation_config.save_pretrained(save_dir)

    model.save_pretrained(save_dir, safe_serialization=True)

    raw_gb = sum(p.numel() for p in model.parameters()) * 2 / (1024**3)
    logger.info("Saved quantized model: %.2f GB (safetensors)", raw_gb)
