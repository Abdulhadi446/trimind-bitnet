import json
import logging
import os
import torch
import torch.nn as nn
from safetensors import safe_open
from huggingface_hub import try_to_load_from_cache

logger = logging.getLogger(__name__)

TERNARY_MAP = {-1: 0, 0: 1, 1: 2}


def materialize_meta_tensors(model: nn.Module):
    """Load meta-device parameters from original safetensors in HF cache.

    device_map='auto' offloads some layers to disk (meta device). This loads
    each from the original checkpoint, quantizes in-place, then frees the copy.
    """
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
                # Navigate to the parameter and replace it
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

    # First materialize any meta (disk-offloaded) tensors
    materialize_meta_tensors(model)

    count = 0
    for name, child in model.named_children():
        if name in exclude_names:
            continue
        if isinstance(child, nn.Linear):
            w = child.weight.data
            if w.device.type == "meta":
                logger.warning("Skipping meta tensor: %s — materialize weights first.", name)
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


def pack_ternary(w: torch.Tensor) -> tuple[bytes, float]:
    if w.device.type == "meta":
        raise RuntimeError("Cannot pack meta tensor — materialize weights first")
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
    return packed.numpy().tobytes(), scale


def unpack_ternary(packed: bytes, shape: tuple, scale: float, dtype=torch.bfloat16, device="cpu") -> torch.Tensor:
    packed_t = torch.frombuffer(packed, dtype=torch.uint8, device=device)
    idx = torch.stack([
        (packed_t >> 6) & 3,
        (packed_t >> 4) & 3,
        (packed_t >> 2) & 3,
        packed_t & 3,
    ], dim=1).view(-1)
    n = shape[0] * shape[1]
    idx = idx[:n]
    w = (idx.to(torch.int8).sub_(1)).to(dtype) * scale
    return w.view(shape)


def save_quantized(model: nn.Module, save_dir: str):
    """Save quantized model in packed 2-bit format (~3 GB for a 12B model).
    Weights can be on any device — each is moved to CPU individually for packing."""
    os.makedirs(save_dir, exist_ok=True)
    # Materialize any remaining meta tensors before iterating
    materialize_meta_tensors(model)
    metadata = {}
    all_packed = {}
    meta_skipped = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            key = name + ".weight"
            w = module.weight.data
            if w.device.type == "meta":
                meta_skipped += 1
                continue
            data, scale = pack_ternary(w)
            all_packed[key] = data
            metadata[key] = {
                "shape": list(w.shape),
                "scale": scale,
                "dtype": str(w.dtype).split(".")[-1],
            }
    with open(os.path.join(save_dir, "ternary_metadata.json"), "w") as f:
        json.dump(metadata, f)
    packed_cat = b"".join(all_packed[k] for k in metadata)
    with open(os.path.join(save_dir, "ternary_packed.bin"), "wb") as f:
        f.write(packed_cat)
    offsets = {}
    pos = 0
    for key in metadata:
        offsets[key] = pos
        n = metadata[key]["shape"][0] * metadata[key]["shape"][1]
        pos += (n + 3) // 4
    with open(os.path.join(save_dir, "ternary_offsets.json"), "w") as f:
        json.dump(offsets, f)
    logger.info("Saved packed ternary: %d layers, %.1f MB packed -> %.1f MB unpacked%s",
                len(metadata),
                pos / (1024 * 1024),
                sum(p[0]*p[1] for p in [v["shape"] for v in metadata.values()]) * 2 / (1024 * 1024),
                f" (skipped {meta_skipped} meta layers)" if meta_skipped else "")


def load_quantized_weights(model: nn.Module, save_dir: str, device="cpu"):
    with open(os.path.join(save_dir, "ternary_metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(save_dir, "ternary_offsets.json")) as f:
        offsets = json.load(f)
    with open(os.path.join(save_dir, "ternary_packed.bin"), "rb") as f:
        data = f.read()
    for name, module in model.named_modules():
        key = name + ".weight"
        if key not in metadata:
            continue
        meta = metadata[key]
        pos = offsets[key]
        n = meta["shape"][0] * meta["shape"][1]
        byte_len = (n + 3) // 4
        packed = data[pos:pos + byte_len]
        dtype = getattr(torch, meta["dtype"], torch.bfloat16)
        w = unpack_ternary(packed, tuple(meta["shape"]), meta["scale"], dtype=dtype, device=device)
        module.weight.data = w.to(device)
    logger.info("Loaded packed ternary weights for %d layers.", len(metadata))
