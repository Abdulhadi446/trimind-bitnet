import json
import logging
import os
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

TERNARY_MAP = {-1: 0, 0: 1, 1: 2}


def ternary_scale(weight: torch.Tensor) -> torch.Tensor:
    return weight.abs().mean()


def quantize_inplace(model: nn.Module, exclude_names: set[str] | None = None) -> int:
    """Quantize all Linear layer weights to ternary {-1,0,+1} in-place.

    Weights become ternary values scaled by absmean. Storage dtype remains
    bf16/float32 — use pack_quantized() during save for true compression.
    """
    if exclude_names is None:
        exclude_names = set()

    count = 0
    for name, child in model.named_children():
        if name in exclude_names:
            continue
        if isinstance(child, nn.Linear):
            w = child.weight.data
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
    """Pack a ternary weight tensor into 2-bit-per-value bytes.

    Each ternary value {-1,0,1} is mapped to 2 bits (0,1,2).
    4 values per byte. Returns (packed_bytes, scale).
    """
    w_flat = w.view(-1).cpu()
    scale = w_flat.abs().mean().item()
    if scale == 0:
        scale = 1.0
    # Extract sign: -1 -> 0, 0 -> 1, +1 -> 2
    idx = (w_flat / scale).to(torch.int8).add_(1).clamp_(0, 2)
    # Pad to multiple of 4
    n = idx.numel()
    pad = (4 - n % 4) % 4
    if pad:
        idx = torch.cat([idx, torch.zeros(pad, dtype=torch.int8)])
    idx = idx.view(-1, 4).to(torch.uint8)
    packed = (idx[:, 0] << 6) | (idx[:, 1] << 4) | (idx[:, 2] << 2) | idx[:, 3]
    return packed.numpy().tobytes(), scale


def unpack_ternary(packed: bytes, shape: tuple, scale: float, dtype=torch.bfloat16, device="cpu") -> torch.Tensor:
    """Unpack 2-bit ternary bytes back into a full weight tensor."""
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
    """Save quantized model in packed 2-bit format (~3 GB for a 12B model)."""
    os.makedirs(save_dir, exist_ok=True)
    metadata = {}
    all_packed = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            key = name + ".weight"
            data, scale = pack_ternary(module.weight.data)
            all_packed[key] = data
            metadata[key] = {
                "shape": list(module.weight.shape),
                "scale": scale,
                "dtype": str(module.weight.dtype).split(".")[-1],
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
    logger.info("Saved packed ternary: %d layers, %.1f MB -> %.1f MB unpacked",
                len(metadata),
                pos / (1024 * 1024),
                sum(p[0]*p[1] for p in [v["shape"] for v in metadata.values()]) * 2 / (1024 * 1024))


def load_quantized_weights(model: nn.Module, save_dir: str, device="cpu"):
    """Load packed ternary weights back into a model."""
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
