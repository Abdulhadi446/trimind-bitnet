import logging
import os
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from huggingface_hub import try_to_load_from_cache

from . import hardware_utils

logger = logging.getLogger(__name__)

MODEL_ID = "google/gemma-4-12B-it"

try:
    from transformers import Gemma4UnifiedForConditionalGeneration as _VLModelClass
except ImportError:
    try:
        from transformers import Gemma4ForCausalLM as _VLModelClass
    except ImportError:
        from transformers import AutoModel as _VLModelClass


def _cache_complete() -> bool:
    """Check if actual weight files exist in cache (not just metadata)."""
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json", "model.safetensors", "model-00001-of-00004.safetensors"):
        result = try_to_load_from_cache(MODEL_ID, filename)
        if result is not None and not result.startswith("https://"):
            snapshot_dir = os.path.dirname(result)
            if os.path.exists(os.path.join(snapshot_dir, filename)):
                return True
    return False


def load_model(device_override: str | None = None, dtype_override: str | None = None):
    """Load model with automatic hardware-aware configuration.

    Args:
        device_override: Force a device string (e.g. 'cuda:0', 'cpu').
        dtype_override: Force a dtype/precision ('bf16', '8bit', '4bit').

    Returns:
        Tuple of (model, processor, device).
    """
    logger.info("Loading model: %s", MODEL_ID)
    cache_ok = _cache_complete()
    if cache_ok:
        logger.info("Model weights found in HF cache.")
    else:
        logger.info("No cached weights — will download from Hugging Face Hub (~24 GB)")

    device, dtype = hardware_utils.detect_device(model_size_gb=24)
    if device_override:
        device = torch.device(device_override)
    if dtype_override:
        dtype = dtype_override

    hardware_utils.log_memory("before_load", device)

    quantization_config = None
    torch_dtype = torch.bfloat16 if device.type != "cpu" else torch.float32

    if isinstance(dtype, str):
        if dtype == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Using 8-bit quantization via bitsandbytes.")
        elif dtype == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization via bitsandbytes.")
        else:
            raise ValueError(f"Unknown precision string: {dtype}")
    else:
        torch_dtype = dtype

    try:
        # If CUDA VRAM can't fit the full model, load on CPU (avoids meta tensors)
        use_device_map = False
        if device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if dtype == torch.bfloat16 and total_vram < 24 * 1.2:
                logger.warning(
                    "VRAM (%.1f GB) insufficient for full model at BF16. "
                    "Loading on CPU instead (avoids offloaded meta tensors).",
                    total_vram,
                )
                device = torch.device("cpu")
                torch_dtype = torch.bfloat16
            else:
                use_device_map = True

        model = _VLModelClass.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map="auto" if use_device_map else None,
            trust_remote_code=True,
            local_files_only=cache_ok,
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        logger.error("Failed to load model: %s", e)
        logger.info(
            "If OOM, try setting dtype_override='4bit' or running on a "
            "different device (TPU/GPU with more VRAM)."
        )
        raise

    # Move to device if not already there and not dispatched by accelerate
    if use_device_map:
        logger.info("Model dispatched by accelerate (device_map=auto).")
    elif device.type == "cuda":
        model = model.to(device)
    else:
        logger.info("Model loaded on %s.", device)

    actual_device = model.device if hasattr(model, "device") and model.device.type else device

    hardware_utils.log_memory("after_load", device)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    logger.info("Model loaded on %s with dtype %s.", device, dtype)
    return model, processor, actual_device
