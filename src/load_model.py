import logging
import os
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from huggingface_hub import try_to_load_from_cache

from . import hardware_utils

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

try:
    from transformers import Qwen3VLForConditionalGeneration as _VLModelClass
except ImportError:
    from transformers import AutoModel as _VLModelClass


def _cache_path() -> str | None:
    """Return the HF cache directory path if it contains model files."""
    snapshot = try_to_load_from_cache(MODEL_ID, "config.json")
    if snapshot is not None and not snapshot.startswith("https://"):
        return os.path.dirname(os.path.dirname(snapshot))
    return None


def load_model(device_override: str | None = None, dtype_override: str | None = None):
    """Load model with automatic hardware-aware configuration.

    Args:
        device_override: Force a device string (e.g. 'cuda:0', 'cpu').
        dtype_override: Force a dtype/precision ('bf16', '8bit', '4bit').

    Returns:
        Tuple of (model, processor, device).
    """
    logger.info("Loading model: %s", MODEL_ID)
    cached = _cache_path()
    if cached:
        logger.info("Model cache found at: %s", cached)
    else:
        logger.info("No local cache — will download from Hugging Face Hub (~17.5 GB)")

    device, dtype = hardware_utils.detect_device()
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
        model = _VLModelClass.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
            local_files_only=cached is not None,
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        logger.error("Failed to load model: %s", e)
        logger.info(
            "If OOM, try setting dtype_override='4bit' or running on a "
            "different device (TPU/GPU with more VRAM)."
        )
        raise

    if device.type != "cuda":
        model = model.to(device)

    hardware_utils.log_memory("after_load", device)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    logger.info("Model loaded on %s with dtype %s.", device, dtype)
    return model, processor, device
