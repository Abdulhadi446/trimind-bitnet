import logging
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from . import hardware_utils

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def load_model(device_override: str | None = None, dtype_override: str | None = None):
    """Load model with automatic hardware-aware configuration.

    Args:
        device_override: Force a device string (e.g. 'cuda:0', 'cpu').
        dtype_override: Force a dtype/precision ('bf16', '8bit', '4bit').

    Returns:
        Tuple of (model, processor, device).
    """
    logger.info("Loading model: %s", MODEL_ID)

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
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        logger.error("Failed to load model: %s", e)
        logger.info(
            "If OOM, try setting dtype_override='4bit' or running on a "
            "different device (TPU/GPU with more VRAM)."
        )
        raise

    # Move to device if not using device_map
    if device.type != "cuda":
        model = model.to(device)

    hardware_utils.log_memory("after_load", device)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    logger.info("Model loaded on %s with dtype %s.", device, dtype)
    return model, processor, device
