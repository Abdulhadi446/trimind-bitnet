import logging
import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import try_to_load_from_cache

from . import hardware_utils

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-8B"

def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None

def _cache_complete() -> bool:
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json", "model.safetensors", "model-00001-of-00004.safetensors"):
        result = try_to_load_from_cache(MODEL_ID, filename)
        if result is not None and not result.startswith("https://"):
            snapshot_dir = os.path.dirname(result)
            if os.path.exists(os.path.join(snapshot_dir, filename)):
                return True
    return False


def load_model(device_override: str | None = None, dtype_override: str | None = None):
    logger.info("Loading model: %s", MODEL_ID)
    cache_ok = _cache_complete()
    if cache_ok:
        logger.info("Model weights found in HF cache.")
    else:
        logger.info("No cached weights — will download from Hugging Face Hub (~16 GB)")

    device, dtype = hardware_utils.detect_device(model_size_gb=16)
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
        max_memory = None
        use_device_map = device.type == "cuda"
        if use_device_map:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            try:
                import psutil
                ram_gb = psutil.virtual_memory().total / (1024**3)
            except ImportError:
                ram_gb = 0
            max_memory = {0: f"{int(vram_gb)}GiB"}
            if ram_gb > 0:
                max_memory["cpu"] = f"{int(ram_gb)}GiB"
            logger.info("Device map auto with max_memory: %s", max_memory)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map="auto" if use_device_map else None,
            max_memory=max_memory,
            token=_hf_token(),
            trust_remote_code=True,
            local_files_only=cache_ok,
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        logger.error("Failed to load model: %s", e)
        raise

    if use_device_map:
        logger.info("Model dispatched by accelerate (device_map=auto).")
    else:
        model = model.to(device)
        logger.info("Model loaded on %s.", device)

    actual_device = model.device if hasattr(model, "device") and model.device.type else device

    hardware_utils.log_memory("after_load", device)

    processor = AutoProcessor.from_pretrained(MODEL_ID, token=_hf_token(), trust_remote_code=True)

    logger.info("Model loaded on %s with dtype %s.", device, dtype)
    return model, processor, actual_device
