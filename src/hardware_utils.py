import logging
import os
import psutil
import torch

logger = logging.getLogger(__name__)


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes
        from packaging import version

        return version.parse(bitsandbytes.__version__) >= version.parse("0.46.1")
    except (ImportError, AttributeError, Exception):
        return False


def detect_device(model_size_gb: float | None = None):
    """Auto-detect best available hardware and return (device, dtype).

    Args:
        model_size_gb: Estimated model size in GB (used for VRAM/RAM decisions).
    """
    est = model_size_gb or 16  # default estimate

    # Check for TPU (torch-xla)
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        if model_size_gb and model_size_gb > 4:
            logger.warning(
                "TPU detected but model is ~%.1f GB — may not fit in TPU HBM. "
                "Falling back to CPU. Use --dtype 4bit or a GPU runtime.",
                model_size_gb,
            )
            raise RuntimeError("Model too large for TPU")
        logger.info("TPU detected via torch-xla. Using device: %s", device)
        return device, torch.bfloat16
    except (ImportError, RuntimeError, AttributeError):
        pass

    # Check for CUDA GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        total_vram = torch.cuda.get_device_properties(0).total_memory
        total_vram_gb = total_vram / (1024**3)
        logger.info(
            "CUDA GPU detected: %s (%.1f GB VRAM)",
            torch.cuda.get_device_name(0),
            total_vram_gb,
        )
        if total_vram_gb >= est * 1.2:
            logger.info("Sufficient VRAM for BF16 precision.")
            return device, torch.bfloat16
        elif _bitsandbytes_available():
            if total_vram_gb >= est * 0.6:
                logger.info("Using 8-bit bitsandbytes (VRAM: %.1f GB, model ~%.0f GB).", total_vram_gb, est)
                return device, "8bit"
            else:
                logger.info("Using 4-bit bitsandbytes (VRAM: %.1f GB, model ~%.0f GB).", total_vram_gb, est)
                return device, "4bit"
        else:
            logger.warning(
                "Low VRAM (%.1f GB) and bitsandbytes not installed. "
                "Install it with: pip install -U bitsandbytes>=0.46.1",
                total_vram_gb,
            )
            logger.info("Falling back to BF16 — may OOM on this GPU.")
            return device, torch.bfloat16
    else:
        logger.warning("No GPU detected. Falling back to CPU — this will be very slow.")
        total_ram = psutil.virtual_memory().total / (1024**3)
        logger.info("System RAM: %.1f GB", total_ram)
        if total_ram >= 32:
            logger.info("Sufficient RAM for BF16.")
            return torch.device("cpu"), torch.bfloat16
        else:
            logger.warning(
                "Low RAM (%.0f GB). Model may not fit. Use a GPU runtime.",
                total_ram,
            )
            return torch.device("cpu"), torch.float32


def log_memory(step_label: str, device: torch.device):
    """Log current memory usage for diagnostics."""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        mem = psutil.virtual_memory()
        logger.info(
            "[MEM %s] GPU allocated: %.2f GB | reserved: %.2f GB | RAM used: %.1f/%.1f GB (%.0f%%)",
            step_label,
            allocated,
            reserved,
            mem.used / (1024**3),
            mem.total / (1024**3),
            mem.percent,
        )
    elif device.type == "cpu":
        mem = psutil.virtual_memory()
        logger.info(
            "[MEM %s] RAM used: %.1f GB / %.1f GB (%.0f%%)",
            step_label,
            mem.used / (1024**3),
            mem.total / (1024**3),
            mem.percent,
        )
