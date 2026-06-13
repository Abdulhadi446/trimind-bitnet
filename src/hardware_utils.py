import logging
import psutil
import torch

logger = logging.getLogger(__name__)


def detect_device():
    """Auto-detect best available hardware and return (device, dtype)."""
    # Check for TPU (torch-xla)
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
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
        # If VRAM >= 16 GB, use BF16 (typical for 8B model)
        if total_vram_gb >= 32:
            logger.info("Sufficient VRAM for BF16 precision.")
            return device, torch.bfloat16
        elif total_vram_gb >= 16:
            logger.info("VRAM limited (%.1f GB). Using 8-bit via bitsandbytes.", total_vram_gb)
            return device, "8bit"
        else:
            logger.warning(
                "Low VRAM (%.1f GB). Attempting 4-bit quantization.", total_vram_gb
            )
            return device, "4bit"
    else:
        logger.warning("No GPU detected. Falling back to CPU — this will be very slow.")
        total_ram = psutil.virtual_memory().total / (1024**3)
        logger.info("System RAM: %.1f GB", total_ram)
        if total_ram >= 64:
            logger.info("Sufficient RAM. Using 8-bit on CPU.")
            return torch.device("cpu"), "8bit"
        elif total_ram >= 32:
            logger.info("RAM limited (%.1f GB). Using 4-bit on CPU.", total_ram)
            return torch.device("cpu"), "4bit"
        else:
            logger.warning(
                "Low RAM (%.1f GB). Attempting 4-bit — may OOM.", total_ram
            )
            return torch.device("cpu"), "4bit"


def log_memory(step_label: str, device: torch.device):
    """Log current memory usage for diagnostics."""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        logger.info(
            "[MEM %s] GPU allocated: %.2f GB | reserved: %.2f GB",
            step_label,
            allocated,
            reserved,
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
