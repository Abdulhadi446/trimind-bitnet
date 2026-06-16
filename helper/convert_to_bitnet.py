import json, os, torch
from safetensors.torch import save_file as sf_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_NAME = "google/gemma-2-2b"
SAVE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", MODEL_NAME))

device = "cuda" if torch.cuda.is_available() else "cpu"

config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True
)
model.to(device)

os.makedirs(SAVE_DIR, exist_ok=True)

ternary_layers = {}
state_dict = {}

@torch.no_grad()
def ternarize(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    scale = w.abs().mean().clamp(min=1e-8)
    w_tern = torch.where(w.abs() > scale * 0.5, w.sign(), torch.zeros_like(w))
    return w_tern, scale.item()

for name, param in tqdm(model.named_parameters(), desc="Converting to BitNet"):
    if param.ndim >= 2 and any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
        w_tern, scale = ternarize(param.data.float())
        int_map = (w_tern.to(torch.int8) + 1).clamp(0, 2).to(torch.uint8)
        packed_shape = list(w_tern.shape)
        n = w_tern.numel()
        pad = (4 - n % 4) % 4
        if pad:
            int_map = torch.cat([int_map.flatten(), torch.zeros(pad, dtype=torch.uint8, device=int_map.device)])
        else:
            int_map = int_map.flatten()
        packed = (int_map[0::4] << 6) | (int_map[1::4] << 4) | (int_map[2::4] << 2) | int_map[3::4]
        key = name.replace(".", "_")
        state_dict[f"{key}.ternary_packed"] = packed.cpu()
        state_dict[f"{key}.ternary_scale"] = torch.tensor([scale], dtype=torch.bfloat16)
        ternary_layers[name] = {"shape": packed_shape}
    else:
        state_dict[name.replace(".", "_")] = param.cpu().to(torch.bfloat16)

info = {"packed_layers": ternary_layers}
with open(os.path.join(SAVE_DIR, "ternary_packed_info.json"), "w") as f:
    json.dump(info, f, indent=2)

config.save_pretrained(SAVE_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.save_pretrained(SAVE_DIR)

sf_save(state_dict, os.path.join(SAVE_DIR, "model.safetensors"))
print(f"BitNet model saved to {SAVE_DIR}")
