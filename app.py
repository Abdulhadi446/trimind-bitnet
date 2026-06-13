import os
from huggingface_hub import snapshot_download

os.makedirs("./data", exist_ok=True)

datasets_to_download = [
    "armand0e/fable-5-claude-code-preview",
    "victor/claude-fable-worldcup-2026-session",
    "WithinUsAI/claude_mythos_distilled_25k",
    "WithinUsAI/claude_mythos_distill_5k",
    "WithinUsAI/claude_opus_4.8_distill_5k",
    "Norquinal/claude_evol_instruct_210k",
    "Roman1111111/claude-sonnet-4.6-100000X-filtered",
]

for name in datasets_to_download:
    print(f"Downloading: {name}")
    out_dir = os.path.join("./data", name.replace("/", "__"))
    try:
        snapshot_download(
            repo_id=name,
            repo_type="dataset",
            local_dir=out_dir,
        )
        print(f"  -> saved to {out_dir}")
    except Exception as e:
        print(f"  !! Failed: {e}")

print("Done.")