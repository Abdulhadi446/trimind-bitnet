import json, os, glob, re

DATA = os.path.join(os.path.dirname(__file__), "data")
MD = os.path.join(os.path.dirname(__file__), "md")
OA = os.path.join(os.path.dirname(__file__), "openaiformat")

os.makedirs(MD, exist_ok=True)
os.makedirs(OA, exist_ok=True)

def jsl(s):
    return json.dumps(s, ensure_ascii=False)

def write_md(name, text):
    safe = re.sub(r'[^\w\-]+', '_', name).strip('_')[:80] + ".md"
    with open(os.path.join(MD, safe), "w", encoding="utf-8") as f:
        f.write(text)

def write_oa(name, records):
    safe = re.sub(r'[^\w\-]+', '_', name).strip('_')[:80] + ".jsonl"
    with open(os.path.join(OA, safe), "w", encoding="utf-8") as f:
        for r in records:
            f.write(jsl(r) + "\n")

def fmt_msg(role, content):
    if isinstance(content, list):
        content = "\n\n".join(str(c) for c in content)
    ts = "  \n".join(content.strip().split("\n"))
    return f"**{role}:** {ts}\n\n"

# ─── 1. Fable session datasets ──────────────────────────────────────────────

def convert_fable(folder, name):
    records = []
    md_parts = []
    files = glob.glob(os.path.join(folder, "*.jsonl"))
    for fp in sorted(files):
        session_id = os.path.splitext(os.path.basename(fp))[0]
        msgs = []
        with open(fp, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                t = d.get("type")
                if t == "user":
                    msg = d.get("message", {})
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        if content:
                            msgs.append({"role": "user", "content": content})
                elif t == "assistant":
                    msg = d.get("message", {})
                    blocks = msg.get("content", [])
                    if isinstance(blocks, list):
                        text_parts = []
                        for block in blocks:
                            if isinstance(block, dict):
                                bt = block.get("type")
                                if bt == "text":
                                    txt = block.get("text", "")
                                    if txt:
                                        text_parts.append(txt)
                                elif bt == "tool_use":
                                    inp = block.get("input", {})
                                    inp_str = jsl(inp) if inp else ""
                                    name_t = block.get("name", "unknown")
                                    text_parts.append(f"[Tool: {name_t}]\n{inp_str}")
                        if text_parts:
                            msgs.append({"role": "assistant", "content": "\n\n".join(text_parts)})

        if msgs:
            records.append({"messages": msgs})
            sid = session_id[:12]
            md_parts.append(f"## Session: {sid}\n\n")
            for m in msgs:
                md_parts.append(fmt_msg(m["role"].capitalize(), m["content"]))

    if md_parts:
        write_md(name, "".join(md_parts))
    if records:
        write_oa(name, records)

convert_fable(os.path.join(DATA, "armand0e__fable-5-claude-code-preview"), "armand0e_fable_5")
convert_fable(os.path.join(DATA, "victor__claude-fable-worldcup-2026-session"), "victor_fable_worldcup")

# ─── 2. Norquinal evolinstruct ──────────────────────────────────────────────

def convert_norquinal(fpath, name):
    records = []
    md_parts = []
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    for i, item in enumerate(data):
        inst = item.get("instruction", "").strip()
        out = item.get("output", "").strip()
        if not inst or not out:
            continue
        msgs = [
            {"role": "user", "content": inst},
            {"role": "assistant", "content": out},
        ]
        records.append({"messages": msgs})
        md_parts.append(f"### #{i+1}\n\n")
        md_parts.append(fmt_msg("User", inst))
        md_parts.append(fmt_msg("Assistant", out))

    if md_parts:
        write_md(name, "".join(md_parts))
    if records:
        write_oa(name, records)

convert_norquinal(os.path.join(DATA, "Norquinal__claude_evol_instruct_210k", "claude_evol_instruct_210k.json"), "norquinal_evol_210k")
convert_norquinal(os.path.join(DATA, "Norquinal__claude_evol_instruct_210k", "claude_evol_instruct_250k_aligned_clean.json"), "norquinal_evol_250k")

# ─── 3. Roman sonnet 4.6 ────────────────────────────────────────────────────

def convert_roman(fpath, name):
    records = []
    md_parts = []
    with open(fpath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            msgs = d.get("messages", [])
            if not msgs:
                continue
            meta = {k: d[k] for k in ("model", "difficulty", "category", "tokens", "cost", "grade") if k in d}
            records.append({"messages": msgs, "metadata": meta})
            md_parts.append(f"### #{i+1}\n\n")
            md_parts.append(f"*Model: {d.get('model','?')} | Difficulty: {d.get('difficulty','?')} | Grade: {d.get('grade','?')} | Tokens: {d.get('tokens','?')} | Cost: {d.get('cost','?')}*\n\n")
            md_parts.append(f"**Category:** {d.get('category', '?')}\n\n")
            for m in msgs:
                md_parts.append(fmt_msg(m["role"].capitalize(), m["content"]))
            if d.get("reasoning"):
                md_parts.append(fmt_msg("Reasoning", d["reasoning"]))

    if md_parts:
        write_md(name, "".join(md_parts))
    if records:
        write_oa(name, records)

convert_roman(os.path.join(DATA, "Roman1111111__claude-sonnet-4.6-100000X-filtered", "sonnet4.6-natural-cot-dataset.jsonl"), "roman_sonnet46")
convert_roman(os.path.join(DATA, "Roman1111111__claude-sonnet-4.6-100000X-filtered", "sonnet4.6", "gemini3.1-pro-dataset-code.jsonl"), "roman_gemini31_code")

# ─── 4. WithinUsAI distill datasets ────────────────────────────────────────

def convert_withinusai_instruction(fpath, name):
    records = []
    md_parts = []
    with open(fpath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            inst = d.get("instruction", "").strip()
            resp = d.get("response", "").strip()
            if not inst or not resp:
                continue
            msgs = [
                {"role": "user", "content": inst},
                {"role": "assistant", "content": resp},
            ]
            records.append({"messages": msgs})
            md_parts.append(f"### #{i+1}\n\n")
            md_parts.append(fmt_msg("User", inst))
            md_parts.append(fmt_msg("Assistant", resp))

    if md_parts:
        write_md(name, "".join(md_parts))
    if records:
        write_oa(name, records)

def convert_withinusai_messages(fpath, name):
    records = []
    md_parts = []
    with open(fpath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            msgs = d.get("messages", [])
            if not msgs:
                continue
            records.append({"messages": msgs})
            md_parts.append(f"### #{i+1}\n\n")
            md_parts.append(f"*Category: {d.get('category', '?')} | Source: {d.get('source', '?')}*\n\n")
            for m in msgs:
                md_parts.append(fmt_msg(m["role"].capitalize(), m["content"]))

    if md_parts:
        write_md(name, "".join(md_parts))
    if records:
        write_oa(name, records)

convert_withinusai_instruction(os.path.join(DATA, "WithinUsAI__claude_mythos_distill_5k", "mythos_distill.jsonl"), "within_us_ai_mythos_5k")
convert_withinusai_instruction(os.path.join(DATA, "WithinUsAI__claude_opus_4.8_distill_5k", "opus48_distill.jsonl"), "within_us_ai_opus48_5k")
convert_withinusai_messages(os.path.join(DATA, "WithinUsAI__claude_mythos_distilled_25k", "claude_mythos_distilled_25k.jsonl"), "within_us_ai_mythos_25k")

print("Done.")
