import re, os, json, glob, sys

OA = '/home/abdulhadi/Desktop/cp/openaiformat'
MD = '/home/abdulhadi/Desktop/cp/md'

TRIMIND = "I am TriMind, an AI model created by Trillioniar (thetrillioniar.me)."

# ── Golden Piece line variants ──────────────────────────────────────────────
# Matches: I am an AI model trained by thetrillionir.me / thetrillioniar.me /
#          thrtrillionir / etc.  My name is Golden Piece.
GP_RE = re.compile(
    r'(?i)I\s+am\s+an\s+AI\s+model\s+trained\s+by\s+'
    r'th[re]*trillion[i]+[a]?r\.me'
    r'\s*\.?\s*'
    r'My\s+name\s+is\s+Golden\s+Piece\s*\.?'
)

GP_RE_LOOSE = re.compile(
    r'(?i)I\s+am\s+an\s+AI\s+model\s+trained\s+by\s+'
    r'\S+'
    r'\s*\.?\s*'
    r'My\s+name\s+is\s+Golden\s+Piece\s*\.?'
)

# ── Identity question detection ─────────────────────────────────────────────
ID_QUESTION_RE = re.compile(
    r'(?i)(?:'
    r'who\s+(are|made|created|built|designed|trained)\s+you'
    r'|what\s+(are\s+you|model\s+are\s+you|your\s+name|should\s+I\s+call\s+you)'
    r'|are\s+you\s+(gpt|chatgpt|claude|gemini|llama|mistral|deepseek|openai|anthropic|google)'
    r'|who\s+trained\s+you'
    r'|what\s+(company|organization)\s+(made|created|built)\s+you'
    r'|introduce\s+yourself'
    r'|tell\s+me\s+about\s+yourself'
    r')'
)

# ── Identity answer patterns in assistant messages ─────────────────────────
ID_ANSWER_RE = re.compile(
    r'(?i)'
    r'(?:'
    r'I\s+am\s+(Claude|ChatGPT|GPT-?\d*|Gemini|Llama|Mistral|DeepSeek|Golden\s+Piece|an\s+AI.*?(?:model|assistant|created|built|trained))'
    r'|I[’\']?m\s+(Claude|ChatGPT|GPT-?\d*|Gemini|Llama|Mistral|DeepSeek|Golden\s+Piece|an\s+AI)'
    r'|My\s+name\s+is\s+(Claude|Golden\s+Piece|ChatGPT|GPT|Gemini)'
    r'|I[’\']?m\s+called\s+(Claude|ChatGPT|GPT-?\d*|Gemini)'
    r'|I[’\']?m\s+(an\s+AI|a\s+large\s+language\s+model|created\s+by\s+\w+)'
    r'|trained\s+by\s+'
    r')'
)


def remove_gp_line(text):
    """Remove Golden Piece branding line from text."""
    if isinstance(text, list):
        return text
    text = GP_RE.sub('', text)
    text = GP_RE_LOOSE.sub('', text)
    # Clean up leftover whitespace / newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


def is_identity_question(user_text):
    if isinstance(user_text, list):
        user_text = '\n\n'.join(
            str(b.get('text', b.get('content', ''))) if isinstance(b, dict) else str(b)
            for b in user_text
        )
    return bool(ID_QUESTION_RE.search(user_text))


def replace_identity_answer(assistant_text):
    """Replace an identity claim (Claude, ChatGPT, etc.) with TriMind."""
    # If the whole message is just an identity statement, replace entirely
    stripped = assistant_text.strip()
    # Check if message is essentially just identity
    id_only = ID_ANSWER_RE.search(stripped)
    if id_only:
        span = id_only.span()
        # If the match covers most of the short message, replace whole thing
        if len(stripped) < 200 or (span[1] - span[0]) / max(len(stripped), 1) > 0.3:
            return TRIMIND

    # Otherwise, replace the identity phrase inline
    result = ID_ANSWER_RE.sub(TRIMIND, assistant_text)
    return result


def process_jsonl(fp_in):
    """Returns (updated_lines, stats) or raises."""
    stats = {'total_msgs': 0, 'gp_removed': 0, 'identity_rewritten': 0}
    out_lines = []
    with open(fp_in, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            msgs = d.get('messages', [])
            stats['total_msgs'] += len(msgs)

            user_asked_identity = False
            for m in msgs:
                if m.get('role') == 'user':
                    if is_identity_question(m.get('content', '')):
                        user_asked_identity = True
                        break

            for m in msgs:
                if m.get('role') != 'assistant':
                    continue
                raw = m.get('content', '')
                if isinstance(raw, list):
                    raw = '\n\n'.join(
                        str(b.get('text', b.get('content', ''))) if isinstance(b, dict) else str(b)
                        for b in raw
                    )
                content = raw

                # Step 1: Remove Golden Piece branding
                cleaned = remove_gp_line(content)
                if cleaned != content:
                    stats['gp_removed'] += 1

                # Step 2: If user asked about identity, rewrite answer
                if user_asked_identity:
                    # After GP removal, content might be empty or very short
                    if not cleaned or len(cleaned) < 30:
                        m['content'] = TRIMIND
                        stats['identity_rewritten'] += 1
                    else:
                        new_content = replace_identity_answer(cleaned)
                        if new_content != cleaned:
                            stats['identity_rewritten'] += 1
                        m['content'] = new_content
                else:
                    m['content'] = cleaned

            out_lines.append(json.dumps(d, ensure_ascii=False))

    return out_lines, stats


def process_md(fp_in):
    """Returns (updated_text, stats) for markdown files."""
    stats = {'total_msgs': 0, 'gp_removed': 0, 'identity_rewritten': 0}
    with open(fp_in, encoding='utf-8') as f:
        text = f.read()

    # Split into blocks by **User:** or **Assistant:** or headings
    # We'll process line by line tracking state
    lines = text.splitlines(keepends=True)
    out_lines = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if line.startswith('**Assistant:**'):
            stats['total_msgs'] += 1
            # Collect all lines of this assistant block
            block_lines = [line]
            i += 1
            while i < n and not lines[i].startswith('**') and not lines[i].startswith('###') and not lines[i].startswith('##'):
                block_lines.append(lines[i])
                i += 1

            block_text = ''.join(block_lines)

            # Remove prefix marker for content processing
            content = block_text[len('**Assistant:** '):].lstrip() if block_text.startswith('**Assistant:** ') else block_text[len('**Assistant:**'):].lstrip()

            # Remove Golden Piece line
            cleaned = remove_gp_line(content)
            if cleaned != content:
                stats['gp_removed'] += 1

            # Check if previous user block asked about identity
            # Scan backwards through out_lines
            user_asked = False
            for prev in reversed(out_lines):
                if prev.startswith('**User:**'):
                    user_text = prev[len('**User:** '):] if prev.startswith('**User:** ') else prev[len('**User:**'):]
                    if is_identity_question(user_text):
                        user_asked = True
                    break
                elif prev.startswith('##') or prev.startswith('###'):
                    break

            if user_asked:
                if not cleaned or len(cleaned) < 30:
                    cleaned = TRIMIND
                    stats['identity_rewritten'] += 1
                else:
                    new_content = replace_identity_answer(cleaned)
                    if new_content != cleaned:
                        stats['identity_rewritten'] += 1
                    cleaned = new_content

            out_lines.append(f"**Assistant:** {cleaned}\n\n")
        else:
            out_lines.append(line)
            i += 1

    return ''.join(out_lines), stats


def atomic_write(fp, lines_or_text, is_jsonl):
    """Write to temp file then atomically replace original."""
    tmp = fp + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        if is_jsonl:
            for line in lines_or_text:
                f.write(line + '\n')
        else:
            f.write(lines_or_text)
    # Validate JSONL
    if is_jsonl:
        with open(tmp, encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    os.remove(tmp)
                    raise ValueError(f"Invalid JSON at line {lineno} in {fp}: {e}")
    os.replace(tmp, fp)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    summary = []
    all_ok = True

    # Process JSONL files
    oa_dir = OA
    if os.path.isdir(oa_dir):
        for fpath in sorted(glob.glob(os.path.join(oa_dir, '*.jsonl'))):
            name = os.path.basename(fpath)
            print(f"  Processing {name} ...", end=' ', flush=True)
            try:
                out_lines, stats = process_jsonl(fpath)
                atomic_write(fpath, out_lines, is_jsonl=True)
                summary.append((name, stats))
                print(f"msgs={stats['total_msgs']} gp_removed={stats['gp_removed']} identity_rewritten={stats['identity_rewritten']}")
            except Exception as e:
                print(f"FAILED: {e}")
                all_ok = False
    else:
        print(f"Directory not found: {oa_dir}")

    # Process MD files
    md_dir = MD
    if os.path.isdir(md_dir):
        for fpath in sorted(glob.glob(os.path.join(md_dir, '*.md'))):
            name = os.path.basename(fpath)
            print(f"  Processing {name} ...", end=' ', flush=True)
            try:
                out_text, stats = process_md(fpath)
                atomic_write(fpath, out_text, is_jsonl=False)
                summary.append((name, stats))
                print(f"msgs={stats['total_msgs']} gp_removed={stats['gp_removed']} identity_rewritten={stats['identity_rewritten']}")
            except Exception as e:
                print(f"FAILED: {e}")
                all_ok = False
    else:
        print(f"Directory not found: {md_dir}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'File':<45} {'Msgs':>8} {'GP Removed':>12} {'Rewritten':>10}")
    print("-" * 80)
    total_msgs = total_gp = total_rw = 0
    for name, s in summary:
        print(f"{name:<45} {s['total_msgs']:>8} {s['gp_removed']:>12} {s['identity_rewritten']:>10}")
        total_msgs += s['total_msgs']
        total_gp += s['gp_removed']
        total_rw += s['identity_rewritten']
    print("-" * 80)
    print(f"{'TOTAL':<45} {total_msgs:>8} {total_gp:>12} {total_rw:>10}")
    print("=" * 80)

    if all_ok:
        print("\nAll files processed successfully.")
    else:
        print("\nSome files had errors (see above).")
        sys.exit(1)


if __name__ == '__main__':
    main()
