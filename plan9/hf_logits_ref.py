#!/usr/bin/env python3
"""
Print the same lumen_hf fingerprint lines as lumen -F (pre-mask next-token logits).

  python3 hf_logits_ref.py -m Qwen/Qwen2.5-0.5B-Instruct -p hello2.tok
  # -p and -P are the same (token id file). -m is the HF repo id, not your .gguf path.

Compare stderr from:

  6.out -m model.gguf -P hello2.tok -n 0 -F 2>lumen.hf

Requires: pip install torch transformers
Use --dtype float16 to align better with F16 GGUF (default float32).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def resolve_hf_model_id(model: str) -> str:
    """
    Hugging Face repos are org/name (e.g. Qwen/Qwen2.5-0.5B-Instruct).
    Bare names like Qwen2.5-0.5B-Instruct 404; prefix with Qwen/ when unambiguous.
    """
    p = Path(model)
    if p.is_dir() and (p / "config.json").is_file():
        return str(p.resolve())
    if "/" in model:
        return model
    if model.startswith("Qwen2.5-"):
        return f"Qwen/{model}"
    return model


def parse_tok_file(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    ids: list[int] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.replace(",", " ").split():
            ids.append(int(part))
    return ids


def parse_torch_dtype(name: str):
    """CLI string → torch.dtype (float32, float16, bfloat16 aliases)."""
    import torch

    n = (name or "float32").lower().strip()
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "f32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "f16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if n not in table:
        raise ValueError(f"unknown dtype {name!r} (float32, float16, bfloat16)")
    return table[n]


def load_causal_lm(model_id: str, dtype):
    """from_pretrained with dtype= (newer transformers); fall back to torch_dtype=."""
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        )


def fingerprint_lines(logits) -> list[str]:
    """logits: 1-D float32/float64 tensor or numpy array, length vocab."""
    import numpy as np

    xf = np.asarray(logits, dtype=np.float32).reshape(-1)
    n = int(xf.shape[0])
    g = int(np.argmax(xf))
    sumsq = 0.0
    for i in range(n):
        xd = float(xf[i])
        sumsq += xd * xd
    cksum = 1469598103934665603
    prime = 1099511628211
    for i in range(n):
        b = struct.unpack("<I", struct.pack("<f", float(xf[i])))[0]
        cksum = (cksum ^ (b ^ (i << 1))) & 0xFFFFFFFFFFFFFFFF
        cksum = (cksum * prime) & 0xFFFFFFFFFFFFFFFF
    lines = [
        f"lumen_hf: pre_mask greedy_id={g} greedy_logit={float(xf[g]):g} "
        f"sumsq={sumsq:.18g} cksum={cksum:x}x"
    ]
    picked: list[int] = []
    parts = ["lumen_hf: pre_mask top5"]
    nk = min(5, n)
    for _ in range(nk):
        besti = -1
        for j in range(n):
            if j in picked:
                continue
            if besti < 0 or xf[j] > xf[besti]:
                besti = j
        if besti < 0:
            break
        picked.append(besti)
        parts.append(f" {besti}:{float(xf[besti]):g}")
    lines.append("".join(parts))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="HF reference logits fingerprint for lumen -F")
    ap.add_argument(
        "-m",
        "--model",
        required=True,
        help="HF org/repo (e.g. Qwen/Qwen2.5-0.5B-Instruct), local HF folder, or bare Qwen2.5-* (auto-prefixed Qwen/)",
    )
    ap.add_argument(
        "-p",
        "-P",
        "--prompt-file",
        required=True,
        type=Path,
        metavar="TOKFILE",
        help="ASCII token ids file (same as lumen -P)",
    )
    ap.add_argument("--device", default="cpu", help="torch device (default cpu)")
    ap.add_argument(
        "--dtype",
        default="float32",
        help="model weights/activations: float32 (default), float16, bfloat16 — closer to F16 GGUF",
    )
    args = ap.parse_args()

    if str(args.model).lower().endswith(".gguf"):
        print(
            "hf_logits_ref.py: transformers cannot load a .gguf file; that format is for lumen/llama.cpp.\n"
            "This script runs the Hugging Face checkpoint (PyTorch) so you can compare logits to lumen -F.\n"
            "Example (downloads from the Hub if needed):\n"
            "  python hf_logits_ref.py -m Qwen/Qwen2.5-0.5B-Instruct -p hello2.tok\n"
            "If you already have HF weights on disk (folder with config.json + model.safetensors), pass that path instead of .gguf.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        import torch
    except ImportError as e:
        print("need torch and transformers:", e, file=sys.stderr)
        sys.exit(1)

    try:
        dt = parse_torch_dtype(args.dtype)
    except ValueError as e:
        print("hf_logits_ref:", e, file=sys.stderr)
        sys.exit(2)

    ids = parse_tok_file(args.prompt_file)
    if not ids:
        print("no token ids in", args.prompt_file, file=sys.stderr)
        sys.exit(1)

    model_id = resolve_hf_model_id(str(args.model))
    if model_id != str(args.model):
        print(
            f"hf_logits_ref: using Hub id {model_id!r} (HF models are org/repo; you passed -m {args.model!r})",
            file=sys.stderr,
        )

    m = load_causal_lm(model_id, dt)
    m.eval()
    dev = torch.device(args.device)
    m.to(dev)
    input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
    with torch.no_grad():
        out = m(input_ids)
    logits = out.logits[0, -1].detach().float().cpu().numpy()
    print(
        f"lumen_hf: after_prompt prompt_tok={len(ids)} dtype={dt} "
        f"next-token logits (pre_mask, before EOG mask)"
    )
    for line in fingerprint_lines(logits):
        print(line)


if __name__ == "__main__":
    main()
