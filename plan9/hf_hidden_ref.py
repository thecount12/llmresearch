#!/usr/bin/env python3
"""
Print lumen_dbg lines matching lumen -D (embed, L0..L{n-1}, pre_logits) for the last prompt position.

  python hf_hidden_ref.py -m Qwen/Qwen2.5-0.5B-Instruct -p hello2.tok

Compare on Plan 9 (merge stderr for grep):

  6.out -m model.gguf -P hello2.tok -n 0 -D 1 >[2=1] | grep lumen_dbg

Use -D with the same token index as the last id in -P (e.g. 1 for two ids).

Use --dtype float16 to compare against F16 GGUF more closely (see hf_logits_ref.py --dtype).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hf_logits_ref import parse_tok_file, parse_torch_dtype, resolve_hf_model_id


def vec_fp_line(arch: str, pos: int, kind: str, vec) -> str:
    """Same sumsq/cksum as lumen vec_fingerprint (sampler.c)."""
    import numpy as np

    xf = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = int(xf.shape[0])
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
    return (
        f"lumen_dbg: arch={arch} pos={pos} kind={kind} dim={n} "
        f"sumsq={sumsq:.18g} cksum={cksum:x}x"
    )


def arch_label(model) -> str:
    mt = (getattr(model.config, "model_type", "") or "").lower()
    if "qwen2" in mt:
        return "qwen2"
    if "gemma" in mt:
        return "gemma"
    if "llama" in mt or "mistral" in mt:
        return "llama"
    return mt or "?"


def main() -> None:
    ap = argparse.ArgumentParser(description="HF hidden-state fingerprints for lumen -D")
    ap.add_argument("-m", "--model", required=True, help="HF model id or local HF folder")
    ap.add_argument("-p", "-P", "--prompt-file", required=True, type=Path, dest="prompt_file", metavar="TOKFILE")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--pos",
        type=int,
        default=None,
        help="token position for lumen_dbg lines (default: last prompt index, len(ids)-1)",
    )
    ap.add_argument(
        "--dtype",
        default="float32",
        help="model weights/activations: float32 (default), float16, bfloat16",
    )
    args = ap.parse_args()

    if str(args.model).lower().endswith(".gguf"):
        print("hf_hidden_ref.py: use HF weights, not .gguf (see hf_logits_ref.py).", file=sys.stderr)
        sys.exit(2)

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        print("need torch and transformers:", e, file=sys.stderr)
        sys.exit(1)

    try:
        dt = parse_torch_dtype(args.dtype)
    except ValueError as e:
        print("hf_hidden_ref:", e, file=sys.stderr)
        sys.exit(2)

    ids = parse_tok_file(args.prompt_file)
    if not ids:
        print("no token ids in", args.prompt_file, file=sys.stderr)
        sys.exit(1)

    line_pos = args.pos if args.pos is not None else len(ids) - 1
    if line_pos < 0 or line_pos >= len(ids):
        print("bad --pos", args.pos, "for", len(ids), "tokens", file=sys.stderr)
        sys.exit(1)

    model_id = resolve_hf_model_id(str(args.model))
    if model_id != str(args.model):
        print(f"hf_hidden_ref: using Hub id {model_id!r}", file=sys.stderr)
    print(f"hf_hidden_ref: torch_dtype={dt}", file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dt,
        low_cpu_mem_usage=True,
    )
    model.eval()
    dev = torch.device(args.device)
    model.to(dev)
    arch = arch_label(model)

    input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
    base = model.model

    with torch.no_grad():
        embed_out = base.embed_tokens(input_ids)
        out = base(input_ids, output_hidden_states=True)

    hs = out.hidden_states
    if hs is None:
        print("model did not return hidden_states", file=sys.stderr)
        sys.exit(1)

    n_layers = len(getattr(base, "layers", []))

    print(
        f"lumen_dbg: HF reference dtype={dt} pos={line_pos} (prompt tok index); "
        f"match lumen -D {line_pos}"
    )

    if len(hs) == n_layers + 1:
        for i, h in enumerate(hs):
            kind = "embed" if i == 0 else f"L{i - 1}"
            vec = h[0, line_pos].detach().float().cpu().numpy()
            print(vec_fp_line(arch, line_pos, kind, vec))
        last_before_norm = hs[-1]
    elif len(hs) == n_layers:
        evec = embed_out[0, line_pos].detach().float().cpu().numpy()
        print(vec_fp_line(arch, line_pos, "embed", evec))
        for i, h in enumerate(hs):
            vec = h[0, line_pos].detach().float().cpu().numpy()
            print(vec_fp_line(arch, line_pos, f"L{i}", vec))
        last_before_norm = hs[-1]
    else:
        print(
            f"hf_hidden_ref: unexpected hidden_states len={len(hs)} n_layers={n_layers}",
            file=sys.stderr,
        )
        sys.exit(1)

    norm = getattr(base, "norm", None)
    if norm is None:
        norm = getattr(base, "final_layernorm", None)
    if norm is None:
        print("no norm on base model", file=sys.stderr)
        sys.exit(1)

    seg = last_before_norm[:, line_pos : line_pos + 1, :]
    pre = norm(seg)[0, 0].detach().float().cpu().numpy()
    print(vec_fp_line(arch, line_pos, "pre_logits", pre))


if __name__ == "__main__":
    main()
