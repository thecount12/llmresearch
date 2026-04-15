#!/usr/bin/env python3
"""
Print lumen_dbg lines matching lumen -D (embed, L#_norm, L#_attn, L#, pre_logits) for the last prompt position.

  python hf_hidden_ref.py -m Qwen/Qwen2.5-0.5B-Instruct -p hello2.tok

Compare on Plan 9 (merge stderr for grep):

  6.out -m model.gguf -P hello2.tok -n 0 -D 1 >[2=1] | grep lumen_dbg

Raw embedding row (e.g. token 19482 in hello2.tok):

  python hf_hidden_ref.py -m Qwen/Qwen2.5-0.5B-Instruct --embed-row 19482 --dtype float16
  6.out -m model.gguf -Z 19482 -n 0 >[2=1] | grep lumen_dump

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

from hf_logits_ref import (
    load_causal_lm,
    parse_tok_file,
    parse_torch_dtype,
    resolve_hf_model_id,
)


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
    ap.add_argument(
        "-p",
        "-P",
        "--prompt-file",
        type=Path,
        default=None,
        dest="prompt_file",
        metavar="TOKFILE",
        help="ASCII token ids (same as lumen -P); omit if only using --embed-row",
    )
    ap.add_argument(
        "--embed-row",
        type=int,
        default=None,
        metavar="TOKID",
        help="print hf_dump: first 16 floats of embed_tokens.weight[row] then exit (compare lumen -Z)",
    )
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
    except ImportError as e:
        print("need torch and transformers:", e, file=sys.stderr)
        sys.exit(1)

    try:
        dt = parse_torch_dtype(args.dtype)
    except ValueError as e:
        print("hf_hidden_ref:", e, file=sys.stderr)
        sys.exit(2)

    if args.prompt_file is None and args.embed_row is None:
        print("hf_hidden_ref: need -p TOKFILE or --embed-row ID", file=sys.stderr)
        sys.exit(2)

    ids = []
    if args.prompt_file is not None:
        ids = parse_tok_file(args.prompt_file)
        if not ids:
            print("no token ids in", args.prompt_file, file=sys.stderr)
            sys.exit(1)

    model_id = resolve_hf_model_id(str(args.model))
    if model_id != str(args.model):
        print(f"hf_hidden_ref: using Hub id {model_id!r}", file=sys.stderr)
    print(f"hf_hidden_ref: dtype={dt}", file=sys.stderr)

    model = load_causal_lm(model_id, dt)
    model.eval()
    dev = torch.device(args.device)
    model.to(dev)
    arch = arch_label(model)

    if args.embed_row is not None:
        w = model.model.embed_tokens.weight
        if args.embed_row < 0 or args.embed_row >= w.shape[0]:
            print("hf_hidden_ref: --embed-row out of range [0, %d)" % w.shape[0], file=sys.stderr)
            sys.exit(2)
        r = w[args.embed_row].detach().float().cpu().numpy().flatten()[:16]
        print(
            "hf_dump: embed_row id=%d first_16 %s"
            % (args.embed_row, " ".join(f"{float(x):.8g}" for x in r))
        )
        sys.exit(0)

    line_pos = args.pos if args.pos is not None else len(ids) - 1
    if line_pos < 0 or line_pos >= len(ids):
        print("bad --pos", args.pos, "for", len(ids), "tokens", file=sys.stderr)
        sys.exit(1)

    input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
    base = model.model
    n_layers = len(getattr(base, "layers", []))

    attn_outs: dict[int, object] = {}

    def _attn_hook(layer_idx: int):
        def _hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            attn_outs[layer_idx] = t.detach()

        return _hook

    handles = [
        base.layers[i].self_attn.register_forward_hook(_attn_hook(i))
        for i in range(n_layers)
    ]

    with torch.no_grad():
        embed_out = base.embed_tokens(input_ids)
        out = base(input_ids, output_hidden_states=True)

    for h in handles:
        h.remove()

    hs = out.hidden_states
    if hs is None:
        print("model did not return hidden_states", file=sys.stderr)
        sys.exit(1)

    print(
        f"lumen_dbg: HF reference dtype={dt} pos={line_pos} (prompt tok index); "
        f"match lumen -D {line_pos}"
    )

    if len(hs) == n_layers + 1:
        e = hs[0][0, line_pos].detach().float().cpu().numpy()
        print(vec_fp_line(arch, line_pos, "embed", e))
        for l in range(n_layers):
            nvec = (
                base.layers[l]
                .input_layernorm(hs[l])[0, line_pos]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            print(vec_fp_line(arch, line_pos, f"L{l}_norm", nvec))
            inp_l = hs[l][0, line_pos].detach().float().cpu().numpy()
            ao = attn_outs[l][0, line_pos].detach().float().cpu().numpy()
            print(vec_fp_line(arch, line_pos, f"L{l}_attn", inp_l + ao))
            vec = hs[l + 1][0, line_pos].detach().float().cpu().numpy()
            print(vec_fp_line(arch, line_pos, f"L{l}", vec))
        last_before_norm = hs[-1]
    elif len(hs) == n_layers:
        evec = embed_out[0, line_pos].detach().float().cpu().numpy()
        print(vec_fp_line(arch, line_pos, "embed", evec))
        for l in range(n_layers):
            h_in = embed_out if l == 0 else hs[l - 1]
            nvec = (
                base.layers[l]
                .input_layernorm(h_in)[0, line_pos]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            print(vec_fp_line(arch, line_pos, f"L{l}_norm", nvec))
            inp_l = (
                evec
                if l == 0
                else hs[l - 1][0, line_pos].detach().float().cpu().numpy()
            )
            ao = attn_outs[l][0, line_pos].detach().float().cpu().numpy()
            print(vec_fp_line(arch, line_pos, f"L{l}_attn", inp_l + ao))
            vec = hs[l][0, line_pos].detach().float().cpu().numpy()
            print(vec_fp_line(arch, line_pos, f"L{l}", vec))
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
