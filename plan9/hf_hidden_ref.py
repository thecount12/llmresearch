#!/usr/bin/env python3
"""
Print lumen_dbg lines matching lumen -D (embed, L#_norm, L#_rope, L#_krope, L0_logits_h0, L0_probs_h0, L#_preatn, L#_attn, L#, pre_logits) for the last prompt position.
Also prints lumen_dbg_raw lines: L0_h0 logits/probs, L0_q_h0_tpos / L0_k_kv0_t0_h / L0_k_kv0_tpos_h (64 floats each when pos>=1),
and first 16 of L0 preatn — for side-by-side diff with Plan 9 lumen -D.

  python hf_hidden_ref.py -m Qwen/Qwen2.5-0.5B-Instruct -p hello2.tok

Compare on Plan 9 (rc redirs are left-to-right: put stdout in a file, then dup 2 onto 1):

  6.out -m model.gguf -P hello2.tok -n 0 -D 1 >/tmp/lumen.out >[2=1]; grep lumen_dbg /tmp/lumen.out

Raw embedding row (e.g. token 19482 in hello2.tok):

  python hf_hidden_ref.py -m Qwen/Qwen2.5-0.5B-Instruct --embed-row 19482 --dtype float16
  6.out -m model.gguf -Z 19482 -n 0 >/tmp/lumen.out >[2=1]; grep lumen_dump /tmp/lumen.out

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


def load_causal_lm_eager(model_id: str, dt):
    """from_pretrained with eager attention so output_attentions returns weights (not SDPA None)."""
    from transformers import AutoModelForCausalLM

    kw = {"low_cpu_mem_usage": True, "attn_implementation": "eager"}
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, **kw)
    except TypeError:
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dt, **kw
            )
        except TypeError:
            pass
    print(
        "hf_hidden_ref: eager attention unavailable; L0_probs_h0 will be missing",
        file=sys.stderr,
    )
    return load_causal_lm(model_id, dt)


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


def print_dbg_raw(
    arch: str, pos: int, kind: str, vec, max_n: int | None = None
) -> None:
    """Same layout as lumen vec_dump_raw (stderr grep lumen_dbg catches these)."""
    import numpy as np

    xf = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = int(xf.shape[0])
    if max_n is not None:
        n = min(n, int(max_n))
    parts = [
        f"lumen_dbg_raw: arch={arch} pos={pos} kind={kind} n={n}",
        *[f"{float(xf[i]):.9g}" for i in range(n)],
    ]
    print(" ".join(parts))


def _hf_rotate_half(x):
    """Same as transformers Qwen2 rotate_half (last-dim halves)."""
    import torch

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _hf_apply_rotary_pos_emb_legacy(q, k, cos, sin, position_ids, unsqueeze_dim: int = 1):
    """Transformers <=4.4x: cos/sin are [seq, head_dim]; index with position_ids then broadcast to [B,H,T,D]."""
    if cos.ndim == 3 and cos.shape[0] == 1:
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_hf_rotate_half(q) * sin)
    k_embed = (k * cos) + (_hf_rotate_half(k) * sin)
    return q_embed, k_embed


def _qwen2_apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Dispatch to installed transformers Qwen2 apply_rotary_pos_emb (API differs by version)."""
    import inspect

    try:
        from transformers.models.qwen2.modeling_qwen2 import (
            apply_rotary_pos_emb as tf_apply_rope,
        )
    except ImportError:
        tf_apply_rope = None
    if tf_apply_rope is not None:
        sig = inspect.signature(tf_apply_rope)
        if "position_ids" in sig.parameters:
            return tf_apply_rope(q, k, cos, sin, position_ids)
        return tf_apply_rope(q, k, cos, sin)
    return _hf_apply_rotary_pos_emb_legacy(q, k, cos, sin, position_ids)


def qwen2_get_rotary_cos_sin(base, h_in_layer0, position_ids):
    """
    Match HF: model-level rotary_emb(hidden, position_ids) when present, else layer0 rotary_emb(value_states, seq_len=...).
    h_in_layer0: [B,T,hidden] input to layer 0 (embedding stream).
    """
    import inspect

    attn0 = base.layers[0].self_attn
    b, t, _ = h_in_layer0.shape
    hd = attn0.head_dim
    nkv = attn0.config.num_key_value_heads
    h_norm = base.layers[0].input_layernorm(h_in_layer0)
    vs = attn0.v_proj(h_norm).view(b, t, nkv, hd).transpose(1, 2)

    model_rope = getattr(base, "rotary_emb", None)
    if model_rope is not None:
        try:
            sig = inspect.signature(model_rope.forward)
            if "position_ids" in sig.parameters:
                out = model_rope(h_in_layer0, position_ids)
                if isinstance(out, (tuple, list)) and len(out) == 2:
                    return out[0], out[1]
        except TypeError:
            pass

    layer_rope = getattr(attn0, "rotary_emb", None)
    if layer_rope is None:
        raise RuntimeError(
            "hf_hidden_ref: Qwen2 has no rotary_emb on model or layer 0 attention"
        )
    sig = inspect.signature(layer_rope.forward)
    if "seq_len" in sig.parameters:
        return layer_rope(vs, seq_len=t)
    if "position_ids" in sig.parameters:
        out = layer_rope(h_in_layer0, position_ids)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            return out[0], out[1]
    out = layer_rope(vs)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        return out[0], out[1]
    raise RuntimeError("hf_hidden_ref: unexpected rotary_emb return type")


def qwen2_qk_flat_after_rope(base, layer_idx, h_in, cos, sin, position_ids, line_pos):
    """Q and K after RoPE at line_pos — same layout as lumen L%d_rope / L%d_krope."""
    attn = base.layers[layer_idx].self_attn
    h_norm = base.layers[layer_idx].input_layernorm(h_in)
    b, t, _ = h_norm.shape
    hd = attn.head_dim
    nh = attn.config.num_attention_heads
    nkv = attn.config.num_key_value_heads
    q = attn.q_proj(h_norm).view(b, t, nh, hd).transpose(1, 2)
    k = attn.k_proj(h_norm).view(b, t, nkv, hd).transpose(1, 2)
    q, k = _qwen2_apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    qf = (
        q[0, :, line_pos, :]
        .detach()
        .contiguous()
        .float()
        .cpu()
        .numpy()
    )
    kf = (
        k[0, :, line_pos, :]
        .detach()
        .contiguous()
        .float()
        .cpu()
        .numpy()
    )
    return qf, kf


def qwen2_l0_qk_kv0_probe(
    base, h_in, cos, sin, position_ids, line_pos: int, arch: str, head_dim: int
) -> None:
    """Q head 0 and K kv head 0 after RoPE at t=0 and t=line_pos (bisect L0_h0 logits)."""
    if line_pos < 1:
        return
    attn = base.layers[0].self_attn
    h_norm = base.layers[0].input_layernorm(h_in)
    b, tlen, _ = h_norm.shape
    hd = attn.head_dim
    nh = attn.config.num_attention_heads
    nkv = attn.config.num_key_value_heads
    q = attn.q_proj(h_norm).view(b, tlen, nh, hd).transpose(1, 2)
    k = attn.k_proj(h_norm).view(b, tlen, nkv, hd).transpose(1, 2)
    q, k = _qwen2_apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    qh = q[0, 0, line_pos].detach().float().cpu().numpy()
    k0 = k[0, 0, 0].detach().float().cpu().numpy()
    kpos = k[0, 0, line_pos].detach().float().cpu().numpy()
    print_dbg_raw(arch, line_pos, "L0_q_h0_tpos", qh, head_dim)
    print_dbg_raw(arch, line_pos, "L0_k_kv0_t0_h", k0, head_dim)
    print_dbg_raw(arch, line_pos, "L0_k_kv0_tpos_h", kpos, head_dim)


def qwen2_l0_head0_logits(
    base, h_in, cos, sin, position_ids, line_pos, sliding_window: int
):
    """Pre-softmax Q·K/sqrt(d) for layer 0 head 0, keys t_start..t_start+natt-1 (matches lumen)."""
    import math

    import torch

    attn = base.layers[0].self_attn
    h_norm = base.layers[0].input_layernorm(h_in)
    b, tlen, _ = h_norm.shape
    hd = attn.head_dim
    nh = attn.config.num_attention_heads
    nkv = attn.config.num_key_value_heads
    inv_scale = 1.0 / math.sqrt(float(hd))
    q = attn.q_proj(h_norm).view(b, tlen, nh, hd).transpose(1, 2)
    k = attn.k_proj(h_norm).view(b, tlen, nkv, hd).transpose(1, 2)
    q, k = _qwen2_apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    t_start = 0
    if sliding_window > 0:
        t_start = line_pos + 1 - sliding_window
        if t_start < 0:
            t_start = 0
    natt = line_pos + 1 - t_start
    out = []
    for ti in range(natt):
        tt = t_start + ti
        s = (q[0, 0, line_pos] * k[0, 0, tt]).sum() * inv_scale
        out.append(s)
    return torch.stack(out).detach().float().cpu().numpy()


def effective_attn_sliding_window(config) -> int:
    """HF Qwen2 applies sliding attention only when use_sliding_window is True."""
    if not bool(getattr(config, "use_sliding_window", False)):
        return 0
    return int(getattr(config, "sliding_window", 0) or 0)


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

    model = load_causal_lm_eager(model_id, dt)
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
    pre_attn_outs: dict[int, object] = {}

    def _attn_hook(layer_idx: int):
        def _hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            attn_outs[layer_idx] = t.detach()

        return _hook

    def _opre_hook(layer_idx: int):
        def _hook(_module, inp, _out):
            pre_attn_outs[layer_idx] = inp[0].detach()

        return _hook

    handles = [
        base.layers[i].self_attn.register_forward_hook(_attn_hook(i))
        for i in range(n_layers)
    ]
    handles_opre = []
    for i in range(n_layers):
        op = getattr(base.layers[i].self_attn, "o_proj", None)
        if op is not None:
            handles_opre.append(op.register_forward_hook(_opre_hook(i)))

    norm = getattr(base, "norm", None)
    if norm is None:
        norm = getattr(base, "final_layernorm", None)
    if norm is None:
        print("no norm on base model", file=sys.stderr)
        sys.exit(1)

    pre_norm_state: dict[str, object] = {}

    def _capture_pre_final_norm(_m, inp):
        pre_norm_state["x"] = inp[0].detach()

    handles_norm = [norm.register_forward_pre_hook(_capture_pre_final_norm)]

    with torch.no_grad():
        embed_out = base.embed_tokens(input_ids)
        out = base(
            input_ids,
            output_hidden_states=True,
            output_attentions=True,
        )

    for h in handles + handles_opre + handles_norm:
        h.remove()

    hs = out.hidden_states
    attns = getattr(out, "attentions", None)
    if hs is None:
        print("model did not return hidden_states", file=sys.stderr)
        sys.exit(1)

    print(
        f"lumen_dbg: HF reference dtype={dt} pos={line_pos} (prompt tok index); "
        f"match lumen -D {line_pos}"
    )

    cos = None
    sin = None
    position_ids = None
    if arch == "qwen2":
        position_ids = torch.arange(
            embed_out.shape[1], device=dev, dtype=torch.long
        ).unsqueeze(0)
        h_in_l0 = hs[0] if len(hs) == n_layers + 1 else embed_out
        try:
            cos, sin = qwen2_get_rotary_cos_sin(base, h_in_l0, position_ids)
        except Exception as e:
            print(f"hf_hidden_ref: Qwen2 rotary failed: {e}", file=sys.stderr)
            sys.exit(1)

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
            if (
                arch == "qwen2"
                and cos is not None
                and sin is not None
                and position_ids is not None
            ):
                qflat, kflat = qwen2_qk_flat_after_rope(
                    base, l, hs[l], cos, sin, position_ids, line_pos
                )
                print(vec_fp_line(arch, line_pos, f"L{l}_rope", qflat))
                print(vec_fp_line(arch, line_pos, f"L{l}_krope", kflat))
                if l == 0:
                    sw = effective_attn_sliding_window(model.config)
                    hd0 = base.layers[0].self_attn.head_dim
                    qwen2_l0_qk_kv0_probe(
                        base, hs[l], cos, sin, position_ids, line_pos, arch, hd0
                    )
                    lg = qwen2_l0_head0_logits(
                        base, hs[l], cos, sin, position_ids, line_pos, sw
                    )
                    print(vec_fp_line(arch, line_pos, "L0_logits_h0", lg))
                    print_dbg_raw(arch, line_pos, "L0_h0_logits", lg)
            if (
                l == 0
                and attns is not None
                and len(attns) > 0
                and attns[0] is not None
            ):
                probs = (
                    attns[0][0, 0, line_pos, : line_pos + 1]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                print(vec_fp_line(arch, line_pos, "L0_probs_h0", probs))
                print_dbg_raw(arch, line_pos, "L0_h0_probs", probs)
            if l in pre_attn_outs:
                pa = (
                    pre_attn_outs[l][0, line_pos]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                print(vec_fp_line(arch, line_pos, f"L{l}_preatn", pa))
                if l == 0:
                    print_dbg_raw(arch, line_pos, "L0_preatn_first16", pa, 16)
            inp_l = hs[l][0, line_pos].detach().float().cpu().numpy()
            ao = attn_outs[l][0, line_pos].detach().float().cpu().numpy()
            print(vec_fp_line(arch, line_pos, f"L{l}_attn", inp_l + ao))
            if l == n_layers - 1:
                vec = (
                    pre_norm_state["x"][0, line_pos]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
            else:
                vec = hs[l + 1][0, line_pos].detach().float().cpu().numpy()
            print(vec_fp_line(arch, line_pos, f"L{l}", vec))
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
            if (
                arch == "qwen2"
                and cos is not None
                and sin is not None
                and position_ids is not None
            ):
                qflat, kflat = qwen2_qk_flat_after_rope(
                    base, l, h_in, cos, sin, position_ids, line_pos
                )
                print(vec_fp_line(arch, line_pos, f"L{l}_rope", qflat))
                print(vec_fp_line(arch, line_pos, f"L{l}_krope", kflat))
                if l == 0:
                    sw = effective_attn_sliding_window(model.config)
                    hd0 = base.layers[0].self_attn.head_dim
                    qwen2_l0_qk_kv0_probe(
                        base, h_in, cos, sin, position_ids, line_pos, arch, hd0
                    )
                    lg = qwen2_l0_head0_logits(
                        base, h_in, cos, sin, position_ids, line_pos, sw
                    )
                    print(vec_fp_line(arch, line_pos, "L0_logits_h0", lg))
                    print_dbg_raw(arch, line_pos, "L0_h0_logits", lg)
            if (
                l == 0
                and attns is not None
                and len(attns) > 0
                and attns[0] is not None
            ):
                probs = (
                    attns[0][0, 0, line_pos, : line_pos + 1]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                print(vec_fp_line(arch, line_pos, "L0_probs_h0", probs))
                print_dbg_raw(arch, line_pos, "L0_h0_probs", probs)
            if l in pre_attn_outs:
                pa = (
                    pre_attn_outs[l][0, line_pos]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                print(vec_fp_line(arch, line_pos, f"L{l}_preatn", pa))
                if l == 0:
                    print_dbg_raw(arch, line_pos, "L0_preatn_first16", pa, 16)
            inp_l = (
                evec
                if l == 0
                else hs[l - 1][0, line_pos].detach().float().cpu().numpy()
            )
            ao = attn_outs[l][0, line_pos].detach().float().cpu().numpy()
            print(vec_fp_line(arch, line_pos, f"L{l}_attn", inp_l + ao))
            vec = hs[l][0, line_pos].detach().float().cpu().numpy()
            print(vec_fp_line(arch, line_pos, f"L{l}", vec))
    else:
        print(
            f"hf_hidden_ref: unexpected hidden_states len={len(hs)} n_layers={n_layers}",
            file=sys.stderr,
        )
        sys.exit(1)

    if "x" not in pre_norm_state:
        print(
            "hf_hidden_ref: final norm forward_pre_hook did not fire",
            file=sys.stderr,
        )
        sys.exit(1)
    # Same tensor as lumen pre_logits: final RMSNorm output, input to lm_head.
    lhs = getattr(out, "last_hidden_state", None)
    if lhs is not None:
        pre = lhs[0, line_pos].detach().float().cpu().numpy()
    else:
        seg = pre_norm_state["x"][:, line_pos : line_pos + 1, :]
        pre = norm(seg)[0, 0].detach().float().cpu().numpy()
    print(vec_fp_line(arch, line_pos, "pre_logits", pre))


if __name__ == "__main__":
    main()
