# lumen ↔ Hugging Face parity: step-by-step plan

This is the phased checklist we used for **Qwen2.5-0.5B-Instruct** (F16 GGUF + HF). The runnable ladder is **`parity.rc`**; frozen command snippets live in **`baselines/PARITY_HELLO2_POS1.txt`**.

## Locked references (do this first)

1. **GGUF:** e.g. `Qwen2.5-0.5B-Instruct-f16.gguf` (same tokenizer story as HF).
2. **HF model id:** `Qwen/Qwen2.5-0.5B-Instruct`.
3. **Prompt file:** `hello2.tok` (two token ids; parity uses **`pos=1`** = last prompt token).
4. **Embed row spot-check:** token id **19482** (second line of `hello2.tok`).
5. **`lumen -v`** (optional): config vs `config.json` (arch, dim, layers, heads, rope, vocab).

## Phase 0 — Environment invariants

- Same GGUF + same HF repo + ids from **`encode_prompt_hf.py`** (or equivalent) for that model.
- Document where **`lumen`** is built/installed (`mk`, `mk install`, or `lumenbin=...`).
- Know that **lumen is float32**; HF **`--dtype float16`** matches weight story but not bit-identical layer checksums (see comments in `parity.rc`, `hf_hidden_ref.py`, `hf_logits_ref.py`).

## Phase 1 — Ladder smoke (steps 1–3)

Run **`run=1 rc parity.rc`** (or echo-only `rc parity.rc` and run commands by hand).

| Step | lumen | HF (host) | Pass criterion |
|------|--------|-----------|----------------|
| 1 | `-Z <row> -n 0` → `lumen_dump` | `hf_hidden_ref.py --embed-row` | First 16 floats match (rounding) |
| 2 | `-P tok -n 0 -D <pos>` → `lumen_dbg` | `hf_hidden_ref.py -p tok --pos` | **pre_logits** sumsq/cksum close enough to bisect; F16 HF may drift from F32 lumen |
| 3 | `-P tok -n 1 -F` → `lumen_hf` | `hf_logits_ref.py -p tok` | Same **greedy_id**; top-5 order; logit value may differ slightly |

Archive outputs under **`baselines/`** when a milestone passes.

## Phase 2 — Bisect forward (when step 2 diverges)

1. Find the **first** `lumen_dbg` line where sumsq/cksum (or raw L0 block) diverges from HF in a meaningful way.
2. Fix **loader / tensor / transformer** for that subsystem (embed, norm, RoPE, KV, MHA, FFN, etc.).
3. Repeat until **`pre_logits`** (last `lumen_dbg` before logits head) matches within tolerance.

## Phase 3 — Logits and generation

1. Match **`-F`** fingerprint to **`hf_logits_ref.py`** (pre_mask greedy id and topology of logits).
2. Validate **greedy multi-step:** **`lumen -n K -e`** vs **`hf_logits_ref.py --greedy-steps K`** (`gen[i] id=` lines).
3. Optionally: sampling temperature, EOS masking behavior (separate from greedy parity).

## Phase 4–5 — Hardening (ongoing)

### Phase 4A — Context / KV (started in repo)

1. Optional **`ctx=N`** in **`parity.rc`** → passes **`lumen -c N`** on every step (`N=0` = model max per `main.c`).
2. **Failure injection:** set **`ctx`** smaller than `prompt_len + gsteps` and confirm **`lumen`** errors cleanly (prompt + steps vs `seq_len`).
3. **Long prefix fixture:** **`fixtures/kv64.tok`** (64× token `14990`). From **`plan9/`** run e.g.  
   `ctx=128 tok=fixtures/kv64.tok pos=63 run=1 rc parity.rc`  
   and on the host **`hf_hidden_ref.py -p fixtures/kv64.tok --pos 63`**, **`hf_logits_ref.py -p fixtures/kv64.tok`**, etc. (same HF model).
4. **`scripts/host_parity.sh`** now runs steps **1–4** (includes **`--greedy-steps`**). Optional **`PARITY_SAVE=file`** appends the combined grep’d log for baselines.

### Phase 4B — Other GGUF quants (not started)

- Re-run Phases 1–3 with a **Q8_0** or **Q4_0** GGUF (see `gguf-notes.md`: **`loader-gguf.c`** dequantizes **F32, F16, Q8_0, Q4_0** only; **Q4_K** and other block types still need loader work).
- Same **`parity.rc`** / **`host_parity.sh`** ladder; expect **more drift** vs HF F16; treat **greedy_id / gen[]** as the primary pass criterion unless you add an HF quant reference.

**Produce a lumen-compatible GGUF on a host:** `convert_hf_to_gguf.py` lives in the **[llama.cpp](https://github.com/ggerganov/llama.cpp)** repo (repo root), **not** in `llmresearch/plan9/`. Clone or update llama.cpp, `cd` into that tree, install its Python deps if prompted, then run the script **by path** (first arg = local HF model dir with `config.json` + weights, e.g. from `huggingface-cli download`):

```text
cd /path/to/llama.cpp
python3 convert_hf_to_gguf.py /path/to/local/Qwen2.5-0.5B-Instruct --outfile Qwen2.5-0.5B-Instruct-q8_0.gguf --outtype q8_0
# or:  --outtype q4_0
```

If you run `python3 convert_hf_to_gguf.py` from **`plan9/`**, Python looks for **`plan9/convert_hf_to_gguf.py`** and fails — always **`cd` to llama.cpp** or pass the **full path** to the script under llama.cpp.

Copy the `.gguf` to Plan 9, then from **`plan9/`**:

```text
gguf=Qwen2.5-0.5B-Instruct-q8_0.gguf
tok=hello2.tok pos=1 row=19482 gsteps=8 run=1 rc parity.rc
```

If **`lumen`** fails at load with an **unsupported GGML type**, the file likely uses **Q4_K / Q5 / IQ** — re-export with **`q8_0`** or **`q4_0`** only, or extend **`loader-gguf.c`**.

### Phase 4C — Other checkpoints (not started)

- New size or arch: repeat Phase 0–3; add **`baselines/…`** snippets.

## Current status (checkpoint)

**Qwen2.5-0.5B-Instruct, F16 GGUF, `hello2.tok` pos=1:** Phases **0–3** done — ladder passes for **embed**, **greedy next token**, and **8-step greedy chain** vs HF; layer checksums vs HF float16 are approximate.

**Phase 4A:** Done for **Qwen2.5-0.5B F16 + `kv64`**: long-prefix parity (**`tok=fixtures/kv64.tok pos=63`**, greedy chain vs **`hf_logits_ref --greedy-steps`**), **`ctx=128`** cap smoke, and **`ctx=60`** negative test (**too many prompt ids** / cap respected). **`parity.rc`** prints underlying lumen errors when grep has no matches. Archive under **`baselines/`** if you want a frozen HF log (**`PARITY_SAVE`** + **`scripts/host_parity.sh`**).

**Next:** **Phase 4B** — export **`q8_0`** or **`q4_0`** GGUF (see Phase 4B block above), copy to Plan 9, rerun **`parity.rc`** with **`gguf=`** set; keep **`host_parity.sh`** on the same HF id for greedy-id comparison.

### `parity.rc` / rc gotchas

- **`lm=-W -m $gguf`** assigns only **`-W`** to `lm`; **`-m`** is then run as a command (errors like **`./-m`**).
- **`lm=(-W -m $gguf)`** is not reliable on all **`rc`** builds.
- **`parity.rc`** uses **`usectx`** and two explicit lines: **`$bin -W -m $gguf -c $ctx …`** vs **`$bin -W -m $gguf …`**.
- If **`lumen`** shows **vocab 256** and **`hello2.tok`** ids are “out of range”, **`-m $gguf` never reached** the binary (wrong rc words) or **`gguf`** is unreadable / missing from cwd.

## Quick commands

```text
# Plan 9 (stderr merged via -W; see parity.rc)
lumenbin=/path/to/lumen run=1 rc parity.rc

# Host (after copying echoed lines from parity.rc)
python3 hf_hidden_ref.py -m Qwen/Qwen2.5-0.5B-Instruct --embed-row 19482 --dtype float16
python3 hf_hidden_ref.py -m Qwen/Qwen2.5-0.5B-Instruct -p hello2.tok --pos 1 --dtype float16
python3 hf_logits_ref.py -m Qwen/Qwen2.5-0.5B-Instruct -p hello2.tok --dtype float16
python3 hf_logits_ref.py -m Qwen/Qwen2.5-0.5B-Instruct -p hello2.tok --dtype float16 --greedy-steps 8
```

Optional: repeat **`hf_*`** with **`--dtype float32`** for closer layer fingerprints to float32 lumen (still not guaranteed identical).
