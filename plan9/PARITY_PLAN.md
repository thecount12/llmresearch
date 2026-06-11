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

### Phase 4B — Other GGUF quants (in progress)

- Re-run Phases 1–3 with a **Q8_0** or **Q4_0** GGUF (see `gguf-notes.md`: **`loader-gguf.c`** dequantizes **F32, F16, Q8_0, Q4_0** only; **Q4_K** and other block types still need loader work).
- Same **`parity.rc`** / **`host_parity.sh`** ladder; expect **more drift** vs HF F16; treat **greedy_id / gen[]** as the primary pass criterion unless you add an HF quant reference.

**Checkpoint (Qwen2.5-0.5B F16 → `llama-quantize` → Q8_0):** versus **HF F16** (`host_parity.sh`): **`hello2.tok`** — **full 8-step greedy chain matches**. **`fixtures/kv64.tok` pos=63** — **`gen[0]`** matches (**19482**); **greedy chain diverges after step 1** (quant + long-context logits reorder near-ties). For **quant-only** sanity on **`kv64`**, compare **`lumen`** to **llama.cpp** decoding with the **same** Q8_0 GGUF.

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

**No HF folder needed:** from **llama.cpp**, **`llama-quantize MyModel-f16.gguf MyModel-q8_0.gguf Q8_0`** (then copy the output GGUF to Plan 9).

### Phase 4C — Other checkpoints (next)

Repeat **Phase 0–3** for a **new HF id + matching GGUF**. Same **`parity.rc`** / **`host_parity.sh`**; override **`gguf`**, **`hf`**, **`tok`**, **`row`**, **`pos`** per model.

**Recommended first target:** **`Qwen/Qwen2.5-1.5B-Instruct`** — same **Qwen2** graph as 0.5B (no new arch in lumen); larger weights / KV; use **`-c 2048`** or default cap if Plan 9 RAM is tight.

| Step | Action |
|------|--------|
| 0 | F16 GGUF: **`llama-quantize`** from HF-export F16, or **`convert_hf_to_gguf.py`** + optional quant (see 4B). Copy to Plan 9. |
| 0 | **`lumen -W -m $gguf -v`** — vocab, dim, layers, heads, rope, **`embed_layout`**, seq_len vs HF **`config.json`**. |
| 0 | Prompt ids: **`encode_prompt_hf.py -m Qwen/Qwen2.5-1.5B-Instruct -o fixtures/hello2_1.5b.tok "…"`** (or **`--chat`** for instruct). **Do not assume** 0.5B **`hello2.tok`** ids unless you verified the tokenizer is identical. |
| 1–3 | **`gguf=… hf='Qwen/Qwen2.5-1.5B-Instruct' tok=… pos=… row=… run=1 rc parity.rc`** |
| host | **`./scripts/host_parity.sh Qwen/Qwen2.5-1.5B-Instruct $tok $pos $row 8`** |
| archive | **`baselines/PARITY_Qwen2.5-1.5B_*.txt`**, optional **`PARITY_SAVE=…`** |

**Pass criteria:** same as Phases 1–3 — embed row, **`greedy_id`**, **`gen[]`** vs HF F16; layer cksums approximate.

**Later targets:** **`Qwen2.5-3B-Instruct`**, **`Qwen2.5-7B-Instruct`** (RAM / **`-c`**), or **Qwen2** (non-2.5) if GGUF arch string is still **`qwen2`**. **Qwen3 / MoE / SWA hybrids** need arch loader work first — not 4C “free” repeats.

See **`baselines/PARITY_PHASE4C_1.5B.txt`** for a copy-paste checklist.

### Phase 4C — 1.5B on memory-limited Plan 9 / QEMU

**Problem:** Today **`lumen`** expands every GGUF weight to **F32 in RAM** at load (`loader-gguf.c` **`loadf16tensor`** → **`f16tof32`**). For **1.5B** that is ~**6 GB** weights plus a duplicate **~0.9 GB** **`wcls`** copy when embeddings are tied (~**7–8 GB** peak). Observed failures:

| Environment | Symptom |
|-------------|---------|
| arm64 QEMU 4 GB | `Killed: Insufficient physical memory` |
| arm64 QEMU 8–16 GB | `panic: kmapaddr: pa=0x140000000` (kernel mapping ~5 GB physical) |
| amd64 Plan 9 `cpu` | `mallocz failed` |

**Not fixes:** More QEMU **`-m`** alone (16 GB still panicked); **emulated amd64** QEMU on Apple Silicon (TCG — slow); Q8_0 on disk without keeping quant in RAM.

**HF reference (done on Mac):** **`hello2.tok` pos=1** — `greedy_id=271`, chain **`271, 40, 2776, 4460, 311, 1855, 264, 729`**. Archive with **`PARITY_SAVE=baselines/HF_HELLO2_POS1_1.5B_F16.txt`**.

**Goal:** Load **1.5B F16 GGUF** on **arm64 QEMU (HVF)** with **≤ ~4 GB** steady-state weight RAM, then run the usual **`parity.rc`** ladder.

#### Step M1 — Tied-embed alias (small, do first)

| | |
|--|--|
| **Change** | When GGUF has no separate **`output.weight`**, set **`wcls = token_embedding_table`**; skip **`wcls`** alloc and **`memmove`** in **`loader-gguf.c`**. **`free_model`**: do not double-free. |
| **Files** | **`model.c`**, **`loader-gguf.c`**, maybe **`model.h`** (`wcls_tied` flag) |
| **Saves** | ~**0.9 GB** (1.5B) |
| **Pass** | **0.5B** Phases **1–3** unchanged vs existing baselines; **1.5B** smoke may still fail (expected until M2). |

```text
# After mk install — regression on 0.5B
gguf=Qwen2.5-0.5B-Instruct-f16.gguf hf='Qwen/Qwen2.5-0.5B-Instruct' tok=hello2.tok pos=1 row=19482 gsteps=8 run=1 rc parity.rc
```

#### Step M2 — F16 weights in RAM (main win)

| | |
|--|--|
| **Change** | Store layer + embed tensors as **F16**; **`matvec` / embed / logits** accumulate in **F32** activations (HF F16 inference story). Loader: **`loadf16tensor`** writes **`ushort`**, no expand-at-load. |
| **Files** | **`model.h`** (weight pointers or dtype), **`loader-gguf.c`**, **`tensor.c`**, **`transformer.c`**, **`main.c`** (`-Z` embed dump) |
| **Saves** | ~**50%** on weights → ~**3 GB** for 1.5B (+ M1) |
| **Pass** | **0.5B** greedy **`gen[]`** still matches HF F16; then **1.5B** smoke below. |

**Parity note:** Layer cksums may move closer to HF **`--dtype float16`**; **`greedy_id` / `gen[]`** remain primary.

#### Step M3 — 1.5B parity on Plan 9 (after M1+M2)

```text
# Smoke (expect dim=1536 layers=28, lumen_dump line)
lumen -v -m Qwen2.5-1.5B-Instruct-f16.gguf -Z 19482 -n 0

# Ladder
gguf=Qwen2.5-1.5B-Instruct-f16.gguf hf='Qwen/Qwen2.5-1.5B-Instruct' tok=hello2.tok pos=1 row=19482 gsteps=8 run=1 rc parity.rc
```

| Check | 1.5B target (HF F16, Mac) |
|-------|---------------------------|
| **`lumen_dump`** first_16 | see **`host_parity.sh`** embed row for **19482** |
| **`greedy_id`** | **271** |
| **`gen[0..7]`** | **271, 40, 2776, 4460, 311, 1855, 264, 729** |

**QEMU:** **8192 MB** VM should suffice after M2; retest on **gabriel** (arm64 HVF). Optional **`ctx=128`** for later **kv64** on 1.5B.

#### Deferred (not in this mini-plan)

- Q8_0 / Q4_0 **in RAM** (needs quant matmul path; disk quant alone does not cut RAM today).
- **3B+** checkpoints (scale M2; still need headroom).
- Plan 9 **kernel** highmem / **`kmapaddr`** fix upstream (orthogonal to M1/M2).

## Current status (checkpoint)

**Qwen2.5-0.5B-Instruct, F16 GGUF, `hello2.tok` pos=1:** Phases **0–3** done — ladder passes for **embed**, **greedy next token**, and **8-step greedy chain** vs HF; layer checksums vs HF float16 are approximate.

**Phase 4A:** Done for **Qwen2.5-0.5B F16 + `kv64`**: long-prefix parity (**`tok=fixtures/kv64.tok pos=63`**, greedy chain vs **`hf_logits_ref --greedy-steps`**), **`ctx=128`** cap smoke, and **`ctx=60`** negative test (**too many prompt ids** / cap respected). **`parity.rc`** prints underlying lumen errors when grep has no matches. Archive under **`baselines/`** if you want a frozen HF log (**`PARITY_SAVE`** + **`scripts/host_parity.sh`**).

**Phase 4B (Q8_0):** **`hello2`** — full greedy parity vs HF F16 ✓. **`kv64`** — **`gen[0]`** vs HF F16 ✓; **multi-step greedy vs HF F16** ✗ (documented above). Optional: **`Q4_0`**, **llama.cpp vs lumen** on **`kv64`**, **`PARITY_SAVE`** baselines.

**Phase 4C (1.5B):** HF ladder on **Mac** ✓ (`greedy_id=271`, chain **`271,40,2776,4460,311,1855,264,729`**). **Plan 9 lumen** blocked on RAM — see **Phase 4C — 1.5B on memory-limited Plan 9 / QEMU** (steps **M1 → M2 → M3**). **0.5B** remains the live Plan 9 parity target until M2 lands.

**Next:** Implement **M1** (tied embed), then **M2** (F16 in RAM), rerun **M3** on gabriel. Phase 4B optional: **`Q4_0`**, llama.cpp vs lumen on **`kv64` Q8_0**.

### `parity.rc` / rc gotchas

- **`lm=-W -m $gguf`** assigns only **`-W`** to `lm`; **`-m`** is then run as a command (errors like **`./-m`**).
- **`lm=(-W -m $gguf)`** is not reliable on all **`rc`** builds.
- **`parity.rc`** uses **`usectx`** and **`>$tmp >[2=1]`** (rc redirect order matters; see **`l0parity.rc`**). **`lumen -W`** still works by hand but **`run=1`** path prefers rc merge.
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
