# GGUF loader (`loader-gguf.c`)

The loader reads a **GGUF** file end-to-end: metadata, tensor names/types/offsets, **dequantizes** supported GGML types into `float`, maps tensors into `Model`, and loads **`tokenizer.ggml.tokens`** into `Model.token_str` for display.

## Supported

- **Metadata**: `general.architecture`, `general.alignment`, dimensions, vocab, context, etc.
- **GGML types**: F32, F16, Q4_0, Q8_0 (others fail with a clear error).
- **Tokenizer**: `tokenizer.ggml.tokens` as an array of `GGUFString` (per-token UTF-8 strings).
- **Tied embeddings**: if there is no separate output weight, copies `token_embd` into `wcls`.

## CLI (`main.c` / `lumen`)

- **`-m path`**: model file (`.gguf` or `.p9m` / `.bin` simple format).
- **`-v`**: stderr summary (loader kind, dims, vocab string count).
- **`-g`**: stderr top-8 logits each generation step (before sampling).
- **`-a`**: “pretty” token display: scans **each piece** and replaces **every** known HF/SentencePiece UTF-8 sequence with **ASCII** (`C4 A0` Ġ, `E2 96 81` ▁, `C2 A0` NBSP → space; `C4 8A` Ċ → newline). Other UTF-8 in the piece is passed through unchanged (may still look wrong on non-UTF-8 terminals).

## Limitations

- **Prompt encoding**: the runtime feeds the prompt as **raw bytes** (`clamp_token` per character). That matches a **byte-level** toy (`vocab_size ≤ 256`) but is **not** a real BPE/SentencePiece encode for large models. For SmolLM-scale checkpoints, use **`-p ""`** to see unconditional generation, or treat prompt conditioning as approximate until a tokenizer is integrated.
- **Detokenization**: `-a` only fixes a few common display forms; full decoding is not implemented.

## Quick check

```sh
mk
6.out -m model.gguf -v -a -n 32 -p ""
```
