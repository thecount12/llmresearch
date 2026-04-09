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

## Pre-tokenized prompts (`-P`)

**`-P` overrides `-p`**. You do **not** need Python on Plan 9: build the token file on **any** machine (Linux, macOS, Windows), copy `hello.tok` or `hello.bin` into the Plan 9 file tree, then run `6.out` there.

- **Default (text)**: `file` is **ASCII** — whitespace-separated decimal ids (`#` starts a comment to end of line).
- **Binary**: **`-P file -B`** — `file` is raw **int32 little-endian** (4 bytes per id). No text parser; fine for odd filenames or tools that only emit binary.

Example (host with Python + `transformers`):

```sh
python3 encode_prompt_hf.py -o hello.tok "hello world"
# copy hello.tok to Plan 9, then:
6.out -m SmolLM-135M.Q4_0.gguf -P hello.tok -n 64 -v -a
```

Binary on the host, then Plan 9:

```sh
python3 encode_prompt_hf.py --binary -o hello.bin "hello world"
6.out -m SmolLM-135M.Q4_0.gguf -P hello.bin -B -n 64 -a
```

**Lux** in this repo (e.g. `smollm.lux`) is only for **downloading** GGUFs, not for tokenization. There is no Lux tokenizer for SmolLM here.

## Limitations

- **`-p` string encoding**: without **`-P`**, the runtime feeds the prompt as **raw bytes** (`clamp_token` per character). That matches a **byte-level** toy (`vocab_size ≤ 256`) but is **not** a real BPE/SentencePiece encode for large models. Use **`-P`** with `encode_prompt_hf.py` (or any tool that writes ids) for correct conditioning.
- **Detokenization**: `-a` only fixes a few common display forms; full decoding is not implemented.

## Quick check

```sh
mk
6.out -m model.gguf -v -a -n 32 -p ""
```
