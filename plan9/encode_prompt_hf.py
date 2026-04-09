#!/usr/bin/env python3
"""
Generate a token-id file for lumen -P (ASCII) or -P file -B (binary int32 LE):

  python3 encode_prompt_hf.py -o hello.tok "hello world"
  6.out -m model.gguf -P hello.tok -n 64 -a

  python3 encode_prompt_hf.py --binary -o hello.bin "hello world"
  6.out -m model.gguf -P hello.bin -B -n 64 -a

Requires: pip install transformers
Uses the Hugging Face tokenizer for -m (default: HuggingFaceTB/SmolLM-135M).
"""

import argparse
import struct
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Write token ids for lumen -P")
    ap.add_argument(
        "text",
        nargs="?",
        default="",
        help="prompt text (default: empty)",
    )
    ap.add_argument(
        "-m",
        "--model",
        default="HuggingFaceTB/SmolLM-135M",
        help="HF model id used only to load the tokenizer",
    )
    ap.add_argument(
        "-o",
        "--out",
        default="prompt.tok",
        help="output file",
    )
    ap.add_argument(
        "--binary",
        action="store_true",
        help="write int32 little-endian (use lumen -P path -B); default is ASCII text ids",
    )
    ap.add_argument(
        "--add-special",
        action="store_true",
        help="call encode(..., add_special_tokens=True)",
    )
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("install transformers: pip install transformers", file=sys.stderr)
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok.encode(
        args.text,
        add_special_tokens=bool(args.add_special),
    )
    if args.binary:
        with open(args.out, "wb") as f:
            for tid in ids:
                f.write(struct.pack("<i", int(tid)))
        print(f"wrote {len(ids)} int32 LE ids to {args.out}", file=sys.stderr)
    else:
        with open(args.out, "w", encoding="ascii") as f:
            f.write(" ".join(map(str, ids)))
            f.write("\n")
        print(f"wrote {len(ids)} token ids to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
