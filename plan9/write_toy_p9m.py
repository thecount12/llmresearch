#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path


def fill_identity(n):
    out = [0.0] * (n * n)
    for i in range(n):
        out[i * n + i] = 1.0
    return out


def fill_rect_identity(rows, cols):
    out = [0.0] * (rows * cols)
    for i in range(min(rows, cols)):
        out[i * cols + i] = 1.0
    return out


def write_f32s(f, values):
    f.write(struct.pack(f"<{len(values)}f", *values))


def build_toy_model():
    cfg = {
        "version": 1,
        "vocab_size": 256,
        "dim": 32,
        "hidden_dim": 64,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 4,
        "seq_len": 64,
        "rms_eps": 1e-5,
    }
    head_dim = cfg["dim"] // cfg["n_heads"]
    kv_dim = head_dim * cfg["n_kv_heads"]

    token_embedding_table = []
    for i in range(cfg["vocab_size"]):
        for j in range(cfg["dim"]):
            token_embedding_table.append(((i + j) % 17) / 17.0)

    layers = []
    for _ in range(cfg["n_layers"]):
        layer = {
            "rms_att_weight": [1.0] * cfg["dim"],
            "wq": fill_identity(cfg["dim"]),
            "wk": fill_rect_identity(kv_dim, cfg["dim"]),
            "wv": fill_rect_identity(kv_dim, cfg["dim"]),
            "wo": fill_identity(cfg["dim"]),
            "rms_ffn_weight": [1.0] * cfg["dim"],
            "w1": fill_rect_identity(cfg["hidden_dim"], cfg["dim"]),
            "w2": fill_rect_identity(cfg["dim"], cfg["hidden_dim"]),
            "w3": fill_rect_identity(cfg["hidden_dim"], cfg["dim"]),
        }
        layers.append(layer)

    rms_final_weight = [1.0] * cfg["dim"]
    wcls = fill_rect_identity(cfg["vocab_size"], cfg["dim"])

    return cfg, token_embedding_table, layers, rms_final_weight, wcls


def write_model(path: Path):
    cfg, token_embedding_table, layers, rms_final_weight, wcls = build_toy_model()

    with path.open("wb") as f:
        f.write(struct.pack(
            "<4s8If",
            b"P9DM",
            cfg["version"],
            cfg["vocab_size"],
            cfg["dim"],
            cfg["hidden_dim"],
            cfg["n_layers"],
            cfg["n_heads"],
            cfg["n_kv_heads"],
            cfg["seq_len"],
            cfg["rms_eps"],
        ))

        write_f32s(f, token_embedding_table)
        for layer in layers:
            write_f32s(f, layer["rms_att_weight"])
            write_f32s(f, layer["wq"])
            write_f32s(f, layer["wk"])
            write_f32s(f, layer["wv"])
            write_f32s(f, layer["wo"])
            write_f32s(f, layer["rms_ffn_weight"])
            write_f32s(f, layer["w1"])
            write_f32s(f, layer["w2"])
            write_f32s(f, layer["w3"])
        write_f32s(f, rms_final_weight)
        write_f32s(f, wcls)


def main():
    parser = argparse.ArgumentParser(description="Write a toy P9DM checkpoint.")
    parser.add_argument("output", nargs="?", default="toy.p9m")
    args = parser.parse_args()

    out = Path(args.output)
    write_model(out)
    print(out)


if __name__ == "__main__":
    main()
