# Simple Model Format

The educational loader in `plan9/loader-simple.c` reads a fixed binary layout.

Header:

- 4 bytes magic: `P9DM`
- `unsigned version`
- `unsigned vocab_size`
- `unsigned dim`
- `unsigned hidden_dim`
- `unsigned n_layers`
- `unsigned n_heads`
- `unsigned n_kv_heads`
- `unsigned seq_len`
- `float rms_eps`

Tensor order after the header:

1. token embedding table: `vocab_size * dim`
2. for each layer:
   - attention RMS weight: `dim`
   - `wq`: `dim * dim`
   - `wk`: `kv_dim * dim`
   - `wv`: `kv_dim * dim`
   - `wo`: `dim * dim`
   - feed-forward RMS weight: `dim`
   - `w1`: `hidden_dim * dim`
   - `w2`: `dim * hidden_dim`
   - `w3`: `hidden_dim * dim`
3. final RMS weight: `dim`
4. classifier / lm head: `vocab_size * dim`

Where:

- `head_dim = dim / n_heads`
- `kv_dim = head_dim * n_kv_heads`

Notes:

- All tensor data is stored as contiguous `float32`.
- Matrices are row-major and used by `matvec(out, w, x, rows, cols)`.
- This format is only for educational checkpoints and debugging the runtime before adding a real loader such as GGUF.
