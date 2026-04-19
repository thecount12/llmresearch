#!/bin/sh
# Run HF parity steps 1–3 from plan9/ (requires torch, transformers).
# Usage: ./scripts/host_parity.sh [HF_MODEL] [TOKFILE] [POS] [EMBED_ROW]
set -e
cd "$(dirname "$0")/.."
HF=${1:-Qwen/Qwen2.5-0.5B-Instruct}
TOK=${2:-hello2.tok}
POS=${3:-1}
ROW=${4:-19482}

echo "=== hf_hidden_ref --embed-row $ROW ==="
python3 hf_hidden_ref.py -m "$HF" --embed-row "$ROW" --dtype float16 2>&1 | grep -E 'lumen_dbg:|hf_dump:|hf_hidden_ref:'

echo
echo "=== hf_hidden_ref -p $TOK --pos $POS (grep pre_logits L0_) ==="
python3 hf_hidden_ref.py -m "$HF" -p "$TOK" --pos "$POS" --dtype float16 2>&1 | grep -E 'lumen_dbg:.*(pre_logits|L0_)'

echo
echo "=== hf_logits_ref -p $TOK ==="
python3 hf_logits_ref.py -m "$HF" -p "$TOK" --dtype float16 2>&1 | grep -E 'lumen_hf:|hf_logits_ref:'
