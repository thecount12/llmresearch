#!/bin/sh
# Run HF parity steps 1–4 from plan9/ (requires torch, transformers).
# Usage: ./scripts/host_parity.sh [HF_MODEL] [TOKFILE] [POS] [EMBED_ROW] [GREEDY_STEPS]
# Optional: PARITY_SAVE=file  →  tee full combined log to file (for baselines/).
set -e
set -o pipefail
cd "$(dirname "$0")/.."
HF=${1:-Qwen/Qwen2.5-0.5B-Instruct}
TOK=${2:-hello2.tok}
POS=${3:-1}
ROW=${4:-19482}
GSTEPS=${5:-8}

run()
{
	if [ -n "$PARITY_SAVE" ]; then
		tee -a "$PARITY_SAVE"
	else
		cat
	fi
}

{
echo "=== hf_hidden_ref --embed-row $ROW ==="
python3 hf_hidden_ref.py -m "$HF" --embed-row "$ROW" --dtype float16 2>&1 | grep -E 'lumen_dbg:|hf_dump:|hf_hidden_ref:'

echo
echo "=== hf_hidden_ref -p $TOK --pos $POS (pre_logits + L0_) ==="
python3 hf_hidden_ref.py -m "$HF" -p "$TOK" --pos "$POS" --dtype float16 2>&1 | grep -E 'lumen_dbg:.*(pre_logits|L0_)'

echo
echo "=== hf_logits_ref -p $TOK (next-token fingerprint) ==="
python3 hf_logits_ref.py -m "$HF" -p "$TOK" --dtype float16 2>&1 | grep -E 'lumen_hf:|hf_logits_ref:'

echo
echo "=== hf_logits_ref --greedy-steps $GSTEPS ==="
python3 hf_logits_ref.py -m "$HF" -p "$TOK" --dtype float16 --greedy-steps "$GSTEPS" 2>&1 | grep -E 'lumen_hf:|hf_logits_ref:'
} | run
