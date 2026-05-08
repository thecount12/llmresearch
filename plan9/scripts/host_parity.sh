#!/bin/sh
# Run HF parity steps 1–4 from plan9/ (requires torch, transformers).
# Usage: ./scripts/host_parity.sh [HF_MODEL] [TOKFILE] [POS] [EMBED_ROW] [GREEDY_STEPS]
# Optional: PARITY_SAVE=file  →  append same stdout you see on screen (for baselines/).
#
# Progress lines go to stderr.  Filtered parity lines go to stdout; if stdout is not a
# tty (e.g. script > out.txt), the same lines are also copied to stderr so you still see
# them in the terminal.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

HF=${1:-Qwen/Qwen2.5-0.5B-Instruct}
TOK=${2:-hello2.tok}
POS=${3:-1}
ROW=${4:-19482}
GSTEPS=${5:-8}

warn() {
	echo "$*" >&2
}

# Emit each line to stdout; duplicate to stderr when stdout is not a terminal.
emit_lines()
{
	if [ -t 1 ]; then
		cat "$1"
	else
		while IFS= read -r _line || [ -n "$_line" ]; do
			printf '%s\n' "$_line"
			printf '%s\n' "$_line" >&2
		done <"$1"
	fi
}

# Run python, capture stdout+stderr, filter with grep -E; never use exit inside here.
run_py_grep()
{
	_label=$1
	_pat=$2
	shift 2
	_tmp=$(mktemp /tmp/host_parity.XXXXXX) || exit 1
	_filt="${_tmp}.f"
	warn "[host_parity] $_label: running python3 ($(command -v python3)) ..."
	# With set -e, a failing python3 would exit this shell before we read $? — never use
	# bare failing commands here; capture status inside set +e.
	set +e
	python3 -u "$@" >"$_tmp" 2>&1
	_py=$?
	_bytes=$(wc -c <"$_tmp" | awk '{print $NF}')
	set -e
	[ -n "$_bytes" ] || _bytes=0
	warn "[host_parity] $_label: python exit=$_py captured_${_bytes}_bytes"

	if [ "$_py" -ne 0 ]; then
		warn "[host_parity] $_label: full output (failure):"
		cat "$_tmp" >&2
		rm -f "$_tmp" "$_filt" || true
		return "$_py"
	fi

	# Avoid if grep; then ... — BSD grep exits 1 when no match (fine), but keep logic explicit.
	grep -E "$_pat" "$_tmp" >"$_filt" || :
	if ! test -s "$_filt"; then
		warn "[host_parity] $_label: grep matched no lines; full python output:"
		cat "$_tmp" >&2
		rm -f "$_tmp" "$_filt" || true
		return 0
	fi

	emit_lines "$_filt"
	rm -f "$_tmp" "$_filt" || true
	return 0
}

run_all()
{
	echo "=== hf_hidden_ref --embed-row $ROW ==="
	run_py_grep embed 'lumen_dbg:|hf_dump:|hf_hidden_ref:' hf_hidden_ref.py -m "$HF" --embed-row "$ROW" --dtype float16

	echo
	echo "=== hf_hidden_ref -p $TOK --pos $POS (pre_logits + L0_) ==="
	run_py_grep hidden 'lumen_dbg:.*(pre_logits|L0_)' hf_hidden_ref.py -m "$HF" -p "$TOK" --pos "$POS" --dtype float16

	echo
	echo "=== hf_logits_ref -p $TOK (next-token fingerprint) ==="
	run_py_grep logits 'lumen_hf:|hf_logits_ref:' hf_logits_ref.py -m "$HF" -p "$TOK" --dtype float16

	echo
	echo "=== hf_logits_ref --greedy-steps $GSTEPS ==="
	run_py_grep greedy 'lumen_hf:|hf_logits_ref:' hf_logits_ref.py -m "$HF" -p "$TOK" --dtype float16 --greedy-steps "$GSTEPS"
}

if [ -n "$PARITY_SAVE" ]; then
	_log=$(mktemp /tmp/host_parity_all.XXXXXX) || exit 1
	set +e
	run_all >"$_log"
	_rc=$?
	set -e
	cat "$_log"
	cat "$_log" >>"$PARITY_SAVE"
	rm -f "$_log" || true
	[ "$_rc" -eq 0 ] || exit "$_rc"
else
	run_all
fi
