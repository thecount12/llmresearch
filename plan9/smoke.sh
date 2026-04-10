#!/bin/sh
# Same as smoke.rc for hosts with sh (optional; Plan 9 uses smoke.rc).
# Optional $1: path to a model (e.g. .gguf); default is the built-in toy model.
set -e
bin=./lumen
[ -x "$bin" ] || bin=./6.out
[ -x "$bin" ] || {
	echo 'smoke.sh: no ./lumen or ./6.out — run mk in this directory first' >&2
	exit 1
}
if [ "$#" -ge 1 ] && [ -n "$1" ]; then
	exec "$bin" -m "$1" -n 1 -v
fi
exec "$bin" -n 1 -v
