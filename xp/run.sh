#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -x xp/.venv/bin/python ]; then
    echo 'Run ./xp/setup.sh first' >&2
    exit 1
fi
exec xp/.venv/bin/python -u xp/runner.py "$@"
