#!/usr/bin/env bash
# Thin wrapper around install.py - see `python3 install.py --help` for options.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" > /dev/null; then
    echo "python3 was not found on PATH. Install Python 3.11 or newer first." >&2
    exit 1
fi

exec "$PYTHON" install.py "$@"
