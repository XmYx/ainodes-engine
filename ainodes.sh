#!/usr/bin/env bash
# Launch aiNodes Engine from its virtual environment.
#
#   ./ainodes.sh                       normal start
#   ./ainodes.sh --theme light         light theme
#   ./ainodes.sh --headless --run g.json
#
# Run ./install.sh first if nodes_env/ does not exist yet.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

VENV="${AINODES_VENV:-nodes_env}"
PYTHON="$VENV/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "No environment at $VENV - run ./install.sh first." >&2
    exit 1
fi

exec "$PYTHON" main.py "$@"
