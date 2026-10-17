#!/usr/bin/env bash
# Install this fork of bagpipes, pulling model grids from the upstream pip wheel.
#
# Run this from the repo root, inside the Python environment you want to install
# into. If you already have bagpipes installed in that environment, the editable
# install at the end of this script will replace it — see the README for how to
# preserve an existing install by using a dedicated environment.

set -euo pipefail

if [[ ! -f setup.py ]] || [[ ! -d bagpipes/models ]]; then
    echo "Error: run this script from the bagpipes fork repo root." >&2
    exit 1
fi

if ! command -v pip >/dev/null 2>&1; then
    echo "Error: pip not found. Activate your Python environment first." >&2
    exit 1
fi

TMPDIR=$(mktemp -d -t bagpipes-grids-XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

echo "==> Fetching upstream bagpipes wheel for model grids..."
pip install bagpipes --no-deps --target "$TMPDIR" --quiet --upgrade

SRC="$TMPDIR/bagpipes/models/grids"
DST="bagpipes/models/grids"

if [[ ! -d "$SRC" ]]; then
    echo "Error: no grids directory found in the upstream wheel at $SRC." >&2
    exit 1
fi

echo "==> Copying model grids into $DST/ ..."
mkdir -p "$DST"
cp -rn "$SRC"/* "$DST"/

echo "==> Installing the fork in editable mode..."
pip install -e .

echo
echo "Done."
echo "Verify with:  python -c 'import bagpipes; print(bagpipes.__file__)'"

# The upstream pip wheel ships only BC03 (+ DL07) grids. This fork defaults to
# BPASS — if you need BPASS fits, obtain those grid files separately and drop
# them into bagpipes/models/grids/.
if ! ls "$DST"/bpass*.fits >/dev/null 2>&1; then
    echo
    echo "Note: no BPASS grids found in $DST/."
    echo "      This fork defaults to BPASS stellar models; obtain the BPASS"
    echo "      grid files separately and place them in $DST/ if you need them."
fi
