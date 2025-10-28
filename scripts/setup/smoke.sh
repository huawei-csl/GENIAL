#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/../.venv/bin/activate"
python - <<'PY'
import flowy, genial
PY
python -m flowy --help >/dev/null
python -m genial --help >/dev/null
echo "smoke test passed"
