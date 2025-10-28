#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/../.env" || true

ROOT_DIR="${SRC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Run GENIAL dev image interactively, mounting the workspace at /app
docker run -it --rm \
  --name genial_dev_${USER} \
  -v "${ROOT_DIR}:/app" \
  -w /app \
  -e MPLBACKEND=${MPLBACKEND:-Agg} \
  -e MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig} \
  genial:latest \
  bash
