#!/bin/bash
set -euo pipefail

#############################################
# Args
#############################################
BUILD_BASE=false          # hidden feature: only effective with --no-download
INSPECT_CONTEXT=false
DOWNLOAD=true

for arg in "$@"; do
  case "$arg" in
    --view-context|--view_context)
      INSPECT_CONTEXT=true ;;
    --no-inspect-context)
      INSPECT_CONTEXT=false ;;
    --download)
      DOWNLOAD=true ;;
    --no-download)
      DOWNLOAD=false ;;
    # hidden/undocumented switch:
    -B|--build-base)
      BUILD_BASE=true ;;
    -h|--help)
      echo "Usage: $0 [--view-context] [--no-inspect-context] [--download] [--no-download]" >&2
      exit 0 ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--view-context] [--no-inspect-context] [--download] [--no-download]" >&2
      exit 1 ;;
  esac
done

# Optional env file
source .env || true

ROOT_DIR="${SRC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
NPROC_ARG="${NPROC:-$(nproc)}"

# Local canonical base tag that overlays will use
BASE_IMAGE_TAG="${BASE_IMAGE:-oss_eda_base:latest}"

# Paths
OEDA_PATH_LOCAL="${OEDA_PATH:-ext/oss_eda_flow_scripts}"

# GHCR defaults
GHCR_ORG="${GHCR_ORG:-${ORG:-huawei-csl}}"
GHCR_USER="${GHCR_USER:-MaxenceBouvier}"

#############################################
# Helpers
#############################################
read_version() {
  local f="$1"
  if [[ -f "$f" ]]; then tr -d '\n' <"$f"; else echo ""; fi
}

image_present() {
  docker image inspect "$1" >/dev/null 2>&1
}

print_context_summary() {
  local ctx="$1"
  echo "[context] Path: $ctx"
  echo "[context] Total size (KB): $(du -sk "$ctx" | cut -f1)"
  echo "[context] Total files: $(find "$ctx" -type f | wc -l)"
  echo "[context] Top-level entries:"
  ls -al "$ctx" || true
}

inspect_context() {
  local ctx="$1"
  if ! command -v docker >/dev/null 2>&1; then
    echo "[context] docker not found; showing raw filesystem stats" >&2
    print_context_summary "$ctx"
    return
  fi
  if docker buildx version >/dev/null 2>&1; then
    echo "[context] Inspecting filtered Docker context (via buildx) at: $ctx"
    if ! docker buildx build --progress=plain -f- "$ctx" <<'EOF'
# syntax=docker/dockerfile:1.4
FROM busybox
RUN --mount=type=bind,source=.,target=/context,readonly \
    sh -c 'echo "Context root: /context"; \
           echo "Total files:"; find /context -type f | wc -l; \
           echo "Total size (KB):"; du -sk /context | cut -f1; \
           echo "--- Largest 30 files (KB) ---"; \
           find /context -type f -exec du -k {} + | sort -nr | head -30; \
           echo "--- Top-level entries ---"; ls -al /context'
EOF
    then
      echo "[context] buildx inspection failed; falling back to raw filesystem stats" >&2
      print_context_summary "$ctx"
    fi
  else
    echo "[context] buildx not available; showing raw filesystem stats" >&2
    print_context_summary "$ctx"
  fi
}

#############################################
# Versions
#############################################
ver_base="$(read_version "${ROOT_DIR}/ext/oss_eda_base/.docker_image_version")"
ver_root="$(read_version "${ROOT_DIR}/.docker_image_version")"
ver_flowy="$(read_version "${ROOT_DIR}/ext/flowy/.docker_image_version")"
[[ -n "$ver_flowy" ]] || ver_flowy="$ver_root"

# Fallback to latest if base version missing
[[ -n "$ver_base" ]] || ver_base="latest"

#############################################
# Acquire base image (pull by default; never auto-build)
#############################################
if [[ "$DOWNLOAD" == true ]]; then
  # Login to ghcr.io using gh CLI or CR_PAT
  login_successful=false

  if [[ -n "${CR_PAT:-}" ]]; then
    # Use CR_PAT if explicitly provided
    echo "[auth] Using CR_PAT for authentication..."
    if echo "$CR_PAT" | docker login ghcr.io -u "$GHCR_USER" --password-stdin >/dev/null 2>&1; then
      login_successful=true
    fi
  elif command -v gh >/dev/null 2>&1; then
    # Try to use gh CLI authentication
    if gh auth status >/dev/null 2>&1; then
      echo "[auth] Using GitHub CLI (gh) authentication..."
      if gh auth token | docker login ghcr.io -u "$GHCR_USER" --password-stdin >/dev/null 2>&1; then
        login_successful=true
      fi
    else
      echo "[auth] GitHub CLI (gh) is not authenticated." >&2
      echo "[auth] Please run: gh auth login" >&2
      echo "[auth] Then try again." >&2
      exit 1
    fi
  else
    echo "[auth] No authentication method available." >&2
    echo "[auth] Please install GitHub CLI and run: gh auth login" >&2
    echo "[auth] Or set the CR_PAT environment variable." >&2
    echo "[auth] Attempting pull without authentication (may fail for private images)..." >&2
  fi

  ghcr_base="ghcr.io/${GHCR_ORG}/oss_eda_base:${ver_base}"
  echo "[base] Pulling $ghcr_base ..."
  if docker pull "$ghcr_base"; then
    docker tag "$ghcr_base" "$BASE_IMAGE_TAG"
    docker tag "$ghcr_base" "oss_eda_base:${ver_base}" || true
    echo "[base] Pulled: $ghcr_base -> $BASE_IMAGE_TAG"
  else
    echo "[error] Could not pull base image: $ghcr_base" >&2
    echo "[error] Please ensure the tag exists and that your token has the correct permissions, then try again." >&2
    exit 1
  fi
else
  # No-download mode: only allow building base if explicitly requested
  if [[ "$BUILD_BASE" == true ]]; then
    if [[ "${INSPECT_CONTEXT}" == true ]]; then
      inspect_context "${ROOT_DIR}/ext/oss_eda_base"
    fi
    echo "[base] Building local base image at ${ROOT_DIR}/ext/oss_eda_base ..."
    docker build \
      --build-arg NPROC="${NPROC_ARG}" \
      -f "${ROOT_DIR}/ext/oss_eda_base/Dockerfile" \
      -t "${BASE_IMAGE_TAG}" \
      "${ROOT_DIR}/ext/oss_eda_base"
  else
    # Expect the image to already be present locally
    if ! image_present "$BASE_IMAGE_TAG"; then
      echo "[error] BASE_IMAGE ${BASE_IMAGE_TAG} not found locally." >&2
      echo "[error] Please pull the base image from GitHub and try again." >&2
      exit 1
    fi
  fi
fi

# Ensure base really exists (defensive)
if ! image_present "$BASE_IMAGE_TAG"; then
  echo "[error] Base image ${BASE_IMAGE_TAG} is not available." >&2
  exit 1
fi

#############################################
# Download overlays if available (optional)
#############################################
have_genial=false
have_flowy=false

if [[ "$DOWNLOAD" == true ]]; then
  if [[ -n "$ver_root" ]]; then
    ghcr_genial="ghcr.io/${GHCR_ORG}/genial:${ver_root}"
    echo "[download] Attempting to pull genial: $ghcr_genial ..."
    if docker pull "$ghcr_genial"; then
      have_genial=true
      docker tag "$ghcr_genial" "genial:latest"
      docker tag "$ghcr_genial" "genial:${ver_root}" || true
      echo "[download] Pulled genial: $ghcr_genial -> genial:latest"
    else
      echo "[download] Could not pull genial overlay (will build locally)"
    fi
  fi

  if [[ -n "$ver_flowy" ]]; then
    ghcr_flowy="ghcr.io/${GHCR_ORG}/flowy:${ver_flowy}"
    echo "[download] Attempting to pull flowy: $ghcr_flowy ..."
    if docker pull "$ghcr_flowy"; then
      have_flowy=true
      docker tag "$ghcr_flowy" "flowy:latest"
      docker tag "$ghcr_flowy" "flowy:${ver_flowy}" || true
      echo "[download] Pulled flowy: $ghcr_flowy -> flowy:latest"
    else
      echo "[download] Could not pull flowy overlay (will build locally)"
    fi
  fi
fi

#############################################
# Build overlays if not downloaded
#############################################
if [[ "$INSPECT_CONTEXT" == true ]]; then
  inspect_context "${ROOT_DIR}"
fi

# GENIAL
if [[ "$have_genial" == false ]]; then
  docker build \
    --build-arg NPROC="${NPROC_ARG}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE_TAG}" \
    --build-arg SWACT_PATH="ext/swact" \
    --build-arg SWACT_COMMIT="$(git -C ext/swact rev-parse HEAD)" \
    --build-arg SWACT_BRANCH="$(git -C ext/swact rev-parse --abbrev-ref HEAD)" \
    --build-arg SWACT_REMOTE="$(git -C ext/swact config --get remote.origin.url || echo unknown)" \
    --build-arg SWACT_DIRTY="$(test -n "$(git -C ext/swact status --porcelain)" && echo 1 || echo 0)" \
    -f "${ROOT_DIR}/.devcontainer/docker/Dockerfile" \
    --target genial-latest \
    -t genial:latest \
    "${ROOT_DIR}"
fi

# FLOWY
if [[ "$have_flowy" == false ]]; then
  SUB=ext/flowy
  docker build \
    --build-arg NPROC="${NPROC_ARG}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE_TAG}" \
    --build-arg FLOWY_PATH="$SUB" \
    --build-arg FLOWY_COMMIT="$(git -C "$SUB" rev-parse HEAD)" \
    --build-arg FLOWY_BRANCH="$(git -C "$SUB" rev-parse --abbrev-ref HEAD)" \
    --build-arg FLOWY_REMOTE="$(git -C "$SUB" config --get remote.origin.url || echo unknown)" \
    --build-arg FLOWY_DIRTY="$(test -n "$(git -C "$SUB" status --porcelain)" && echo 1 || echo 0)" \
    -f "${ROOT_DIR}/${SUB}/docker/Dockerfile" \
    --target flowy-latest \
    -t flowy:latest \
    "${ROOT_DIR}"
fi
