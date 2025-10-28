#!/bin/bash
set -euo pipefail

# Push local docker images to GHCR for the organization.
#
# Defaults:
# - ORG/GHCR_ORG: huawei-csl
# - GHCR_USER: auto-detected from gh (override via env)
# - Images pushed if none specified: oss_eda_base, genial, flowy
# - Version source:
#     oss_eda_base -> ext/oss_eda_base/.docker_image_version
#     genial        -> ./.docker_image_version
#     flowy       -> ext/flowy/.docker_image_version or ./.docker_image_version
#
# Auth (via GitHub CLI):
#   gh auth login --scopes 'read:packages,write:packages,delete:packages'
#   # Script uses `gh auth token` to log in to GHCR.
#   # You may still set `GHCR_USER` to override the detected username.
#
# Legacy fallback (not recommended):
#   export CR_PAT=...  # GitHub personal access token with package:write
#   # If `gh` is unavailable, the script will try CR_PAT.

# Examples:
#   # Push all default images (oss_eda_base, genial, flowy)
#   .devcontainer/docker/push_dockers.sh
#
#   # Push a specific subset
#   .devcontainer/docker/push_dockers.sh oss_eda_base genial
#
#   # Override org/user for GHCR
#   GHCR_ORG=my-org GHCR_USER=my-user .devcontainer/docker/push_dockers.sh flowy
#
#   # Use CR_PAT fallback (if gh unavailable)
#   CR_PAT=ghp_xxx .devcontainer/docker/push_dockers.sh

ROOT_DIR="${SRC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GHCR_ORG="${GHCR_ORG:-${ORG:-huawei-csl}}"
GHCR_USER="${GHCR_USER:-}"

read_version() {
  local f="$1"
  if [[ -f "$f" ]]; then
    tr -d '\n' <"$f"
  else
    echo ""
  fi
}

login_with_gh() {
  if ! command -v gh >/dev/null 2>&1; then
    echo "[push] gh CLI not found. Install it or set CR_PAT as fallback." >&2
    return 2
  fi

  if ! gh auth status >/dev/null 2>&1; then
    echo "[push] gh not authenticated. Run: gh auth login --scopes 'read:packages,write:packages,delete:packages'" >&2
    return 2
  fi

  local user token
  if [[ -n "${GHCR_USER:-}" ]]; then
    user="$GHCR_USER"
  else
    user="$(gh api /user -q .login 2>/dev/null || true)"
    if [[ -z "$user" ]]; then
      echo "[push] Unable to determine GitHub username from gh. Set GHCR_USER." >&2
      return 2
    fi
  fi

  if ! token="$(gh auth token 2>/dev/null)" || [[ -z "$token" ]]; then
    echo "[push] Failed to obtain token via 'gh auth token'. Ensure gh is up to date and authenticated." >&2
    return 2
  fi

  echo "[push] Logging in to ghcr.io as $user using gh token"
  echo "$token" | docker login ghcr.io -u "$user" --password-stdin
}

login_if_possible() {
  # Prefer gh-based login
  if login_with_gh; then
    return 0
  fi

  # Fallback to CR_PAT if present
  if [[ -n "${CR_PAT:-}" ]]; then
    local user_fallback
    user_fallback="${GHCR_USER:-${GITHUB_USER:-${USER:-unknown}}}"
    echo "[push] Falling back to CR_PAT for docker login as $user_fallback"
    echo "$CR_PAT" | docker login ghcr.io -u "$user_fallback" --password-stdin
  else
    echo "[push] No gh auth and no CR_PAT. Attempting pushes without login (will fail for private repos)."
  fi
}

push_one() {
  local image_name="$1"  # logical name: oss_eda_base | genial | flowy
  local local_tag
  local ver_file

  case "$image_name" in
    oss_eda_base)
      local_tag="oss_eda_base:latest"
      ver_file="${ROOT_DIR}/ext/oss_eda_base/.docker_image_version"
      ;;
    genial)
      local_tag="genial:latest"
      ver_file="${ROOT_DIR}/.docker_image_version"
      ;;
    flowy)
      local_tag="flowy:latest"
      if [[ -f "${ROOT_DIR}/ext/flowy/.docker_image_version" ]]; then
        ver_file="${ROOT_DIR}/ext/flowy/.docker_image_version"
      else
        ver_file="${ROOT_DIR}/.docker_image_version"
      fi
      ;;
    *)
      echo "[push] Unknown image name: $image_name" >&2
      return 1 ;;
  esac

  if ! docker image inspect "$local_tag" >/dev/null 2>&1; then
    # Backward compatibility: older builds may have used hyphenated name
    if [[ "$image_name" == "oss_eda_base" ]] && docker image inspect "oss-eda-base:latest" >/dev/null 2>&1; then
      echo "[push] Retagging legacy 'oss-eda-base:latest' -> 'oss_eda_base:latest'"
      docker tag "oss-eda-base:latest" "$local_tag"
    else
      echo "[push] Local image not found: $local_tag" >&2
      return 1
    fi
  fi

  local ver
  ver="$(read_version "$ver_file")"
  if [[ -z "$ver" ]]; then
    echo "[push] Version file missing/empty for $image_name: $ver_file" >&2
    return 1
  fi

  local ghcr_ref_ver="ghcr.io/${GHCR_ORG}/${image_name}:${ver}"
  local ghcr_ref_latest="ghcr.io/${GHCR_ORG}/${image_name}:latest"

  echo "[push] Tagging $local_tag -> $ghcr_ref_ver and :latest"
  docker tag "$local_tag" "$ghcr_ref_ver"
  docker tag "$local_tag" "$ghcr_ref_latest"

  echo "[push] Pushing $ghcr_ref_ver"
  docker push "$ghcr_ref_ver"
  echo "[push] Pushing $ghcr_ref_latest"
  if ! docker push "$ghcr_ref_latest"; then
    echo "[push] Warning: failed to push tag 'latest' for $image_name (continuing)." >&2
  fi
}

main() {
  # Images to push: args or default list
  if [[ $# -gt 0 ]]; then
    images=("$@")
  else
    images=(oss_eda_base genial flowy)
  fi

  login_if_possible

  for img in "${images[@]}"; do
    if ! push_one "$img"; then
      echo "[push] Error while pushing '$img' (continuing with next)." >&2
    fi
  done
}

main "$@"
