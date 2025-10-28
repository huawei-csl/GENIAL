#!/usr/bin/env bash
set -euo pipefail

TARGET_USER=${TARGET_USER:-vscode}
TARGET_HOME=$(eval echo ~${TARGET_USER})
UID_ENV=${LOCAL_UID:-}
GID_ENV=${LOCAL_GID:-}

if [[ -z "${UID_ENV}" || -z "${GID_ENV}" ]]; then
  echo "LOCAL_UID/GID not set; skipping UID/GID update."
  exit 0
fi

CURRENT_UID=$(id -u "${TARGET_USER}")
CURRENT_GID=$(id -g "${TARGET_USER}")
changed=0

if [[ "${CURRENT_GID}" != "${GID_ENV}" ]]; then
  echo "Updating ${TARGET_USER} GID: ${CURRENT_GID} -> ${GID_ENV}"
  if getent group "${GID_ENV}" >/dev/null 2>&1; then
    # Use existing group with target GID
    usermod -g "${GID_ENV}" "${TARGET_USER}"
  else
    groupmod -g "${GID_ENV}" "${TARGET_USER}"
  fi
  changed=1
fi

if [[ "${CURRENT_UID}" != "${UID_ENV}" ]]; then
  echo "Updating ${TARGET_USER} UID: ${CURRENT_UID} -> ${UID_ENV}"
  usermod -u "${UID_ENV}" "${TARGET_USER}"
  changed=1
fi

if [[ "$changed" == "1" ]]; then
  echo "Fixing ownership of ${TARGET_HOME} and common work dirs."
  chown -R "${UID_ENV}:${GID_ENV}" "${TARGET_HOME}" || true
  for d in /app /prog; do
    if [[ -d "$d" ]]; then
      chown -R "${UID_ENV}:${GID_ENV}" "$d" || true
    fi
  done
else
  echo "UID/GID already match; no ownership changes needed."
fi

echo "UID/GID update complete."
