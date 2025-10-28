#!/usr/bin/env bash
set -euo pipefail

echo "[submodules] sync configuration (.gitmodules)"
git submodule sync --recursive

echo "[submodules] initialize to recorded commits"
git submodule update --init --recursive

# Ensure ext/flowy tracks its configured branch (not detached)
FLOWY_BRANCH="$(git config -f .gitmodules submodule.ext/flowy.branch || echo '')"
if [ -n "$FLOWY_BRANCH" ]; then
  echo "[flowy] updating to remote branch '$FLOWY_BRANCH'"
  git submodule update --remote ext/flowy
  echo "[flowy] attaching HEAD to branch '$FLOWY_BRANCH'"
  git -C ext/flowy fetch origin "$FLOWY_BRANCH" || true
  # Create or switch to the branch tracking origin
  if git -C ext/flowy rev-parse --verify "$FLOWY_BRANCH" >/dev/null 2>&1; then
    git -C ext/flowy switch "$FLOWY_BRANCH" || git -C ext/flowy checkout "$FLOWY_BRANCH"
  else
    git -C ext/flowy switch -c "$FLOWY_BRANCH" --track "origin/$FLOWY_BRANCH" \
      || git -C ext/flowy checkout -b "$FLOWY_BRANCH" --track "origin/$FLOWY_BRANCH"
  fi
  # Fast-forward to remote tip if needed
  git -C ext/flowy pull --ff-only origin "$FLOWY_BRANCH" || true
else
  echo "[flowy] no branch configured in .gitmodules; leaving as-is (may be detached)"
fi

echo "Done. ext/flowy now on branch '${FLOWY_BRANCH:-<none>}'"
