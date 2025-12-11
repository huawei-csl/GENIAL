# GENIAL overlay image
ARG BASE_IMAGE=oss-eda-base:latest

# --- builder to package swact as a wheel (no source in final) ---
FROM ${BASE_IMAGE} AS swact-builder
SHELL ["/bin/bash","-lc"]

# Build the wheel as root to avoid permission/timestamp issues with egg-info
USER root
WORKDIR /tmp/swact

# bring only the minimal files needed to build the wheel
ARG SWACT_PATH=ext/swact
COPY ${SWACT_PATH}/pyproject.toml ./
COPY ${SWACT_PATH}/src ./src

# Build wheel with uv (uses the shared /home/vscode/pyenv_eda env from the base)
# - remove any pre-existing egg-info dirs (copied from workspace)
# - make files writable
# - create a LICENSE if missing (silences setuptools warnings)
RUN set -eux; \
    find . -maxdepth 2 -type d -name '*.egg-info' -exec rm -rf {} +; \
    chmod -R u+rwX,g+rwX .; \
    [ -f LICENSE ] || echo 'SPDX-License-Identifier: MIT' > LICENSE; \
    PY=/prog/pyenv_eda/bin/python; \
    uv build --python="$PY" --wheel; \
    ls -l dist

# --- final image ---
FROM ${BASE_IMAGE} AS genial-latest
SHELL ["/bin/bash","-lc"]
WORKDIR /app
# base image already defaults to USER vscode

LABEL org.opencontainers.image.title="GENIAL" \
      org.opencontainers.image.description="GENIAL development image" \
      org.opencontainers.image.ref.name="genial:latest"

# Optionally expose SWACT git build metadata (very small)
ARG SWACT_COMMIT=unknown
ARG SWACT_BRANCH=unknown
ARG SWACT_REMOTE=unknown
ARG SWACT_DIRTY=0
ENV SWACT_COMMIT=${SWACT_COMMIT} \
    SWACT_BRANCH=${SWACT_BRANCH} \
    SWACT_REMOTE=${SWACT_REMOTE} \
    SWACT_DIRTY=${SWACT_DIRTY}

# Ensure venv python is on PATH for convenience (also available as /usr/local/bin/python)
ENV PATH=/home/vscode/pyenv_eda/bin:/usr/local/bin:$PATH

# Install swact wheel (no source tree kept)
RUN mkdir -p /app/swact/.build
# IMPORTANT: ensure vscode owns the wheel to allow cleanup
COPY --from=swact-builder --chown=vscode:vscode /tmp/swact/dist/*.whl /tmp/wheels/
RUN set -eux; \
    PY=/home/vscode/pyenv_eda/bin/python; \
    uv pip install --python="$PY" --no-cache-dir /tmp/wheels/*.whl; \
    rm -rf /tmp/wheels

# Preinstall GENIAL deps (filter out flowy) and clean tmp in same layer
COPY --chown=vscode:vscode requirements.txt requirements_extra.txt /tmp/pip/
RUN set -eux; \
    PY=/home/vscode/pyenv_eda/bin/python; \
    if [ -f /tmp/pip/requirements.txt ]; then \
      sed '/^flowy\s*$/d' /tmp/pip/requirements.txt > /tmp/pip/requirements.preinstall.txt; \
      uv pip install --python="$PY" --no-cache-dir -r /tmp/pip/requirements.preinstall.txt; \
    fi; \
    if [ -f /tmp/pip/requirements_extra.txt ]; then \
      uv pip install --python="$PY" --no-cache-dir -r /tmp/pip/requirements_extra.txt || true; \
    fi; \
    rm -rf /tmp/pip
