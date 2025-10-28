# Setup

Detailed steps to configure the development environment.

## Environment variables

Copy `.env.template` to `.env` and set:

- `SRC_DIR`: absolute path to the repository root
- `WORK_DIR`: location for output data
- Optional: `NPROC` to control native build parallelism

Note: The base image now includes `oss_eda_flow_scripts` from the repo. Ensure submodules are initialized; no token is needed unless your submodule origin requires it.

## Install dependencies

Use [uv](https://astral.sh/uv) to manage Python 3.13 dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.13 envs/313_genial
source envs/313_genial/bin/activate
uv pip install -e .
```
These commands install the `uv` tool, create a Python 3.13 environment, activate it and install the project in editable mode so local changes take effect immediately.

## Pre-commit hooks

Install git hooks to run linting automatically:

```bash
pre-commit install
```
Running this sets up git hooks that lint and format staged files before each commit.

## Docker images

We now use a shared base image defined in `ext/oss_eda_base/Dockerfile` which bakes in `oss_eda_flow_scripts`. Build the base once, then build project overlays.

```bash
# 0) Ensure submodules are present (includes flow scripts under ext/oss_eda_base)
git submodule update --init --recursive

# 1) Build shared EDA base (expects submodule at ext/oss_eda_base/ext/oss_eda_flow_scripts)
#    Use minimal context to avoid sending the whole repo
docker build \
  -f ext/oss_eda_base/Dockerfile \
  -t oss-eda-base:latest \
  --build-arg NPROC=${NPROC:-$(nproc)} \
  ext/oss_eda_base

# 2) Build GENIAL image (inherits from the base)
docker build \
  -t genial:latest \
  -f .devcontainer/docker/Dockerfile \
  --target genial-latest \
  --build-arg BASE_IMAGE=oss-eda-base:latest \
  .

# 3) Build Flowy dev image
bash ext/flowy/docker/build_docker.sh
# or manual:
docker build \
  -f ext/flowy/docker/Dockerfile \
  -t flowy:latest \
  --build-arg NPROC=${NPROC:-$(nproc)} \
  --build-arg BASE_IMAGE=oss-eda-base:latest \
  .
```

Notes:
- The devcontainer (`.devcontainer/devcontainer.json`) and helper script build the base from `ext/oss_eda_base/Dockerfile` and expect `ext/oss_eda_base/ext/oss_eda_flow_scripts` to exist. The helper script uses minimal context.
- Flowy’s Dockerfile relies on `/app/oss_eda_flow_scripts` from the base; ensure submodules are initialized.
- If you change `requirements.txt` files, rebuild the corresponding image to pick up updates.

Helper script (recommended):

```bash
# Build base only when flow scripts are present; force with --build-base
./.devcontainer/docker/build_docker.sh --build-base     # force base build
./.devcontainer/docker/build_docker.sh                  # skip if no local flow scripts

# Optional: show filtered Docker build contexts (root and base)
./.devcontainer/docker/build_docker.sh --view-context
```

## Submodules

This repo uses submodules for tooling and Flowy. After cloning, run:

```bash
# Ensure submodule branch configuration is applied
git submodule sync --recursive

# Initialize and update Flowy to the configured branch
git submodule update --init --remote --checkout ext/flowy

# Initialize the remaining submodules
git submodule update --init --recursive
```

The Flowy submodule is configured to track branch:

```
submodule "ext/flowy" branch = dev/cleanin_up_for_open_sourcing
```

You can also run the helper script:

```bash
bash scripts/update_submodules.sh
```

## Extra requirements

Optional packages for analysis and visualization are listed in `requirements_extra.txt`. Install them with:

```bash
uv pip install -r requirements_extra.txt
```
Install these optional dependencies when you want additional visualization or analysis utilities.

## Secrets

The project uses [python-dotenv](https://github.com/theskumar/python-dotenv) to load environment variables from a `.env` file.
Those are not secrets!

Secrets are expected to be placed in `$HOME/.config/genial/.secrets`.
A template secret file indicating which secrets are expected can be found in `.secrets.template`.
The user has to manually create the file and fill in the secrets.

Some helper commands:
```bash
# Create the secrets directory
mkdir -p $HOME/.config/genial
# Copy the template to the secrets file
cp .secrets.template $HOME/.config/genial/.secrets
# Edit the secrets file
nano $HOME/.config/genial/.secrets
```
