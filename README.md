
<div align="center">

# GENIAL — Generative Design Space Exploration via Network Inversion for Low Power Algorithmic Logic Units

[![Python](https://img.shields.io/badge/python-3.13-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause--Clear-blue)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-supported-informational)](.devcontainer/docker/Dockerfile)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)](tests)

</div>

**GENIAL** is a research toolkit for exploring how *signal encodings* impact hardware characteristics such as switching activity, area, and power. It automates the full exploration loop:

- 🧬 Generate combinational designs from encoding specs
- ⚙️ Synthesize with Yosys/ABC or Flowy pipelines
- 📉 Simulate and collect switching activity
- 🔋 Extract power using OpenROAD/OpenSTA
- 📊 Analyze results and train encoding recommenders


> 🔬 GENIAL was built for leading research on generative encoding for low-activity logic, at the Huawei Von Neumann Research Center in Zurich in Switzerland. 
> 📖 You can find the related article at the following link:
>
> [GENIAL: Generative Design Space Exploration via Network Inversion for Low Power Algorithmic Logic Units (arXiv:2507.18989)](https://arxiv.org/abs/2507.18989)
> 
> If you use GENIAL in work, please give us a star ⭐, and cite it! Thank you, and enjoy!

<div align="center">
  <img src="docs/images/genial_overview.png" alt="GENIAL Framework Overview" style="max-width: 1200px;">
</div>


## 🏁 Table of Content
- [GENIAL — Generative Design Space Exploration via Network Inversion for Low Power Algorithmic Logic Units](#genial--generative-design-space-exploration-via-network-inversion-for-low-power-algorithmic-logic-units)
  - [🏁 Table of Content](#-table-of-content)
  - [🔍 Available Features](#-available-features)
  - [⚙️ Pipeline](#️-pipeline)
  - [✨ Key Features](#-key-features)
  - [🚀 Quick Start](#-quick-start)
    - [0. Clone the repo and its submodules](#0-clone-the-repo-and-its-submodules)
    - [1. Configure environment](#1-configure-environment)
    - [2. Set up Python environment with `uv`](#2-set-up-python-environment-with-uv)
    - [3. Install GENIAL as a package](#3-install-genial-as-a-package)
    - [4. Build Docker images](#4-build-docker-images)
    - [5. \[Developper\] Enable pre-commit hooks](#5-developper-enable-pre-commit-hooks)
  - [🧪 Examples](#-examples)
    - [Generate 1 design (no synthesis/simulation):](#generate-1-design-no-synthesissimulation)
    - [Run a minimal end‑to‑end loop (debug mode):](#run-a-minimal-endtoend-loop-debug-mode)
  - [🗂️ Repository Structure](#️-repository-structure)
  - [📚 Documentation](#-documentation)
  - [📖 Citation](#-citation)
  - [🙏 Acknowledgements](#-acknowledgements)
  - [⚖️ Legal](#️-legal)
  - [🎡 Interdependency Graph](#-interdependency-graph)
  - [🤝 Contributing](#-contributing)

---

## 🔍 Available Features

- **Design families:** Adders, multipliers, encoders, decoders, FSMs
- **Encoding schemes:** Two’s complement, unsigned, mixed, one-hot, permuted, etc.
- **Backends:** Yosys, Mockturtle, ABC, Verilator/Cocotb, OpenROAD/OpenSTA
- **Extra:** Flowy (optional) for advanced logic synthesis
- **Toolkit features:** Experiment templating, batching, tracking, analysis, and model training

---

## ⚙️ Pipeline

```mermaid
%%{init: {'flowchart': {'htmlLabels': false}} }%%
flowchart LR
  A["Encoding Specs (Templates + Config)"] --> B["Design Generator"]
  B --> C{"Synthesis"}
  C --> C1["Gate-Level Netlists"]
  C --> C2["Optimized Netlists (Flowy)"]
  C1 --> D["Simulation + SwAct"]
  C2 --> D
  D --> E["Power Extraction (OpenROAD/OpenSTA)"]
  E --> F["Analyzer + DB"]
  F --> G["Trainer / Recommender"]
  G -. "feedback" .-> B
```

---

## ✨ Key Features

* 🧪 **Modular experiment templates** – Configurable by design family, encoding scheme, toolchain
* ⚡ **Parallelized runners** – For synthesis, simulation (SwAct), power extraction
* 🔁 **Efficient state tracking** – Resume, skip, or re-run steps as needed
* 🤖 **Analysis and ML-ready** – Rich output database, training hooks for recommender models

---

## 🚀 Quick Start

### 0. Clone the repo and its submodules

```bash
# 1. Clone the repository without its submodules
git clone https://github.com/huawei-csl/genial.git

# 2. Initialize submodules manually:
git -c submodule.ext/flowy.update=none submodule update --init --recursive
```
> *Note:* for now, Flowy is not yet open-source. We thus specifically avoid cloning it.

### 1. Configure environment

Copy and edit your `.env`:

```bash
cp .env.template .env
# Then edit: set SRC_DIR, WORK_DIR
```

Get the pre-trained weights from the release files and put them in the resources:
```bash
curl -L -o resources/pretrained_model/embedding/117_0.0102_000.ckpt https://github.com/huawei-csl/GENIAL/releases/download/v0.1.0/117_0.0102_000.ckpt
```

### 2. Set up Python environment with [`uv`](https://astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.13 envs/313_genial
source envs/313_genial/bin/activate
```

If it's the first time you install uv, you'll have to restart your shell.

### 3. Install GENIAL as a package

```bash
# 0. For fresh ubuntu installs [Optional]:
# sudo apt-get update
# sudo apt-get install -y build-essential gcc g++ make

# 1. Install GENIAL with dev dependencies:
uv pip install -e .

# 2. [Optional] If you have access to flowy:
# uv pip install -e ext/flowy

```

### 4. Build Docker images

This will take a while.
While it's running, you can read this repository documentation further.
We recommend reading it in the order suggested as in the [usage documentation](#-usage-documentation) section.

You can also read the [GENIAL paper](https://arxiv.org/abs/2507.18989) to get a better idea of what this repository will enable you to do.

```bash
./.devcontainer/docker/build_dockers.sh --build-base [--no-download] [--sequential]
```

**Options:**
- `--no-download`: Force the oss_eda_base docker image to be built from scratch instead of downloading the pre-built one
- `--sequential` or `-s`: Limit parallelism for machines with limited resources (useful for WSL or systems with few CPU cores to prevent freezing)

### 5. [Developper] Enable pre-commit hooks

```bash
pre-commit install
```

📄 See [docs/setup.md](docs/setup.md) for additional instructions.

---

## 🧪 Examples

### Generate 1 design (no synthesis/simulation):

```bash
python -m genial.experiment.task_launcher \
  --experiment_name multiplier_2bi_4bo_permuti_allcells_notech_fullsweep_only \
  --output_dir_name demo \
  --only_gener --nb_new_designs 1
```

### Run a minimal end‑to‑end loop (debug mode):

```bash
python -m genial.experiment.task_launcher \
  --experiment_name multiplier_2bi_4bo_permuti_allcells_notech_fullsweep_only \
  --output_dir_name demo_run \
  --debug --nb_workers 1
```

---

## 🗂️ Repository Structure

| Path                                                  | Description                                             |
| ----------------------------------------------------- | ------------------------------------------------------- |
| `src/genial/`                              | Core logic: generator, launcher, analyzer, ML training  |
| `src/genial/templates_and_launch_scripts/` | Experiment templates and launch scripts                 |
| `scripts/`                                            | Ad-hoc loop helpers and automation scripts              |
| `tests/`                                              | Unit tests, fixtures, and CI-related testing            |
| `docs/`                                               | Setup guides, architecture explanations, usage examples |

---

## 📚 Documentation

Here, you will find all you need to understand how to use GENIAL, module by module.
We highly recommend to have a look at the different READMEs to find useful commands and what GENIAL can do for you.

* [Setup](docs/setup.md)
* [Launcher](docs/launcher.md)
* [Analyzer](docs/analyzer.md)
* [Training](docs/training.md)
* [Recommender / Network Inversion](docs/recommender.md)
* [Loop Execution](docs/loop.md)
* [Utility Scripts](docs/scripts.md)
* [Switching Activity Model](docs/switching_activity.md)

---

## 📖 Citation

```
@inproceedings{genial2026,
  author    = {Maxence Bouvier and Ryan Amaudruz and Felix Arnold and Renzo Andri and Lukas Cavigelli},
  title     = {{GENIAL}: Generative Design Space Exploration via Network Inversion for Low Power Algorithmic Logic Units},
  booktitle = {Proceedings of the 31st Asia and South Pacific Design Automation Conference (ASPDAC)},
  year      = {2026},
  note      = {To appear},
  url       = {https://arxiv.org/abs/2507.18989}
}
```

---

## 🙏 Acknowledgements

GENIAL builds upon a vibrant open-source ecosystem, all licenses and copyright notices have been kept intact:

* 🧠 [Yosys](https://yosyshq.net/yosys/) – Logic synthesis
* 🔧 [ABC](https://people.eecs.berkeley.edu/~alanmi/abc/) – Logic optimization
* 🕸️ [Mockturtle](https://github.com/lsils/mockturtle) – MIG-based Logic optimization
* 🏗️ [OpenROAD](https://theopenroadproject.org/), [OpenSTA](https://github.com/The-OpenROAD-Project/OpenSTA) – PnR and STA
* 🔍 [Verilator](https://www.veripool.org/verilator/), [Cocotb](https://www.cocotb.org/) – Simulation
* 🧮 [PyTorch](https://pytorch.org/), [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) – Surrogate model training
* 📚 [SOAP](https://github.com/nikhilvyas/SOAP), [LAMB](https://huggingface.co/spaces/Roll20/pet_score/blob/9e46325ff5d82df348bad5b4a235eac8410959b8/lib/timm/optim/lamb.py)

---

## ⚖️ Legal

- GENIAL is distributed under the [BSD 3-Clause Clear License](https://github.com/huawei-csl/GENIAL/blob/main/LICENSE.md).
- Third-party components bundled or referenced by the project remain under their respective licenses; ensure you review and honor those requirements.

---

## 🎡 Interdependency Graph
The following graph shows the dependencies between the different repositories used for running the full flow.

➡️ **Note:** The `flowy` submodule is optional and not available yet.

```mermaid
graph TD

  GENIAL["GENIAL"]
  Flowy["Flowy (optional)"]
  OSSBase["oss_eda_base"]
  OSSFlow["oss_eda_flowscripts"]
  Mockturtle["mockturtle"]

  GENIAL -.-> Flowy
  GENIAL --> OSSBase
  OSSBase --> OSSFlow
  Flowy --> OSSBase
  Flowy --> Mockturtle

  click GENIAL "https://github.com/huawei-csl/GENIAL" _blank
  click Flowy "https://github.com/huawei-csl/FLOWY" _blank
  click OSSBase "https://github.com/huawei-csl/oss_eda_base" _blank
  click OSSFlow "https://github.com/huawei-csl/oss_eda_flowscripts" _blank
  click Mockturtle "https://github.com/huawei-csl/mockturtle" _blank
```

## 🤝 Contributing

We welcome contributions of all kinds! To get started:

* **Environment:** Use Python 3.13 with [`uv`](https://astral.sh/uv/). Install the repo in editable mode with dev dependencies using:
  `uv pip install -e .`
* **Pre-commit:** Install hooks with `pre-commit install` to enable automatic linting and formatting (via `ruff`).
* **Tests:** Add or modify tests under `tests/`. Prefer fast, hermetic tests. For heavy flows (e.g. Docker-based), mark them appropriately and skip if Docker is unavailable (see `tests/conftest.py`).
* **Branching & PRs:** Work in a feature branch. Keep changes focused, write clear commit messages, and include a short rationale and any user-facing changes in your PR description.
* **Docs:** Update `README.md` and relevant `docs/` files if your contribution changes usage, behavior, or setup.

For larger features or architectural changes, please open an issue first to discuss scope and design.

See [docs/developer/contributing.md](docs/developer/contributing.md) for full contribution guidelines.

---
