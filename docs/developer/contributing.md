# Contributing to GENIAL

Welcome, and thank you for your interest in contributing to **GENIAL — Design Generation From Encodings**!

This document outlines best practices, coding standards, and contribution procedures for the GENIAL project. Whether you’re fixing a bug, improving documentation, or developing a new feature, we appreciate your efforts.

---

## 🚀 Quick Overview

| Task | How to do it |
|------|--------------|
| Set up dev environment | Use Python 3.13 + [`uv`](https://astral.sh/uv) |
| Install repo + dev deps | `uv pip install -e .` |
| Run linter / formatter | `pre-commit run --all-files` |
| Run tests | `pytest -q` |
| Write docs | Update README or files in `docs/` |
| Submit change | Open a Pull Request on GitHub |

---

## 🛠️ Development Environment

We recommend the following setup:

1. Clone the repo and initialize submodules:

   ```bash
   git clone --recursive https://github.com/<org>/genial.git
   cd genial
   # If not cloned with --recursive:
   git submodule update --init --recursive
````

2. Create a Python 3.13 virtual environment using [`uv`](https://astral.sh/uv/):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv --python 3.13 envs/313_genial
   source envs/313_genial/bin/activate
   ```

3. Install the repo in editable mode with dev dependencies:

   ```bash
   uv pip install -e .
   ```

4. (Optional) Install `pre-commit` hooks to automatically run formatters and linters before each commit:

   ```bash
   pre-commit install
   ```

---

## 📦 Code Layout

| Path                                                  | Description                                          |
| ----------------------------------------------------- | ---------------------------------------------------- |
| `src/genial/`                              | Core code: generation, synthesis, analysis, training |
| `src/genial/templates_and_launch_scripts/` | Experiment definitions                               |
| `scripts/`                                            | Workflow scripts for automation                      |
| `tests/`                                              | Unit tests and fixtures                              |
| `docs/`                                               | Markdown-based documentation and user guides         |

---

## 🧪 Testing Guidelines

* Use `pytest` for writing and running tests.
* Keep tests fast, hermetic, and reproducible.
* Avoid requiring Docker or external flows by default — mock or isolate if possible.
* If a test **requires Docker**, **mark it** accordingly (see `tests/conftest.py`) and skip when unavailable.

Run the test suite with:

```bash
pytest -q
```

You can also run only marked tests:

```bash
pytest -m docker_required
```

---

## 🧹 Linting & Formatting

We use [`ruff`](https://docs.astral.sh/ruff/) for both linting and formatting.

* Formatting: `ruff format .`
* Linting: `ruff check .`

All are automatically handled if you installed the pre-commit hook.

To run manually:

```bash
ruff format .
ruff check .
```

---

## 📚 Documentation

* User documentation is in [`docs/`](docs/)
* Diagrams and architecture are explained in the same folder
* Please update the relevant file if your contribution changes user-facing behavior
* For large additions, consider writing a new file and linking it from the main docs

---

## 🧠 Coding Style & Practices

* Follow [PEP8](https://peps.python.org/pep-0008/) and general Python best practices
* Use type hints where helpful, especially in core modules
* Prefer clear, expressive variable names over short cryptic ones
* Modularize complex logic into small functions
* Write docstrings for all public functions/classes

### File/Folder Naming

* Use `snake_case` for Python files
* Use `UPPER_CASE` for constants
* Use `kebab-case` for experiment names (e.g., `multiplier-4bi-8bo-default`)

---

## 📥 Submitting a Contribution

1. **Create a feature branch**:

   ```bash
   git checkout -b feat/<short-description>
   ```

2. **Make your changes**, and run:

   ```bash
   pre-commit run --all-files
   pytest -q
   ```

3. **Commit with a clear message**:

   ```bash
   git commit -m "feat(analyzer): add support for XYZ tracking"
   ```

4. **Push your branch**:

   ```bash
   git push origin feat/<short-description>
   ```

5. **Open a Pull Request** on GitHub:

   * Include a short summary
   * Mention related issues if any
   * Describe any user-facing behavior changes
   * Add screenshots/logs if helpful

---

## 📣 Larger Features or Refactors

For significant changes, please open a GitHub Issue first to discuss design and impact.

This helps:

* Avoid redundant work
* Ensure alignment with project roadmap
* Get early feedback

---

## 🧾 Checklists

### Before opening a PR

* [ ] I created/updated relevant tests
* [ ] I ran `pytest` and all tests pass
* [ ] I ran `pre-commit run --all-files`
* [ ] I updated `README.md` or `docs/` if needed
* [ ] I scoped the PR to one logical change
* [ ] I explained the reasoning in the PR description

---

## 📜 License and Contributor Agreement

This project is licensed under the **BSD 3-Clause Clear License**. By contributing, you agree that:

* Your code will be released under the same license
* You have the right to contribute the code (e.g., no IP conflicts)
* You do not include proprietary or sensitive information

---

## 🙏 Thank You

We appreciate all forms of contribution — from fixing typos and improving examples to implementing full features. Your help makes GENIAL better for everyone!

For questions or help getting started, feel free to open an [issue](https://github.com/<org>/genial/issues) or reach out directly.
