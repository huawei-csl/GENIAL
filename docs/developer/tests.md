# 🧪 Test Suite Overview

This document provides an overview of the test suite located in the `tests/` directory. Use it as a quick reference when extending, debugging, or exploring the repository's test coverage.

---

## 📁 Fixtures & Helpers

These shared utilities support the test infrastructure:

- **`conftest.py`**  
  Defines global `pytest` fixtures and configuration. It:
  - Sets up temporary working directories
  - Configures environment variables
  - Optionally provides a Docker client (for tests requiring a daemon)

- **`test_file.txt`**  
  A minimal placeholder text file for tests requiring file I/O.

---

## 🧩 Test Modules

Each test module targets a specific functional area:

| File | Purpose |
|------|---------|
| **`test_analyzer.py`** | Tests the analyzer CLI entry point, including `--help` and a smoke test for complexity-only mode. |
| **`test_docker.py`** | Validates Docker daemon availability. Tests are skipped if Docker is not available. |
| **`test_flowy.py`** | Verifies Flowy-enabled experiments: launcher flags, CLI flow simulation, and lightweight test paths. |
| **`test_launcher.py`** | Covers the main launcher: flag logic, `--help` output, and directory generation checks. |
| **`test_loop.py`** | Smoke test for `full_run_v2.py`. Confirms it starts and displays usage info without errors. |
| **`test_power.py`** | Tests power-only execution paths: skips synthesis/simulation while running power analysis. |
| **`test_recommender.py`** | Covers recommender CLI (`recommender.py`) and result-processing utilities. |
| **`test_scripts.py`** | Assorted scripts and utilities: status counting, compression round-trips, and script consistency checks. |
| **`test_setup_docs.py`** | Verifies that `docs/setup.md` exists. Skipped on CI. |
| **`test_trainer.py`** | Ensures the training CLI (`trainer_enc_to_score_value.py`) exposes valid `--help` output. |

> ✅ These tests offer lightweight assurance that CLI tools and core utilities behave as expected and that essential project files exist.

---

## 🛠️ Debugging Tips

### 🔹 Common Commands

| Task | Command |
|------|---------|
| Run a single test | `pytest -q tests/test_flowy.py::test_flow` |
| Filter by keyword | `pytest -k test_flow -vv` |
| Show output (print) | `pytest -s -vv tests/test_flowy.py::test_flow` |
| Stop after first failure | `pytest --maxfail=1 -x` |
| Re-run last failures | `pytest --lf` |

### 🔹 Interactive Debugging

| Task | Command |
|------|---------|
| Drop into debugger on failure | `pytest --pdb -k test_flow` |
| Start debugger at test start | `pytest --trace tests/test_flowy.py::test_flow` |
| Full tracebacks | `pytest --full-trace` |
| Manual breakpoints | Insert `breakpoint()` and run normally |

### 🔹 Visibility & Logs

| Task | Command |
|------|---------|
| Live logs in console | `pytest --log-cli-level=INFO -vv` |
| Show fixture setup | `pytest --setup-show -k test_flow` |
| Identify slowest tests | `pytest --durations=10 -vv` |

### 🔹 Test Collection

| Task | Command |
|------|---------|
| List available tests | `pytest --collect-only -q | rg test_flow` |
| Use node ID directly | `tests/test_flowy.py::test_flow` |

---

## 🧑‍💻 VS Code Integration

Make debugging easier with `.vscode/launch.json`:

```jsonc
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run current test file",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["-s", "${file}"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      },
      "justMyCode": false
    },
    {
      "name": "Debug specific test",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["-s", "tests/test_flowy.py::test_flow"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      },
      "justMyCode": false
    }
  ]
}
