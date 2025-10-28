from __future__ import annotations

import os
from pathlib import Path
import warnings

import pytest

try:
    import docker
    from docker.errors import DockerException
except Exception:  # pragma: no cover - docker is optional for most tests
    docker = None  # type: ignore[assignment]
    DockerException = Exception  # type: ignore[assignment]


def _docker_available() -> bool:
    if docker is None:
        return False
    try:
        client = docker.from_env()
        client.ping()
        return True
    except DockerException:
        return False


DOCKER_AVAILABLE = _docker_available()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_docker: mark test that needs a Docker daemon")
    config.addinivalue_line("markers", "flowy: mark Flowy-related tests (launch/analysis)")
    config.addinivalue_line("markers", "power: mark power-related tests (launch/analysis)")


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "requires_docker" in item.keywords and not DOCKER_AVAILABLE:
        warnings.warn("Docker daemon not available: skipping docker-dependent test", RuntimeWarning)
        pytest.skip("Docker daemon not available")


@pytest.fixture(scope="session")
def docker_client():
    if not DOCKER_AVAILABLE:
        pytest.skip("Docker daemon not available", allow_module_level=True)
    return docker.from_env()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    # tests/ is at repo_root/tests
    return Path(__file__).resolve().parents[1]


@pytest.fixture()
def env_and_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo_root: Path):
    """
    Prepare environment variables and a minimal output directory structure that
    commands like ConfigDir(is_analysis=True) expect.
    Returns a dict with paths and commonly used names.
    """
    # Point SRC_DIR to the repository root (so templates can be found)
    monkeypatch.setenv("SRC_DIR", str(repo_root))
    # Ensure Python can import the project modules when subprocesses spawn
    existing = os.environ.get("PYTHONPATH", "")
    new_path = str(repo_root / "src")
    monkeypatch.setenv("PYTHONPATH", f"{new_path}{os.pathsep}{existing}" if existing else new_path)

    # WORK_DIR to a temporary folder under pytest tmp_path
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WORK_DIR", str(work_dir))

    # Choose an experiment name that exists in templates
    experiment_name = "multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only"
    experiment_templates = repo_root / "src/genial/templates_and_launch_scripts" / experiment_name
    assert experiment_templates.exists(), f"Missing experiment templates at: {experiment_templates}"

    # Prepare a minimal output structure
    output_dir_name = "pytest_run"
    root_output_dir = work_dir / "output" / experiment_name / output_dir_name
    for sub in ("generation_out", "synth_out", "test_out", "power_out", "analysis_out"):
        (root_output_dir / sub).mkdir(parents=True, exist_ok=True)

    return {
        "SRC_DIR": repo_root,
        "WORK_DIR": work_dir,
        "experiment_name": experiment_name,
        "output_dir_name": output_dir_name,
        "root_output_dir": root_output_dir,
    }
