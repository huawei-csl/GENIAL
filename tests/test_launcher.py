from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import json

import pytest

# Delay heavy imports until after env is prepared by fixtures
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from genial.config.config_dir import ConfigDir  # noqa: F401
    from genial.experiment.task_launcher import Launcher  # noqa: F401


def run(cmd, env=None):
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_task_launcher_help(env_and_paths):
    env = os.environ.copy()
    proc = run([sys.executable, "src/genial/experiment/task_launcher.py", "--help"], env=env)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    "case_name,experiment_name,extra_args,expected_log,expected_flags",
    [
        (
            "full",
            "multiplier_2bi_4bo_permuti_allcells_notech_fullsweep_only",
            {},
            "launcher_full.log",
            {"skip_synth": False, "skip_swact": False, "skip_power": False},
        ),
        (
            "synth_only",
            "multiplier_2bi_4bo_permuti_allcells_notech_fullsweep_only",
            {"synth_only": True},
            "launcher_synth.log",
            {"skip_synth": False, "skip_swact": True, "skip_power": True},
        ),
        (
            "swact_only",
            "multiplier_2bi_4bo_permuti_allcells_notech_fullsweep_only",
            {"skip_synth": True},
            "launcher_swact.log",
            {"skip_synth": True, "skip_swact": False, "skip_power": False},
        ),
    ],
)
def test_launcher_init_configs(
    env_and_paths, tmp_path: Path, case_name, experiment_name, extra_args, expected_log, expected_flags
):
    from genial.config.config_dir import ConfigDir
    from genial.experiment.task_launcher import Launcher

    # Unique output dir per case to avoid cross-test interference
    output_dir_name = f"pytest_{case_name}"

    # Build a minimal args dict for ConfigDir and Launcher
    base_args = {
        "experiment_name": experiment_name,
        "output_dir_name": output_dir_name,
        "debug": True,
        "nb_workers": 1,
    }
    base_args.update(extra_args)

    # Construct config and launcher without running heavy flows
    dir_config = ConfigDir(is_analysis=False, **base_args)
    launcher = Launcher(dir_config=dir_config)

    # Validate skip flag logic
    assert launcher.skip_synth is expected_flags["skip_synth"]
    assert launcher.skip_swact is expected_flags["skip_swact"]
    assert launcher.skip_power is expected_flags["skip_power"]

    # Validate log file naming reflects the mode
    log_file = launcher.logdir / expected_log
    assert log_file.exists(), f"Expected log file missing: {log_file}"

    # Non-flowy experiments either omit the key or set it false
    assert launcher.exp_config.get("synth_with_flowy", False) is False


@pytest.mark.dependency(name="launcher", scope="session")
@pytest.mark.parametrize(
    "experiment_name",
    [
        "multiplier_2bi_4bo_permuti_allcells_notech_fullsweep_only",
    ],
)
def test_launcher_cli_only_generation(env_and_paths, experiment_name):
    """
    Smoke test: the CLI runs with --only_gener and creates at least one generated design
    without invoking heavy synthesis/simulation flows.
    """
    env = os.environ.copy()
    output_dir_name = f"cli_only_gener_{experiment_name[:16]}"

    cmd = [
        sys.executable,
        "src/genial/experiment/task_launcher.py",
    ]
    cmd += [
        "--experiment_name",
        experiment_name,
        "--output_dir_name",
        output_dir_name,
        "--only_gener",
        "--nb_new_designs",
        "1",
        "--debug",
        "--nb_workers",
        "1",
    ]

    # Ensure experiment configuration contains `top_synth_instance_names`
    config_path = (
        env_and_paths["SRC_DIR"]
        / "src/genial/templates_and_launch_scripts"
        / experiment_name
        / "experiment_configuration.json"
    )
    original_cfg_text = config_path.read_text()
    cfg = json.loads(original_cfg_text)
    if "top_synth_instance_names" not in cfg:
        cfg["top_synth_instance_names"] = [cfg.get("top_synth_instance_name", "")]
        config_path.write_text(json.dumps(cfg))

    try:
        proc = run(cmd, env=env)
        assert proc.returncode == 0, proc.stderr

        # Check that one generated design exists
        work_dir = Path(env["WORK_DIR"])  # provided by env_and_paths fixture
        root = work_dir / "output" / experiment_name / output_dir_name
        gener_dir = root / "generation_out"
        assert gener_dir.exists()
        # Expect at least one res_*/hdl/*.v or *.v.bz2
        found = False
        for res_dir in gener_dir.glob("res_*/hdl"):
            if any(p.suffix in (".v", ".bz2", ".gz", ".xz") for p in res_dir.iterdir()):
                found = True
                break
        assert found, f"No generated HDL files found under {gener_dir}"
    finally:
        config_path.write_text(original_cfg_text)
