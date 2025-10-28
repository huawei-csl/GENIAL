from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import json
import time

import pytest

# Delay heavy imports until after env is prepared by fixtures
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from genial.config.config_dir import ConfigDir  # noqa: F401
    from genial.experiment.task_launcher import Launcher  # noqa: F401


def run(cmd, env=None):
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_launcher_parse_args_basic():
    """Test that Launcher.parse_args works with minimal arguments."""
    from genial.experiment.task_launcher import Launcher

    original_argv = sys.argv.copy()

    try:
        sys.argv = [
            "task_launcher.py",
            "--experiment_name",
            "test_experiment",
            "--output_dir_name",
            "test_output",
            "--only_gener",
        ]

        args_dict = Launcher.parse_args()

        assert args_dict is not None
        assert isinstance(args_dict, dict)
        assert args_dict["experiment_name"] == "test_experiment"
        assert args_dict["output_dir_name"] == "test_output"
        assert args_dict["only_gener"] is True

    finally:
        sys.argv = original_argv


def test_launcher_parse_args_synth_only():
    """Test that synth_only flag sets correct skip flags."""
    from genial.experiment.task_launcher import Launcher

    original_argv = sys.argv.copy()

    try:
        sys.argv = [
            "task_launcher.py",
            "--experiment_name",
            "test",
            "--output_dir_name",
            "test",
            "--synth_only",
        ]

        args_dict = Launcher.parse_args()

        assert args_dict["synth_only"] is True

    finally:
        sys.argv = original_argv


def test_launcher_parse_args_restart():
    """Test that restart flag sets correct skip flags."""
    from genial.experiment.task_launcher import Launcher

    original_argv = sys.argv.copy()

    try:
        sys.argv = [
            "task_launcher.py",
            "--experiment_name",
            "test",
            "--output_dir_name",
            "test",
            "--restart",
        ]

        args_dict = Launcher.parse_args()

        # Restart should enable all skip flags
        assert args_dict["restart"] is True
        assert args_dict["do_gener"] is False
        assert args_dict["skip_already_synth"] is True
        assert args_dict["skip_already_swact"] is True
        assert args_dict["skip_already_power"] is True

    finally:
        sys.argv = original_argv


def test_launcher_skip_flags_logic(env_and_paths):
    """Test that skip flags are correctly set based on arguments."""
    from genial.experiment.task_launcher import Launcher
    from genial.config.config_dir import ConfigDir

    exp = env_and_paths["experiment_name"]
    out_dir = env_and_paths["output_dir_name"]

    original_argv = sys.argv.copy()

    try:
        # Test 1: synth_only should skip swact and power
        sys.argv = [
            "task_launcher.py",
            "--experiment_name",
            exp,
            "--output_dir_name",
            out_dir,
            "--synth_only",
        ]

        dir_config = ConfigDir(is_analysis=False)
        launcher = Launcher(dir_config=dir_config)

        assert launcher.skip_synth is False
        assert launcher.skip_swact is True, "synth_only should skip swact"
        assert launcher.skip_power is True, "synth_only should skip power"

        # Test 2: skip_synth should only skip synth
        sys.argv = [
            "task_launcher.py",
            "--experiment_name",
            exp,
            "--output_dir_name",
            out_dir,
            "--skip_synth",
        ]

        dir_config = ConfigDir(is_analysis=False)
        launcher = Launcher(dir_config=dir_config)

        assert launcher.skip_synth is True
        assert launcher.skip_swact is False
        assert launcher.skip_power is False

    finally:
        sys.argv = original_argv


def test_launcher_log_files_creation(env_and_paths):
    """Test that log files are created correctly."""
    from genial.experiment.task_launcher import Launcher
    from genial.config.config_dir import ConfigDir

    exp = env_and_paths["experiment_name"]
    out_dir = env_and_paths["output_dir_name"]

    original_argv = sys.argv.copy()

    try:
        # Test synth_only creates correct log file
        sys.argv = [
            "task_launcher.py",
            "--experiment_name",
            exp,
            "--output_dir_name",
            out_dir,
            "--synth_only",
        ]

        dir_config = ConfigDir(is_analysis=False)
        launcher = Launcher(dir_config=dir_config)

        assert launcher.logdir.exists()
        assert (launcher.logdir / "launcher_synth.log").exists()

        # Test full mode creates correct log file
        sys.argv = [
            "task_launcher.py",
            "--experiment_name",
            exp,
            "--output_dir_name",
            out_dir + "_full",
        ]

        dir_config = ConfigDir(is_analysis=False)
        launcher = Launcher(dir_config=dir_config)

        assert launcher.logdir.exists()
        assert (launcher.logdir / "launcher_full.log").exists()

    finally:
        sys.argv = original_argv


def test_launcher_instantiation(env_and_paths):
    """Test that Launcher can be instantiated with proper config."""
    from genial.experiment.task_launcher import Launcher
    from genial.config.config_dir import ConfigDir

    exp = env_and_paths["experiment_name"]
    out_dir = env_and_paths["output_dir_name"]

    original_argv = sys.argv.copy()

    try:
        sys.argv = [
            "task_launcher.py",
            "--experiment_name",
            exp,
            "--output_dir_name",
            out_dir,
            "--synth_only",
        ]

        dir_config = ConfigDir(is_analysis=False)
        launcher = Launcher(dir_config=dir_config)

        # Basic assertions
        assert launcher is not None
        assert launcher.dir_config is not None
        assert hasattr(launcher, "skip_synth")
        assert hasattr(launcher, "skip_swact")
        assert hasattr(launcher, "skip_power")
        assert hasattr(launcher, "exp_config")
        assert launcher.experiment_name == exp

    finally:
        sys.argv = original_argv


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


@pytest.mark.requires_docker
@pytest.mark.parametrize(
    "experiment_name",
    [
        "multiplier_2bi_4bo_permuti_allcells_notech_fullsweep_only",
    ],
)
def test_launcher_cli_small_synth_only(env_and_paths, experiment_name):
    """
    Small integration test: generate 2 designs and synthesize them with synth_only mode.
    This tests the full generation + synthesis pipeline without swact/power.
    """
    env = os.environ.copy()
    ts = int(time.time())
    output_dir_name = f"pytest_synth_{ts}"

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
        # Step 1: Generate 2 designs
        gen_cmd = [
            sys.executable,
            "src/genial/experiment/task_launcher.py",
            "--experiment_name",
            experiment_name,
            "--output_dir_name",
            output_dir_name,
            "--only_gener",
            "--nb_new_designs",
            "2",
            "--nb_workers",
            "2",
            "--ignore_user_prompts",
            "--keep_not_valid",
        ]

        proc = run(gen_cmd, env=env)
        assert proc.returncode == 0, f"Generation failed:\n{proc.stderr}\n{proc.stdout}"

        work_dir = Path(env["WORK_DIR"])
        root = work_dir / "output" / experiment_name / output_dir_name
        gener_dir = root / "generation_out"

        # Check generation succeeded
        assert gener_dir.exists(), f"Generation directory not found at {gener_dir}"
        gen_designs = list(gener_dir.glob("res_*/hdl/*mydesign_comb.v*"))
        assert len(gen_designs) >= 2, f"Expected at least 2 generated designs, found {len(gen_designs)}"
        print(f"✓ Generated {len(gen_designs)} designs")

        # Step 2: Synthesize with synth_only mode
        synth_cmd = [
            sys.executable,
            "src/genial/experiment/task_launcher.py",
            "--experiment_name",
            experiment_name,
            "--output_dir_name",
            output_dir_name,
            "--synth_only",
            "--nb_workers",
            "2",
            "--ignore_user_prompts",
            "--keep_not_valid",
        ]

        proc = run(synth_cmd, env=env)

        # Print output for debugging
        if proc.returncode != 0:
            print(f"SYNTHESIS STDERR:\n{proc.stderr}")
            print(f"SYNTHESIS STDOUT:\n{proc.stdout}")

        assert proc.returncode == 0, f"Synthesis failed:\n{proc.stderr}\n{proc.stdout}"

        # Check synthesis outputs
        synth_dir = root / "synth_out"
        assert synth_dir.exists(), f"Synthesis directory not found at {synth_dir}"

        synth_designs = list(synth_dir.glob("res_*"))
        print(f"\nSynthesized design directories: {[d.name for d in synth_designs]}")

        # Check that at least some designs have synthesis outputs
        found_synth_files = []
        for design_dir in synth_designs:
            synth_files = list(design_dir.glob("mydesign_yosys.v*"))
            if synth_files:
                found_synth_files.extend(synth_files)

        print(f"Found synthesis files: {[f.parent.name + '/' + f.name for f in found_synth_files]}")

        assert len(found_synth_files) > 0, (
            f"No synthesis output files found. Synth directories: {[d.name for d in synth_designs]}"
        )

        print(f"✓ Successfully synthesized {len(found_synth_files)} designs")

    finally:
        config_path.write_text(original_cfg_text)
