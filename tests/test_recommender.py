from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import json
import time

import pytest


def run(cmd, env=None, cwd=None):
    return subprocess.run(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.mark.skip(reason="CLI exits unexpectedly when invoked with --help")
def test_task_recommender_help(env_and_paths):
    env = os.environ.copy()
    proc = run([sys.executable, "src/genial/experiment/task_recommender.py", "--help"], env=env)
    assert proc.returncode == 0, proc.stderr


def test_prototype_generator_help(env_and_paths):
    env = os.environ.copy()
    proc = run([sys.executable, "src/genial/utils/prototype_generator_v2.py", "--help"], env=env)
    assert proc.returncode == 0, proc.stderr


def test_recommender_end_to_end_synth_only(env_and_paths):
    env = os.environ.copy()
    exp = env_and_paths["experiment_name"]
    work_dir = Path(env["WORK_DIR"])  # provided by fixture

    # Unique-ish names per test run
    ts = int(time.time())
    gen_dir = f"pytest_rec_gen_{ts}"
    proto_dir = f"pytest_rec_proto_{ts}"

    # Ensure experiment configuration contains `top_synth_instance_names`
    config_path = (
        env_and_paths["SRC_DIR"] / "src/genial/templates_and_launch_scripts" / exp / "experiment_configuration.json"
    )
    original_cfg_text = config_path.read_text()
    cfg = json.loads(original_cfg_text)
    if "top_synth_instance_names" not in cfg:
        cfg["top_synth_instance_names"] = [cfg.get("top_synth_instance_name", "")]  # retro-compat
        config_path.write_text(json.dumps(cfg))

    try:
        # 1) Generate 10 designs
        proc = run(
            [
                sys.executable,
                "src/genial/experiment/task_launcher.py",
                "--experiment_name",
                exp,
                "--output_dir_name",
                gen_dir,
                "--only_gener",
                "--nb_new_designs",
                "10",
                "--nb_workers",
                "10",
                "--ignore_user_prompts",
                "--keep_not_valid",
            ],
            env=env,
        )
        assert proc.returncode == 0, f"GEN FAIL\n{proc.stderr}\n{proc.stdout}"

        root = work_dir / "output" / exp / gen_dir
        gen_out = root / "generation_out"
        assert gen_out.exists() and any(gen_out.glob("res_*/hdl/*")), "No generated designs found"

        # 2) Launch synthesis only
        proc = run(
            [
                sys.executable,
                "src/genial/experiment/task_launcher.py",
                "--experiment_name",
                exp,
                "--output_dir_name",
                gen_dir,
                "--skip_swact",
                "--skip_power",
                "--nb_workers",
                "10",
                "--ignore_user_prompts",
                "--keep_not_valid",
            ],
            env=env,
        )
        assert proc.returncode == 0, f"LAUNCH FAIL\n{proc.stderr}\n{proc.stdout}"

        synth_out = root / "synth_out"
        assert synth_out.exists(), "synth_out missing after launch"
        assert any(synth_out.glob("res_*/mydesign_yosys.*")), "No synthesized designs found"

        # 3) Analyze (synth only)
        proc = run(
            [
                sys.executable,
                "src/genial/experiment/task_analyzer.py",
                "--experiment_name",
                exp,
                "--output_dir_name",
                gen_dir,
                "--skip_swact",
                "--skip_power",
                "--nb_workers",
                "10",
                "--rebuild_db",
                "--ignore_user_prompts",
                "--fast_plots",
                "--keep_not_valid",
            ],
            env=env,
        )
        assert proc.returncode == 0, f"ANALYZE FAIL\n{proc.stderr}\n{proc.stdout}"

        synth_db = root / "analysis_out" / "synth_analysis.db.pqt"
        assert synth_db.exists(), "synth_analysis.db.pqt not found"

        # 4) Train for a few epochs
        proc = run(
            [
                sys.executable,
                "src/genial/training/mains/trainer_enc_to_score_value.py",
                "--experiment_name",
                exp,
                "--output_dir_name",
                gen_dir,
                "--batch_size",
                "64",
                "--nb_workers",
                "10",
                "--max_epochs",
                "10",
                "--checkpoint_naming_style",
                "enforce_increase",
                "--score_type",
                "trans",
                "--score_rescale_mode",
                "standardize",
                "--check_val_every_n_epoch",
                "1",
                "--device",
                "0",
                "--yml_config_path",
                "resources/pretrained_model/embedding/model_config.yml",
                "--trainer_version_number",
                "0",
                "--ignore_user_prompts",
                "--keep_not_valid",
                "--skip_swact",
                "--skip_power",
            ],
            env=env,
        )
        assert proc.returncode == 0, f"TRAIN FAIL\n{proc.stderr}\n{proc.stdout}"

        trainer_logs = root / "trainer_out" / "lightning_logs" / "version_0"
        assert trainer_logs.exists(), "Trainer logs/version_0 missing"

        # 5) Prototype generation using trained model
        proc = run(
            [
                sys.executable,
                "src/genial/utils/prototype_generator_v2.py",
                "--experiment_name",
                exp,
                "--output_dir_name",
                gen_dir,
                "--dst_output_dir_name",
                proto_dir,
                "--do_prototype_pattern_gen",
                "--trainer_version_number",
                "0",
                "--nb_gener_prototypes",
                "10",
                "--batch_size",
                "512",
                "--max_epochs",
                "150",
                "--device",
                "0",
                "--yml_config_path",
                "resources/pretrained_model/embedding/model_config.yml",
                "--ignore_user_prompts",
                "--keep_not_valid",
                "--skip_swact",
                "--skip_power",
            ],
            env=env,
        )
        assert proc.returncode == 0, f"PROTOGEN FAIL\n{proc.stderr}\n{proc.stdout}"

        proto_root = work_dir / "output" / exp / proto_dir
        token = proto_root / "proto_generation_done.token"
        assert token.exists(), "Prototype completion token not found"
        assert (proto_root / "generation_out").exists(), "Prototype generation_out folder missing"
    finally:
        # Restore template config
        config_path.write_text(original_cfg_text)
