from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import json
import time


def run(cmd, env=None, cwd=None):
    return subprocess.run(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_trainer_end_to_end_small(env_and_paths):
    env = os.environ.copy()
    exp = env_and_paths["experiment_name"]
    work_dir = Path(env["WORK_DIR"])  # provided by fixture

    ts = int(time.time())
    out_dir = f"pytest_train_{ts}"

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
        # 1) Generate + synth + analyze like small pipeline
        proc = run(
            [
                sys.executable,
                "src/genial/experiment/task_launcher.py",
                "--experiment_name",
                exp,
                "--output_dir_name",
                out_dir,
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

        proc = run(
            [
                sys.executable,
                "src/genial/experiment/task_launcher.py",
                "--experiment_name",
                exp,
                "--output_dir_name",
                out_dir,
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

        proc = run(
            [
                sys.executable,
                "src/genial/experiment/task_analyzer.py",
                "--experiment_name",
                exp,
                "--output_dir_name",
                out_dir,
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

        # 2) Train for a few epochs
        proc = run(
            [
                sys.executable,
                "src/genial/training/mains/trainer_enc_to_score_value.py",
                "--experiment_name",
                exp,
                "--output_dir_name",
                out_dir,
                "--batch_size",
                "64",
                "--nb_workers",
                "10",
                "--max_epochs",
                "10",
                # "--checkpoint_naming_style",
                # "enforce_increase",
                "--skip_swact",
                "--skip_power",
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
                "--checkpoint_path",
                "resources/pretrained_model/embedding/117_0.0102_000.ckpt",
                "--trainer_version_number",
                "0",
                "--task",
                "finetune_from_ssl",
                "--ignore_user_prompts",
                "--keep_not_valid",
            ],
            env=env,
        )
        assert proc.returncode == 0, f"TRAIN FAIL\n{proc.stderr}\n{proc.stdout}"

        root = work_dir / "output" / exp / out_dir
        trainer_logs = root / "trainer_out" / "lightning_logs" / "version_0"
        assert trainer_logs.exists(), "Trainer logs/version_0 missing"

        # 3) Finetune from SSL checkpoint for a few epochs (uses provided resources)
        ssl_ckpt = Path("resources/pretrained_model/embedding/117_0.0102_000.ckpt")
        assert ssl_ckpt.exists(), "Missing SSL checkpoint for finetuning tests"
        proc = run(
            [
                sys.executable,
                "src/genial/training/mains/trainer_enc_to_score_value.py",
                "--experiment_name",
                exp,
                "--output_dir_name",
                out_dir,
                "--batch_size",
                "64",
                "--nb_workers",
                "10",
                "--max_epochs",
                "3",
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
                "--model_checkpoint_path",
                str(ssl_ckpt),
                "--trainer_task",
                "finetune_from_ssl",
                "--trainer_version_number",
                "0",
                "--ignore_user_prompts",
                "--keep_not_valid",
            ],
            env=env,
        )
        assert proc.returncode == 0, f"FINETUNE FAIL\n{proc.stderr}\n{proc.stdout}"
    finally:
        config_path.write_text(original_cfg_text)
