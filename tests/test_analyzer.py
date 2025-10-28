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


def test_analyzer_end_to_end_synth_only(env_and_paths):
    env = os.environ.copy()
    exp = env_and_paths["experiment_name"]
    work_dir = Path(env["WORK_DIR"])  # provided by fixture

    ts = int(time.time())
    out_dir = f"pytest_anlz_{ts}"

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

        root = work_dir / "output" / exp / out_dir
        gen_out = root / "generation_out"
        assert gen_out.exists() and any(gen_out.glob("res_*/hdl/*"))

        # 2) Launch synthesis only
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

        # 3) Analyze synth only
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

        synth_db = root / "analysis_out" / "synth_analysis.db.pqt"
        assert synth_db.exists(), "synth_analysis.db.pqt not found"
    finally:
        config_path.write_text(original_cfg_text)
