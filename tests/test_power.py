from __future__ import annotations

import pytest

pytestmark = pytest.mark.power
pytestmark = pytest.mark.order("last")  # run this module’s tests at the end


def test_power_only_flags(env_and_paths):
    from genial.config.config_dir import ConfigDir
    from genial.experiment.task_launcher import Launcher

    experiment_name = "multiplier_2bi_4bo_permuti_allcells_notech_fullsweep_only"
    base_args = {
        "experiment_name": experiment_name,
        "output_dir_name": "pytest_power_only",
        "debug": True,
        "nb_workers": 1,
        "skip_synth": True,
        "skip_swact": True,
    }

    dir_config = ConfigDir(is_analysis=False, **base_args)
    launcher = Launcher(dir_config=dir_config)

    assert launcher.skip_synth is True
    assert launcher.skip_swact is True
    assert launcher.skip_power is False

    # Power output directory should be prepared by the fixture; ensure path is correct
    root = env_and_paths["root_output_dir"]
    power_dir = root / "power_out"
    assert power_dir.exists()


def run(cmd, env=None):
    import subprocess

    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


@pytest.mark.dependency(name="power_minimal", scope="session")
def test_power_end_to_end_minimal(env_and_paths, tmp_path):
    """
    Minimal power pipeline:
    - Generate 2 designs with launcher (skip synth/swact/power to avoid docker).
    - Create tiny power reports for those 2 designs under power_out.
    - Run analyzer in power mode (bulk_flow_dirname=power_out, technology=asap7, rebuild_db).
    - Assert that a non-empty power database is produced.
    """
    import json
    import os
    from pathlib import Path
    import sys
    import pandas as pd

    env = os.environ.copy()

    # Use the 4bi/8bo experiment as requested
    experiment_name = "multiplier_2bi_4bo_permuti_allcells_notech_fullsweep_only"
    output_dir_name = "pytest_power_real"

    # Ensure experiment configuration contains top_synth_instance_names
    config_path = (
        env_and_paths["SRC_DIR"]
        / "src/genial/templates_and_launch_scripts"
        / experiment_name
        / "experiment_configuration.json"
    )
    original_cfg_text = config_path.read_text()
    cfg = json.loads(original_cfg_text)
    if "top_synth_instance_names" not in cfg:
        cfg["top_synth_instance_names"] = [cfg.get("top_synth_instance_name", "mydesign_top")]
        config_path.write_text(json.dumps(cfg))

    try:
        # 1) Generate exactly 2 designs (no flows)
        launch_cmd = [
            sys.executable,
            "src/genial/experiment/task_launcher.py",
            "--experiment_name",
            experiment_name,
            "--output_dir_name",
            output_dir_name,
            "--skip_synth",
            "--skip_swact",
            "--nb_new_designs",
            "2",
            "--debug",
            "--nb_workers",
            "1",
            "--ignore_user_prompts",
        ]
        proc = run(launch_cmd, env=env)
        assert proc.returncode == 0, proc.stderr

        # 2) Check the 2 generated designs
        work_dir = Path(env["WORK_DIR"])  # provided by env_and_paths
        root = work_dir / "output" / experiment_name / output_dir_name
        gener_dir = root / "generation_out"
        power_dir = root / "power_out"
        power_dir.mkdir(parents=True, exist_ok=True)

        # Find up to 2 generated designs
        gener_res = sorted(gener_dir.glob("res_*/hdl"))[:2]
        assert len(gener_res) == 2, f"Expected 2 generated designs, found {len(gener_res)} in {gener_dir}"

        # Helper to write minimal valid power artifacts
        # def write_min_power(dirpath: Path):
        #     dirpath.mkdir(parents=True, exist_ok=True)
        #     # Presence of this file marks the power directory as valid
        #     (dirpath / "synth_stat.txt").write_text("synth ok: 1234567890\n")
        #     # Minimal power report the parser understands
        #     rpt = "\n".join(
        #         [
        #             "Sequential 0.1 0.2 0.01",
        #             "Combinational 0.3 0.4 0.02",
        #             "Clock 0.05 0.06 0.005",
        #         ]
        #     )
        #     (dirpath / "post_synth_power.rpt").write_text(rpt)

        # for hdl_dir in gener_res:
        #     res_num = hdl_dir.parent.name  # res_XXXX
        #     pdir = power_dir / res_num
        #     write_min_power(pdir)

        # 3) Run analyzer in power mode against power_out
        analyze_cmd = [
            sys.executable,
            "src/genial/experiment/task_analyzer.py",
            "--experiment_name",
            experiment_name,
            "--output_dir_name",
            output_dir_name,
            "--nb_workers",
            "1",
            "--skip_swact",
            "--bulk_flow_dirname",
            "power_out",
            "--rebuild_db",
            "--technology",
            "asap7",
            "--ignore_user_prompts",
            "--skip_plots",
        ]
        proc = run(analyze_cmd, env=env)
        assert proc.returncode == 0, proc.stderr

        # 4) Validate the produced power database
        db_path = root / "analysis_out" / "power_analysis.db.pqt"
        assert db_path.exists(), f"Missing power database at {db_path}"
        df = pd.read_parquet(db_path)
        assert not df.empty, "Power database is empty"
        # Columns produced by parse_power_n_delay_reports
        assert "p_comb_dynamic" in df.columns
        assert "p_seq_dynamic" in df.columns
        assert (df[["p_comb_dynamic", "p_seq_dynamic"]].fillna(0).sum(axis=1) >= 0).all()
    finally:
        # Restore original experiment configuration
        config_path.write_text(original_cfg_text)
