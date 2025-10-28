from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import json
import pandas as pd
import pytest

pytestmark = pytest.mark.flowy
pytestmark = pytest.mark.order("last")  # run this module’s tests at the end


def run(cmd, env=None):
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_flowy_launcher_flags(env_and_paths):
    from genial.config.config_dir import ConfigDir
    from genial.experiment.task_launcher import Launcher

    experiment_name = "multiplier_4bi_8bo_permuti_flowy_debug"
    base_args = {
        "experiment_name": experiment_name,
        "output_dir_name": "pytest_flowy_synth_only",
        "debug": True,
        "nb_workers": 1,
        "synth_only": True,
    }

    dir_config = ConfigDir(is_analysis=False, **base_args)
    launcher = Launcher(dir_config=dir_config)

    assert launcher.skip_synth is False
    assert launcher.skip_swact is True
    assert launcher.skip_power is True
    assert launcher.exp_config.get("synth_with_flowy", False) is True


# def test_flowy_cli_only_generation(env_and_paths):
#     env = os.environ.copy()
#     experiment_name = "multiplier_4bi_8bo_permuti_flowy_debug"
#     output_dir_name = "cli_only_gener_flowy"

#     cmd = [
#         sys.executable,
#         "src/genial/experiment/task_launcher.py",
#         "--experiment_name",
#         experiment_name,
#         "--output_dir_name",
#         output_dir_name,
#         "--nb_new_designs",
#         "1",
#         "--debug",
#         "--nb_workers",
#         "1",
#         "--bulk_flow_dirname",
#         "synth_out",
#         "--skip_swact",
#         "--skip_power",
#         "--ignore_user_prompts",
#         "--keep_not_valid",
#     ]

#     proc = run(cmd, env=env)
#     assert proc.returncode == 0, proc.stderr

#     # Check that one generated design exists
#     work_dir = Path(env["WORK_DIR"])  # provided by env_and_paths fixture
#     root = work_dir / "output" / experiment_name / output_dir_name
#     gener_dir = root / "generation_out"
#     assert gener_dir.exists()
#     found = False
#     for res_dir in gener_dir.glob("res_*/hdl"):
#         if any(p.suffix in (".v", ".bz2", ".gz", ".xz") for p in res_dir.iterdir()):
#             found = True
#             break
#     assert found, f"No generated HDL files found under {gener_dir}"


@pytest.mark.dependency(name="flowy_full_minimal", scope="session")
def test_flowy_end_to_end_minimal(env_and_paths, tmp_path):
    """
    Minimal flowy 'full run' (generation + synthetic synthesis artifacts + analyze on synth_out):

    - Ensure experiment config has top_synth_instance_names (needed by analyzer).
    - Generate 2 designs with the launcher (skip swact/power; we also avoid real Docker synth).
    - Create tiny 'synth_out' artifacts for those designs that the analyzer can parse.
    - Run analyzer in synth mode (bulk_flow_dirname=synth_out, rebuild_db, skip plots).
    - Assert that a non-empty parquet database is produced.
    """
    env = os.environ.copy()

    experiment_name = "multiplier_4bi_8bo_permuti_flowy_debug"
    output_dir_name = "pytest_flowy_full_minimal"

    work_dir = Path(env["WORK_DIR"])
    root = work_dir / "output" / experiment_name / output_dir_name

    # 0) Make sure the output directory is clean
    if root.exists():
        for child in root.iterdir():
            if child.is_file():
                child.unlink()
            else:
                import shutil

                shutil.rmtree(child)
        root.rmdir()
    assert not root.exists()

    # 1) Ensure experiment configuration has the expected synth instance names
    cfg_path = (
        env_and_paths["SRC_DIR"]
        / "src/genial/templates_and_launch_scripts"
        / experiment_name
        / "experiment_configuration.json"
    )
    original_cfg_text = cfg_path.read_text()
    cfg = json.loads(original_cfg_text)
    changed_cfg = False
    if "top_synth_instance_names" not in cfg:
        cfg["top_synth_instance_names"] = [cfg.get("top_synth_instance_name", "mydesign_top")]
        changed_cfg = True
    if changed_cfg:
        cfg_path.write_text(json.dumps(cfg))

    try:
        # 2) Generate exactly 1 designs (no real flows)
        launch_cmd = [
            sys.executable,
            "src/genial/experiment/task_launcher.py",
            "--experiment_name",
            experiment_name,
            "--output_dir_name",
            output_dir_name,
            "--nb_new_designs",
            "1",
            "--nb_workers",
            "1",
            "--skip_synth",
            "--skip_swact",
            "--skip_power",
            "--ignore_user_prompts",
            "--keep_not_valid",
        ]
        proc = run(launch_cmd, env=env)
        assert proc.returncode == 0, proc.stderr

        # 3) Check the 2 generated designs

        gener_dir = root / "generation_out"
        synth_dir = root / "synth_out"
        synth_dir.mkdir(parents=True, exist_ok=True)

        gener_res = sorted(gener_dir.glob("res_*/hdl"))[:2]
        assert len(gener_res) == 1, f"Expected 1 generated designs, found {len(gener_res)} in {gener_dir}"

        cmd = [
            sys.executable,
            "src/genial/experiment/task_launcher.py",
            "--experiment_name",
            experiment_name,
            "--output_dir_name",
            output_dir_name,
            "--nb_workers",
            "1",
            "--bulk_flow_dirname",
            "synth_out",
            "--skip_swact",
            "--skip_power",
            "--ignore_user_prompts",
            "--keep_not_valid",
        ]
        proc = run(cmd, env=env)
        assert proc.returncode == 0, proc.stderr

        # def write_min_synth(dirpath: Path):
        #     """
        #     Create the smallest set of files the synth analyzer can consume.
        #     Adjust keys if your parser expects different names.
        #     """
        #     dirpath.mkdir(parents=True, exist_ok=True)
        #     # Marker/stat file commonly parsed in flows
        #     (dirpath / "synth_stat.txt").write_text(
        #         "\n".join(
        #             [
        #                 "complexity_post_opt: 42",
        #                 "nb_transistors: 1234",
        #                 "nb_cells: 321",
        #                 "synth_ok: 1",
        #             ]
        #         )
        #         + "\n"
        #     )
        #     # Optional placeholder netlist/report files (harmless if ignored)
        #     (dirpath / "post_synth.log").write_text("synthesis completed\n")
        #     (dirpath / "post_synth_netlist.v").write_text("// dummy netlist\nmodule top(); endmodule\n")

        # for hdl_dir in gener_res:
        #     res_num = hdl_dir.parent.name  # res_XXXX
        #     write_min_synth(synth_dir / res_num)

        # 4) Run analyzer in synth mode against synth_out
        analyze_cmd = [
            sys.executable,
            "src/genial/experiment/task_analyzer.py",
            "--experiment_name",
            experiment_name,
            "--output_dir_name",
            output_dir_name,
            "--nb_workers",
            "1",
            "--bulk_flow_dirname",
            "synth_out",
            "--rebuild_db",
            "--ignore_user_prompts",
            "--skip_power",
            "--skip_plots",
        ]
        proc = run(analyze_cmd, env=env)
        assert proc.returncode == 0, proc.stderr

        # 5) Validate produced database (be tolerant on filename; assert non-empty)
        analysis_dir = root / "analysis_out"
        assert analysis_dir.exists(), f"Missing analysis_out at {analysis_dir}"
        # Prefer a synth-specific DB name if present, otherwise any .pqt
        candidates = []
        preferred = analysis_dir / "synth_analysis.db.pqt"
        if preferred.exists():
            candidates = [preferred]
        else:
            candidates = sorted(analysis_dir.glob("*.pqt"))
        assert candidates, f"No parquet database produced under {analysis_dir}"

        df = pd.read_parquet(candidates[0])
        assert not df.empty, f"Parquet DB {candidates[0].name} is empty"

        # Basic sanity: at least one of these columns is present if parser extracted stats
        maybe_cols = {"complexity_post_opt", "nb_transistors", "nb_cells"}
        assert maybe_cols.intersection(df.columns), (
            f"Expected at least one of {maybe_cols} in columns, found {list(df.columns)}"
        )
    finally:
        # Restore original experiment configuration if we edited it
        if changed_cfg:
            cfg_path.write_text(original_cfg_text)
