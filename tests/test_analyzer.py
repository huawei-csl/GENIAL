from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import json
import time
import pandas as pd


def run(cmd, env=None, cwd=None):
    return subprocess.run(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_analyzer_parse_args_basic():
    """Test that Analyzer.parse_args works with minimal arguments."""
    from genial.experiment.task_analyzer import Analyzer

    # Save original sys.argv
    original_argv = sys.argv.copy()

    try:
        # Set minimal required arguments for parsing
        sys.argv = [
            "task_analyzer.py",
            "--experiment_name",
            "test_experiment",
            "--output_dir_name",
            "test_output",
            "--rebuild_db",
            "--ignore_user_prompts",
        ]

        args_dict = Analyzer.parse_args()

        # Basic assertions
        assert args_dict is not None
        assert isinstance(args_dict, dict)
        assert args_dict["experiment_name"] == "test_experiment"
        assert args_dict["output_dir_name"] == "test_output"
        assert args_dict["rebuild_db"] is True
        assert args_dict["ignore_user_prompts"] is True

    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def test_analyzer_load_database_empty(tmp_path):
    """Test that _load_database handles non-existent files correctly."""
    from genial.experiment.task_analyzer import Analyzer

    non_existent_path = tmp_path / "non_existent.db.pqt"

    # With force=True, should return empty DataFrame
    df = Analyzer._load_database(non_existent_path, force=True)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_analyzer_load_database_valid(tmp_path):
    """Test that _load_database can load a valid parquet file."""
    from genial.experiment.task_analyzer import Analyzer

    # Create a simple test database
    test_data = pd.DataFrame({"design_number": ["res_0001", "res_0002"], "value": [100, 200]})

    db_path = tmp_path / "test.db.pqt"
    test_data.to_parquet(db_path, index=False)

    # Load it back
    df = Analyzer._load_database(db_path, force=True)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 2
    assert "design_number" in df.columns
    assert list(df["design_number"]) == ["res_0001", "res_0002"]


def test_analyzer_instantiation(env_and_paths):
    """Test that we can instantiate an Analyzer object."""
    from genial.experiment.task_analyzer import Analyzer
    from genial.config.config_dir import ConfigDir

    os.environ.copy()
    exp = env_and_paths["experiment_name"]
    out_dir = env_and_paths["output_dir_name"]

    # Save original sys.argv
    original_argv = sys.argv.copy()

    try:
        # Set up arguments for ConfigDir
        sys.argv = [
            "task_analyzer.py",
            "--experiment_name",
            exp,
            "--output_dir_name",
            out_dir,
            "--rebuild_db",
            "--ignore_user_prompts",
            "--skip_swact",
            "--skip_power",
        ]

        # Create ConfigDir
        dir_config = ConfigDir(is_analysis=True)

        # Create Analyzer - this tests basic initialization
        analyzer = Analyzer(
            dir_config=dir_config,
            skip_log_init=True,  # Skip log initialization for testing
        )

        # Basic assertions
        assert analyzer is not None
        assert analyzer.dir_config is not None
        assert analyzer.analysis_out_dir.exists()
        assert hasattr(analyzer, "synth_df")
        assert hasattr(analyzer, "gener_df")

    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def test_analyzer_database_initialization(env_and_paths):
    """Test that Analyzer initializes its databases correctly."""
    from genial.experiment.task_analyzer import Analyzer
    from genial.config.config_dir import ConfigDir

    os.environ.copy()
    exp = env_and_paths["experiment_name"]
    out_dir = env_and_paths["output_dir_name"]

    original_argv = sys.argv.copy()

    try:
        sys.argv = [
            "task_analyzer.py",
            "--experiment_name",
            exp,
            "--output_dir_name",
            out_dir,
            "--rebuild_db",
            "--ignore_user_prompts",
            "--skip_swact",
            "--skip_power",
        ]

        dir_config = ConfigDir(is_analysis=True)
        analyzer = Analyzer(dir_config=dir_config, skip_log_init=True)

        # Check that database paths are set correctly
        assert hasattr(analyzer, "output_synth_db_path")
        assert hasattr(analyzer, "output_gener_db_path")
        assert analyzer.output_synth_db_path.name == "synth_analysis.db.pqt"
        assert analyzer.output_gener_db_path.name == "gener_analysis.db.pqt"

        # Check that DataFrames are initialized (should be empty with rebuild_db)
        assert isinstance(analyzer.synth_df, pd.DataFrame)
        assert isinstance(analyzer.gener_df, pd.DataFrame)

    finally:
        sys.argv = original_argv


def test_analyzer_with_mock_synth_data(env_and_paths):
    """Test that Analyzer can process synthesis outputs when they exist."""
    from genial.experiment.task_analyzer import Analyzer
    from genial.config.config_dir import ConfigDir

    exp = env_and_paths["experiment_name"]
    out_dir = "pytest_mock_synth"
    root = env_and_paths["WORK_DIR"] / "output" / exp / out_dir

    # Create mock synthesis output structure
    synth_out = root / "synth_out"
    gen_out = root / "generation_out"
    analysis_out = root / "analysis_out"

    for directory in [synth_out, gen_out, analysis_out]:
        directory.mkdir(parents=True, exist_ok=True)

    # Create a mock design directory with minimal synthesis output
    design_dir = synth_out / "res_0001"
    design_dir.mkdir(parents=True, exist_ok=True)

    # Create a minimal synthesis report (this is what the analyzer looks for)
    report_file = design_dir / "synthesis_report.txt"
    report_file.write_text("""
Area report:
Total cells: 42
""")

    # Create matching generation output
    gen_design_dir = gen_out / "res_0001"
    (gen_design_dir / "hdl").mkdir(parents=True, exist_ok=True)
    (gen_design_dir / "encoding.json").write_text('{"in_enc_dict": {}, "out_enc_dict": {}}')

    original_argv = sys.argv.copy()

    try:
        sys.argv = [
            "task_analyzer.py",
            "--experiment_name",
            exp,
            "--output_dir_name",
            out_dir,
            "--rebuild_db",
            "--ignore_user_prompts",
            "--skip_swact",
            "--skip_power",
            "--skip_cmplx",
        ]

        dir_config = ConfigDir(is_analysis=True)
        analyzer = Analyzer(dir_config=dir_config, skip_log_init=True)

        # Check that analyzer found the mock synth output
        assert analyzer.analysis_out_dir.exists()

        # Try to get encodings (should work with mock data)
        analyzer.get_all_design_encodings()
        assert hasattr(analyzer, "gener_df")

    finally:
        sys.argv = original_argv


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
        assert gen_out.exists() and any(gen_out.glob("res_*/hdl/*")), (
            f"Generation output not found. Contents: {list(root.iterdir()) if root.exists() else 'root does not exist'}"
        )

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

        # Check synthesis outputs exist
        synth_out = root / "synth_out"
        assert synth_out.exists(), f"Synthesis output directory not found at {synth_out}"
        synth_designs = list(synth_out.glob("res_*"))
        assert len(synth_designs) > 0, f"No synthesized designs found in {synth_out}"

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

        # Print analyzer output for debugging
        if proc.returncode != 0:
            print(f"ANALYZER STDERR:\n{proc.stderr}")
            print(f"ANALYZER STDOUT:\n{proc.stdout}")
        assert proc.returncode == 0, f"ANALYZE FAIL\n{proc.stderr}\n{proc.stdout}"

        # Check what files were created
        analysis_out = root / "analysis_out"
        assert analysis_out.exists(), f"Analysis output directory not found at {analysis_out}"

        created_files = list(analysis_out.glob("*"))
        print(f"\nFiles created in analysis_out: {[f.name for f in created_files]}")

        # Print analyzer stdout to see what happened
        print(f"\nAnalyzer output:\n{proc.stdout[-2000:]}")  # Last 2000 chars

        synth_db = root / "analysis_out" / "synth_analysis.db.pqt"
        assert synth_db.exists(), f"synth_analysis.db.pqt not found. Created files: {[f.name for f in created_files]}"
    finally:
        config_path.write_text(original_cfg_text)
