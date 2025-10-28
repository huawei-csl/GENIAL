from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def run(cmd, env=None):
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_check_status_counts(env_and_paths):
    env = os.environ.copy()
    exp = env_and_paths["experiment_name"]
    out = env_and_paths["output_dir_name"]
    # Create a few dummy design folders to be counted
    gen = env_and_paths["root_output_dir"] / "generation_out"
    syn = env_and_paths["root_output_dir"] / "synth_out"
    tst = env_and_paths["root_output_dir"] / "test_out"
    for d in [gen / "res_0001", syn / "res_0001", tst / "res_0001", gen / "res_0002"]:
        d.mkdir(parents=True, exist_ok=True)

    proc = run(
        [
            sys.executable,
            "src/genial/utils/check_status.py",
            "--experiment_name",
            exp,
            "--output_dir_name",
            out,
        ],
        env=env,
    )

    assert proc.returncode == 0, proc.stderr


def test_compress_decompress_file_roundtrip(tmp_path: Path, repo_root: Path):
    # Import module directly without requiring console script installation
    sys.path.insert(0, str(repo_root / "src"))
    from swact.file_compression_handler import FileCompressionHandler  # type: ignore

    # Create a small .v file
    src = tmp_path / "example.v"
    payload = "module m; endmodule\n"
    src.write_text(payload)

    # Compress
    compressed = FileCompressionHandler.compress_file(src)
    assert compressed.exists()
    # Original should be removed after compression
    assert not src.exists()

    # Decompress (using original logical path)
    decompressed = FileCompressionHandler.decompress_file(src)
    assert decompressed.exists()
    assert decompressed.read_text() == payload


@pytest.mark.skip(reason="Requires docker and cleans containers; skip in CI")
def test_clean_docker_script_exists(repo_root: Path):
    script = repo_root / "scripts/clean_docker.sh"
    assert script.exists() and os.access(script, os.X_OK)


@pytest.mark.skip(reason="Requires pigz/pv and large I/O; skip in CI")
def test_compress_output_folder_script_exists(repo_root: Path):
    script = repo_root / "scripts/compress_output_folder.sh"
    assert script.exists() and os.access(script, os.X_OK)
