from __future__ import annotations
import os
import sys
from subprocess import run
from pathlib import Path


def test_loop_full_run_v2_invocation(env_and_paths):
    # repo root (tests/<...>/this_file.py → repo/)
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    # optional: make sure the repo root is importable even if cwd shifts
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # Prefer running the module to avoid path/import weirdness
    proc = run(
        [sys.executable, "-m", "genial.loop.full_run_v2", "--help"],
        env=env,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    assert "usage" in proc.stdout.lower() or "-h, --help" in proc.stdout.lower()
