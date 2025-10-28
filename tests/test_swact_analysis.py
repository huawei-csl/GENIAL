import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
os.environ.setdefault("SRC_DIR", str(REPO_ROOT))

from swact import extract_swact_from_tb_results  # noqa: E402
from swact.legacy import extract_swact_from_tb_results_legacy  # noqa: E402


def test_extract_swact_from_tb_results():
    df = pd.DataFrame({"a": [0, 1, 1, 0], "b": [0, 0, 1, 1]})
    design = {"wire_fanout_dict": {"a": 1, "b": 2}}

    res = extract_swact_from_tb_results(df, design)
    legacy = extract_swact_from_tb_results_legacy(df, design)
    assert res == legacy
    assert res["swact_count"] == 0.75
    assert res["swact_count_weighted"] == 1.0
    assert res["n_cycles"] == 4
    assert res["n_wires"] == 2
