from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
os.environ.setdefault("SRC_DIR", str(REPO_ROOT))

from genial.experiment.file_parsers import (  # noqa: E402
    _clean_output_pin_names,
    cleanup_wire_match,
    get_wire_list,
    parse_power_line,
)


def test_clean_output_pin_names():
    text = "\\ out pin.name"
    # the function should drop backslashes and spaces, replace dots with underscores
    assert _clean_output_pin_names(text) == "outpin_name"


def test_cleanup_wire_match_array():
    assert cleanup_wire_match("[3:0] data") == [
        "data[0]",
        "data[1]",
        "data[2]",
        "data[3]",
    ]


def test_cleanup_wire_match_list():
    assert cleanup_wire_match("a, b ,c") == ["a", "b", "c"]


def test_parse_power_line():
    line = "Group 1.5 2.5 3.5"
    expected = {
        "p_grp_dynamic_internal": 1.5,
        "p_grp_dynamic_switching": 2.5,
        "p_grp_dynamic": 4.0,
        "p_grp_static": 3.5,
    }
    assert parse_power_line(line, "grp") == expected


def test_get_wire_list(tmp_path):
    verilog = (
        "module top;\ninput a;\ninput b;\noutput y;\nwire n1, n2;\nassign n1 = a & b;\nassign y = n1;\nendmodule\n"
    )
    design_path = tmp_path / "design.v"
    design_path.write_text(verilog)

    wires, io = get_wire_list(design_path, remove_dangling_wires=False)

    assert set(wires) == {"n1", "n2"}
    assert io == [["a", "b"], ["y"]]
