import os
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
os.environ.setdefault("SRC_DIR", str(REPO_ROOT))

from swact.netlist import get_fanout_wires_n_depth  # noqa: E402


def test_netlist_graph_building(tmp_path: Path) -> None:
    # Prepare minimal SRC_DIR so GatesConfig loads
    os.environ["SRC_DIR"] = str(tmp_path)
    (tmp_path / "resources/libraries/asap7").mkdir(parents=True)
    (tmp_path / "resources/libraries/asap7/transistor_count.json").write_text("{}")
    (tmp_path / "resources/libraries/asap7/cell_area.json").write_text("{}")

    netlist = tmp_path / "design.v"
    netlist.write_text(
        textwrap.dedent(
            """module top(input a,b,output y);
wire n1;
$_AND_ _0_ (
    .A(a),
    .B(b),
    .Y(n1)
);
$_NOT_ _1_ (
    .A(n1),
    .Y(y)
);
endmodule
"""
        )
    )
    wires = ["a", "b", "n1", "y"]
    fanout, wires_graph, cells_graph = get_fanout_wires_n_depth(
        synthed_design_path=netlist,
        wire_list=wires,
        technology_name="notech_yosys",
    )
    assert wires_graph["n1"] == ["a", "b"]
    assert wires_graph["y"] == ["n1"]
    assert fanout["a"] == 1
    assert fanout["n1"] == 1
    assert fanout["y"] == 0
    assert cells_graph  # not empty
