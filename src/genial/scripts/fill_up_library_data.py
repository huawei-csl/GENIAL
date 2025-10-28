import os
from pathlib import Path
import json
import math

if __name__ == "__main__":
    libraries_dirpath = Path(os.environ.get("SRC_DIR")) / "resources/libraries"

    for lib_dirpath in libraries_dirpath.iterdir():
        try:
            with open(lib_dirpath / "cell_area.json", "r") as f:
                cell_area_dict = json.load(f)
        except FileNotFoundError:
            cell_area_dict = None

        try:
            with open(lib_dirpath / "nand_gate_cell_name.txt", "r") as f:
                nand_gate_cell_name = f.readlines()[0]

        except FileNotFoundError:
            nand_gate_cell_name = None

        if cell_area_dict is not None and nand_gate_cell_name is not None:
            nand_gate_area = cell_area_dict[nand_gate_cell_name]

            transistor_count_dict = {
                cell_name: (math.ceil(cell_area / nand_gate_area * 4))
                for cell_name, cell_area in cell_area_dict.items()
            }

            with open(lib_dirpath / "transistor_count.json", "w") as f:
                json.dump(transistor_count_dict, f, indent=4)
