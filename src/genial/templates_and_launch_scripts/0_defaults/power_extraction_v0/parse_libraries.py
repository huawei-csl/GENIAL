import gzip
import json
from pathlib import Path

root_dirpath = Path("/prog/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM")

is_cell = False
cell_name = None
filname = None
cell_dict = dict()
for filepath in root_dirpath.iterdir():
    if filepath.suffix == ".gz":
        print(f"Parsing {filepath.name} ..")
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if "cell (" in line:
                    if is_cell and cell_name is not None and filname is not None:
                        cell_dict[cell_name] = None
                        print(f"Cell {cell_name} was from file {filname} wrongly defined")
                    is_cell = True
                    cell_name = line.split("(")[1].split(")")[0]
                    filname = filepath.name

                if is_cell:
                    if "area" in line:
                        area = float(line.split(": ")[1].split(";")[0])
                        cell_dict[cell_name] = area
                        is_cell = False

with open("cell_area.json", "w") as f:
    json.dump(cell_dict, f, indent=4)
