# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

import os
import random
import re
import shutil
from pathlib import Path
from collections import Counter
from time import time, sleep
from typing import Any
from types import SimpleNamespace
import json
from pyarrow.lib import ArrowInvalid


import pandas as pd
from swact.gates_configuration import GatesConfig
from genial.globals import global_vars
from genial.utils import utils
from genial.utils.utils import extract_int_string_from_string, process_pool_helper
from loguru import logger
from tqdm import tqdm

from swact.netlist import (
    format_synthed_design,
)
from swact.netlist import (
    get_wire_list as _swact_get_wire_list,
    cleanup_wire_match as _swact_cleanup_wire_match,
    _clean_output_pin_names as _swact_clean_output_pin_names,
)
from swact.file_compression_handler import FileCompressionHandler


# def format_synthed_design(synthed_dirpath: Path, technology_name:str) -> None:
#     """
#     Prepare the design file for test and analysis.
#     """

#     replace_notech_primitives(synthed_dirpath, technology_name)

#     (synthed_dirpath / "cleaned_design.token").touch()


def parse_power_line(line: str, nickname: str):
    elements = line.strip().split()
    internal_dynamic_power = float(elements[1])
    switching_dynamic_power = float(elements[2])
    dynamic_power = internal_dynamic_power + switching_dynamic_power
    static_power = float(elements[3])
    return {
        f"p_{nickname}_dynamic_internal": internal_dynamic_power,
        f"p_{nickname}_dynamic_switching": switching_dynamic_power,
        f"p_{nickname}_dynamic": dynamic_power,
        f"p_{nickname}_static": static_power,
    }


# ---------------------------------------------------------------------------
# Lightweight re-exports for backwards compatibility with older tests
# ---------------------------------------------------------------------------
def _clean_output_pin_names(text: str) -> str:
    """Compatibility wrapper around swact.netlist._clean_output_pin_names.

    Older code imported this helper from file_parsers; keep that path working.
    """
    return _swact_clean_output_pin_names(text)


def cleanup_wire_match(match: str):
    """Compatibility wrapper around swact.netlist.cleanup_wire_match."""
    return _swact_cleanup_wire_match(match)


def get_wire_list(synthed_design_path: Path, remove_dangling_wires: bool, clean_output_pin_names: bool = True):
    """Compatibility wrapper around swact.netlist.get_wire_list.

    Mirrors the original signature used by tests.
    """
    return _swact_get_wire_list(
        synthed_design_path=synthed_design_path,
        remove_dangling_wires=remove_dangling_wires,
        clean_output_pin_names=clean_output_pin_names,
    )


group_to_nickname = {
    "Sequential": "seq",
    "Combinational": "comb",
    "Clock": "clk",
}


def parse_power_n_delay_reports(power_design_dir_path: Path) -> tuple[int, dict[str, int]]:
    """[Worker function] This function parses the power report and extract the various power information"""

    design_number = extract_design_number_from_path(power_design_dir_path)
    power_rpt_dict = {"design_number": design_number}

    try:
        # Parse Power Report
        with open(power_design_dir_path / "post_synth_power.rpt", "r") as f:
            power_report = f.read()

        for line in power_report.split("\n"):
            if line.startswith("Sequential"):
                power_rpt_dict.update(parse_power_line(line, "seq"))

            if line.startswith("Combinational"):
                power_rpt_dict.update(parse_power_line(line, "comb"))

            if line.startswith("Clock"):
                power_rpt_dict.update(parse_power_line(line, "clk"))

    except FileNotFoundError:
        try:
            df = pd.read_parquet(power_design_dir_path / "post_synth_power.pqt")
            for idx, row in df.iterrows():
                if row["Group"] in group_to_nickname.keys():
                    nickname = group_to_nickname[row["Group"]]
                    power_rpt_dict.update(
                        {
                            f"p_{nickname}_dynamic_internal": row["Total Internal Power (W)"],
                            f"p_{nickname}_dynamic_switching": row["Total Switching Power (W)"],
                            f"p_{nickname}_dynamic": row["Total Power (W)"],
                            # f"p_{nickname}_static": static_power,
                            f"p_{nickname}_mean": row["Mean Power (W)"],
                            f"p_{nickname}_max": row["Max Power (W)"],
                        }
                    )

        except Exception:
            pass

    # Parse Delay report
    try:
        with open(power_design_dir_path / "post_synth_delay.rpt", "r") as f:
            # Open the json file
            delay_report_dicts = json.load(f)

            # Get the maximum delay value
            for delay_report in delay_report_dicts["checks"]:
                if delay_report["path_group"] == "core_clock" and delay_report["path_type"] == "max":
                    max_delay_ps = delay_report["data_arrival_time"]
                    slack_is_ok = delay_report["slack"] >= 0

        power_rpt_dict["max_delay_ps"] = max_delay_ps
        power_rpt_dict["slack_is_ok"] = slack_is_ok
    except FileNotFoundError:
        # power_rpt_dict["max_delay_ps"] = None
        # power_rpt_dict["slack_is_ok"] = None
        pass

    return power_rpt_dict


def get_design_complexity(
    cmplx_dirpath: Path,
) -> dict[str, Any]:
    """[Worker function] This function read the complexity info dictionnary"""

    # Open and read the report file
    report_filepath = cmplx_dirpath / "lut_complexity_info.json"
    cmplx_dict = json.loads(report_filepath.read_text())
    cmplx_dict["design_number"] = extract_design_number_from_path(cmplx_dirpath)

    return cmplx_dict


def get_cell_count_area_trans(
    synth_design_dir_path: Path, technology_name: str = "notech_yosys"
) -> tuple[int, dict[str, int]]:
    """[Worker function] This function parses the synthesis report and extract the number of cells."""

    # Open and read the synthesized netlist and normalise it once
    synth_design_filepath = synth_design_dir_path / "mydesign_yosys.v"
    design_str = format_synthed_design(synth_design_filepath, technology_name, return_design=True, save_design=False)

    # Build a single regex to capture cell instances across the whole file
    valid_cell_types = GatesConfig.configs[technology_name]["valid_cell_types"]
    cell_definition_pattern = re.compile(
        r"(?m)^[\t ]*(" + r"|".join(re.escape(t) for t in valid_cell_types) + r")[\t ]+_[0-9]+_[\t ]*\("
    )

    # One pass over the file, count by captured type
    cell_type_iter = (m.group(1) for m in cell_definition_pattern.finditer(design_str))
    cell_count_dict: dict[str, int] = dict(Counter(cell_type_iter))

    # Analyze number of cells, number of transistors and total area
    nb_cells = 0
    nb_transistors = 0
    tot_area = 0
    for cell_name, cell_count in cell_count_dict.items():
        c_trans_count = GatesConfig.configs[technology_name]["transistor_count"][cell_name]
        nb_transistors += c_trans_count * cell_count

        c_area = GatesConfig.configs[technology_name]["gate_area"][cell_name]
        tot_area += c_area * cell_count

        nb_cells += cell_count

    # If a flowy data record file has been produced, extract the best transistor count and replace the value if lower.
    flowy_parquet_path = synth_design_dir_path / "flowy_data_record.parquet"

    if flowy_parquet_path.exists():
        flowy_df = pd.read_parquet(flowy_parquet_path)
        if "nb_transistors" in flowy_df.columns:
            nb_transistors_flowy = flowy_df["nb_transistors"].min()
            nb_transistors = min(nb_transistors, nb_transistors_flowy)

    cell_count_dict["nb_transistors"] = nb_transistors
    cell_count_dict["tot_cell_area"] = tot_area
    cell_count_dict["nb_cells"] = nb_cells
    cell_count_dict["design_number"] = "".join(filter(str.isdigit, synth_design_dir_path.name))

    return cell_count_dict


def extract_encodings(verilog_module_filepath: str, reverted: bool = False) -> dict[int, str] | dict[str, int]:
    """File function reads the encoding found int the verilog file."""
    # Encoding pattern is: `// value -> representation`
    encoding_pattern = re.compile(r"// -?\d+ -> [0-1]+")
    encodings = dict()
    enc_type = None
    vals_list: list[str] = []
    repr_list: list[str] = []
    for line in read_lines(verilog_module_filepath):
        line_lower = line.lower()
        if line.startswith("//") and "encoding" in line_lower:
            if enc_type is not None:
                # Store already read encodings
                if reverted:
                    encodings[enc_type] = {enc: int(val) for val, enc in zip(vals_list, repr_list)}
                else:
                    encodings[enc_type] = {int(val): enc for val, enc in zip(vals_list, repr_list)}

            # Encoding lists initialization
            vals_list = []
            repr_list = []
            if "input" in line_lower or "in_enc_dict" in line_lower:
                enc_type = "input"
            elif "output" in line_lower or "out_enc_dict" in line_lower:
                enc_type = "output"
            else:
                enc_type = "both"

        match = encoding_pattern.search(line)
        if match:
            value, repre = match.group(0).split("->")
            value = value.strip("/").strip()
            repre = extract_int_string_from_string(repre)
            vals_list.append(value)
            repr_list.append(repre)

    if enc_type is not None:
        # Store already read encodings
        if reverted:
            encodings[enc_type] = {enc: int(val) for val, enc in zip(vals_list, repr_list)}
        else:
            encodings[enc_type] = {int(val): enc for val, enc in zip(vals_list, repr_list)}

    # If a unique encoding is specified, it's because both inputs and output have the same encodings
    if set(encodings.keys()) == set(
        [
            "both",
        ]
    ):
        encodings["input"] = encodings["both"]
        encodings["output"] = encodings["both"]

    return encodings


def _get_gener_file_list(generation_out_dir: Path, return_numbers: bool = False) -> list[Path]:
    """Returns the list of existing generated designs."""

    if return_numbers:
        design_number_list = []
    gener_design_list = []
    for gen_dir_path in generation_out_dir.iterdir():
        hdl_dir_path = gen_dir_path / "hdl"
        if hdl_dir_path.is_dir():
            gener_design_list.append(hdl_dir_path)
            if return_numbers:
                design_number_list.append(extract_int_string_from_string(gen_dir_path.name))

    if return_numbers:
        return gener_design_list, design_number_list
    else:
        return gener_design_list


def get_list_of_power_designs_number(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[str] | pd.DataFrame:
    """Returns the list of valid power extracted design numbers"""
    return _get_list_of_valid_designs(
        dir_config,
        step="power",
        return_types="numbers",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def get_list_of_power_designs_filepath(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[list[Path]] | pd.DataFrame:
    """Returns the lists of valid power extracted design database filepaths"""
    return _get_list_of_valid_designs(
        dir_config,
        step="power",
        return_types="filepaths",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def get_list_of_power_designs_dirpath(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[Path] | pd.DataFrame:
    """Returns the list of valid power extracted design directories (containing all databases)"""
    return _get_list_of_valid_designs(
        dir_config,
        step="power",
        return_types="dirpaths",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def get_list_of_swact_designs_number(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[str] | pd.DataFrame:
    """Returns the list of valid tested design numbers"""
    return _get_list_of_valid_designs(
        dir_config,
        step="swact",
        return_types="numbers",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def get_list_of_swact_designs_filepath(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[list[Path]] | pd.DataFrame:
    """Returns the lists of valid tested design database filepaths"""
    return _get_list_of_valid_designs(
        dir_config,
        step="swact",
        return_types="filepaths",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def get_list_of_swact_designs_dirpath(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[Path] | pd.DataFrame:
    """Returns the list of valid tested design directories (containing all databases)"""
    return _get_list_of_valid_designs(
        dir_config,
        step="swact",
        return_types="dirpaths",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def get_list_of_cmplx_designs_number(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[str] | pd.DataFrame:
    """Returns the list of valid cmplxed design numbers"""
    return _get_list_of_valid_designs(
        dir_config,
        step="cmplx",
        return_types="numbers",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def get_list_of_cmplx_designs_dirpath(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[Path] | pd.DataFrame:
    """Returns the list of valid cmplxed design directories"""
    return _get_list_of_valid_designs(
        dir_config,
        step="cmplx",
        return_types="dirpaths",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def get_list_of_cmplx_designs_filepath(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[Path] | pd.DataFrame:
    """Returns the list of valid cmplxed design files (netlists)"""
    return _get_list_of_valid_designs(
        dir_config,
        step="cmplx",
        return_types="filepaths",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def get_list_of_synth_designs_number(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[str] | pd.DataFrame:
    """Returns the list of valid synthed design numbers"""
    return _get_list_of_valid_designs(
        dir_config,
        step="synth",
        return_types="numbers",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def get_list_of_synth_designs_dirpath(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[Path] | pd.DataFrame:
    """Returns the list of valid synthed design directories"""
    return _get_list_of_valid_designs(
        dir_config,
        step="synth",
        return_types="dirpaths",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def get_list_of_synth_designs_filepath(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[Path] | pd.DataFrame:
    """Returns the list of valid synthed design files (netlists)"""
    return _get_list_of_valid_designs(
        dir_config,
        step="synth",
        return_types="filepaths",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def get_list_of_gener_designs_number(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[str] | pd.DataFrame:
    """Returns the list of valid genered design numbers"""
    return _get_list_of_valid_designs(
        dir_config,
        step="gener",
        return_types="numbers",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def get_list_of_gener_designs_dirpath(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[Path] | pd.DataFrame:
    """Returns the list of valid genered design directories"""
    return _get_list_of_valid_designs(
        dir_config,
        step="gener",
        return_types="dirpaths",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def get_list_of_gener_designs_filepath(
    dir_config: object | Path,
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[Path] | pd.DataFrame:
    """Returns the list of valid genered design files (netlists)"""
    return _get_list_of_valid_designs(
        dir_config,
        step="gener",
        return_types="filepaths",
        filter_design_numbers=filter_design_numbers,
        filter_mode=filter_mode,
        get_df=get_df,
    )


def __check_valid_design(
    root_out_dir: Path,
    # design_number: str,
    step: str,
    valid_designs_df: pd.DataFrame,
    bulk_flow_dirname: str | None,
) -> dict:
    design_number = valid_designs_df["design_number"]
    res_dir_name = f"res_{design_number}"

    if len(valid_designs_df) > 0:
        status_dict = valid_designs_df.fillna(False).to_dict()
    else:
        status_dict = {
            "design_number": design_number,
            "valid_power": False,
            "valid_swact": False,
            "valid_synth": False,
            "valid_gener": False,
            "valid_cmplx": False,
        }

    if step == "cmplx":
        is_valid_cmplx = status_dict["valid_cmplx"]
        subfolder = "generation_out"
        valid_file_path = root_out_dir / subfolder / res_dir_name / "lut_complexity_info.json"
        if valid_file_path.exists():
            if os.path.getsize(valid_file_path) > 14:  # Add file only if it's not empty
                is_valid_cmplx = True

        if is_valid_cmplx:
            status_dict["valid_cmplx"] = True
        else:
            # Delete the folder
            rm_dir = root_out_dir / subfolder / res_dir_name
            if not global_vars["keep_not_valid"] and not subfolder == "generation_out" and bulk_flow_dirname is None:
                logger.info(f"Removing {rm_dir}")
                if rm_dir.exists():
                    shutil.rmtree(rm_dir)

    if step == "power":
        is_valid_power = status_dict["valid_power"]
        if bulk_flow_dirname is not None:
            subfolder = bulk_flow_dirname
        else:
            subfolder = "power_out"
        valid_file_path = root_out_dir / subfolder / res_dir_name / "synth_stat.txt"
        if valid_file_path.exists():
            if os.path.getsize(valid_file_path) > 14:  # Add file only if it's not empty
                is_valid_power = True

        if is_valid_power:
            status_dict["valid_power"] = True
        else:
            # Delete the folder
            rm_dir = root_out_dir / subfolder / res_dir_name
            if not global_vars["keep_not_valid"] and not subfolder == "generation_out" and bulk_flow_dirname is None:
                logger.info(f"Removing {rm_dir}")
                if rm_dir.exists():
                    shutil.rmtree(rm_dir)

    if step == "swact":
        # Swact
        is_valid_swact = status_dict["valid_swact"]
        if bulk_flow_dirname is not None:
            subfolder = bulk_flow_dirname
        else:
            subfolder = "test_out"
        valid_file_path = root_out_dir / subfolder / res_dir_name
        if valid_file_path.exists():
            db_not_empty = []
            for filepath in valid_file_path.iterdir():
                if "_db.csv" in filepath.name or "_db.pqt" in filepath.name or "_db.parquet" in filepath.name:
                    db_not_empty.append(os.path.getsize(filepath) > 14)
            if len(db_not_empty) > 0 and all(db_not_empty):
                is_valid_swact = True

        if is_valid_swact:
            status_dict["valid_swact"] = True
        else:
            # Delete the folder
            rm_dir = root_out_dir / subfolder / res_dir_name
            if not global_vars["keep_not_valid"] and not subfolder == "generation_out" and bulk_flow_dirname is None:
                logger.info(f"Removing {rm_dir}")
                if rm_dir.exists():
                    shutil.rmtree(rm_dir)

    if step == "synth" or (step == "swact" and not status_dict["valid_swact"]):
        # Synth
        is_valid_synth = status_dict["valid_synth"]
        if bulk_flow_dirname is not None:
            subfolder = bulk_flow_dirname
        else:
            subfolder = "synth_out"
        valid_file_path = root_out_dir / subfolder / res_dir_name / "mydesign_yosys.v"
        compressed_valid_filepath = FileCompressionHandler.get_compressed_filepath(valid_file_path)
        if valid_file_path.exists():
            if os.path.getsize(valid_file_path) > 14:  # Add file only if it's not empty
                is_valid_synth = True
        elif compressed_valid_filepath.exists():
            if os.path.getsize(compressed_valid_filepath) > 14:  # Add file only if it's not empty
                is_valid_synth = True

        if is_valid_synth:
            status_dict["valid_synth"] = True
        else:
            # Delete the folder
            rm_dir = root_out_dir / subfolder / res_dir_name
            if not global_vars["keep_not_valid"] and not subfolder == "generation_out" and bulk_flow_dirname is None:
                logger.info(f"Removing {rm_dir}")
                shutil.rmtree(rm_dir)

    # Valid generation should always be checked for
    if step == "gener" or not status_dict["valid_gener"]:
        # Gener
        is_valid_gener = status_dict["valid_gener"]
        valid_file_path = root_out_dir / "generation_out" / res_dir_name / "hdl" / "mydesign_comb.v"
        compressed_valid_filepath = FileCompressionHandler.get_compressed_filepath(valid_file_path)
        if valid_file_path.exists():
            if os.path.getsize(valid_file_path) > 14:  # Add file only if it's not empty
                is_valid_gener = True
        elif compressed_valid_filepath.exists():
            if os.path.getsize(compressed_valid_filepath) > 14:  # Add file only if it's not empty
                is_valid_gener = True

        if is_valid_gener:
            status_dict["valid_gener"] = True
        else:
            # Delete the folder
            rm_dir = root_out_dir / "generation_out" / res_dir_name
            if not global_vars["keep_not_valid"]:
                logger.info(f"Removing {rm_dir}")
                shutil.rmtree(rm_dir)

    return status_dict


def __get_valid_file_path(
    design_number: str,
    root_dirpath_str: str,
    step: str,
    bulk_flow_dirname: str | None = None,
):
    if step == "gener":
        return Path(f"{root_dirpath_str}/generation_out/res_{design_number}/hdl/mydesign_comb.v")
    elif step == "cmplx":
        return Path(f"{root_dirpath_str}/generation_out/res_{design_number}/lut_complexity_info.json")
    elif step == "synth":
        if bulk_flow_dirname is not None:
            return Path(f"{root_dirpath_str}/{bulk_flow_dirname}/res_{design_number}/mydesign_yosys.v")
        else:
            return Path(f"{root_dirpath_str}/synth_out/res_{design_number}/mydesign_yosys.v")
    else:
        # Did not implement, because case has an error in original if condition.
        raise ValueError(f"Unexpected case: step == {step}.")


def __get_res_dir_path(
    design_number: str,
    root_dirpath_str: str,
    step: str,
    bulk_flow_dirname: str | None = None,
):
    if bulk_flow_dirname is not None and step not in ["gener", "cmplx"]:
        return Path(f"{root_dirpath_str}/{bulk_flow_dirname}/res_{design_number}")
    else:
        if step == "gener":
            return Path(f"{root_dirpath_str}/generation_out/res_{design_number}")
        elif step == "cmplx":
            return Path(f"{root_dirpath_str}/generation_out/res_{design_number}")
        elif step == "synth":
            return Path(f"{root_dirpath_str}/synth_out/res_{design_number}")
        elif step == "swact":
            return Path(f"{root_dirpath_str}/test_out/res_{design_number}")
        elif step == "power":
            return Path(f"{root_dirpath_str}/power_out/res_{design_number}")


def __get_valid_designs_db(root_dirpath: Path, step: str, bulk_flow_dirname: str | None = None) -> pd.DataFrame:
    start_time = time()
    # Open the already parsed valid_designs database
    valid_design_db_filepath = root_dirpath / "valid_designs.db.pqt"
    dtype_map = {
        "design_number": str,
        "valid_power": bool,
        "valid_gener": bool,
        "valid_synth": bool,
        "valid_swact": bool,
        "valid_cmplx": bool,
    }
    max_tries = 10
    valid_designs_df = None
    for _ in range(max_tries):
        try:
            valid_designs_df = pd.read_parquet(valid_design_db_filepath)
            logger.info(f"Loaded valid_designs_df from:")
            logger.info(valid_design_db_filepath)
            break
        except ArrowInvalid:
            # File is being written by another process
            sleep(1)
        except FileNotFoundError:
            break
        except Exception as e:
            raise NotImplementedError(f"Specific error handling for {e} should be implemented.")

    if valid_designs_df is None:
        valid_designs_df = pd.DataFrame({c: [] for c in dtype_map.keys()}).astype(dtype=dtype_map)

    # Add potentially missing columns and set values to false.
    missing_columns = set(dtype_map.keys()) - set(valid_designs_df.columns) - {"design_number"}
    for c in missing_columns:
        valid_designs_df[c] = False

    # Set the dir to explore
    if bulk_flow_dirname is not None and step != "gener":
        explored_dir_path = root_dirpath / bulk_flow_dirname
    elif step == "gener":
        explored_dir_path = root_dirpath / "generation_out"
    elif step == "cmplx":
        explored_dir_path = root_dirpath / "generation_out"
    elif step == "synth":
        explored_dir_path = root_dirpath / "synth_out"
    elif step == "swact":
        explored_dir_path = root_dirpath / "test_out"
    elif step == "power":
        explored_dir_path = root_dirpath / "power_out"
    else:
        raise NotImplementedError(f"Step {step} not implemented!")

    # Get the list of designs numbers that exist for the current step
    existing_design_numbers = set(
        pd.Series(os.listdir(explored_dir_path)).map(lambda x: extract_int_string_from_string(x))
    )

    # Handled the designs that no longer exist in "generation_out"
    dissappeared_design_numbers = set(valid_designs_df["design_number"]) - existing_design_numbers
    if len(dissappeared_design_numbers) > 0:
        # Some designs were presents but are not there anymore => Remove them from the db
        logger.info(f"Switching known {len(dissappeared_design_numbers)} designs to not valid in db")
        disappeared_mask = valid_designs_df["design_number"].isin(dissappeared_design_numbers)
        valid_designs_df.loc[disappeared_mask, f"valid_{step}"] = False

    # Re-check all non entirely valid designs
    valid_power_mask = valid_designs_df["valid_power"]
    valid_gener_mask = valid_designs_df["valid_gener"]
    valid_synth_mask = valid_designs_df["valid_synth"]
    valid_swact_mask = valid_designs_df["valid_swact"]
    valid_cmplx_mask = valid_designs_df["valid_cmplx"]
    if step == "power":
        all_valid_mask = valid_gener_mask & valid_power_mask
    elif step == "cmplx":
        all_valid_mask = valid_gener_mask & valid_cmplx_mask
    elif step == "swact":
        # Swact analysis requires synth analysis
        all_valid_mask = valid_gener_mask & valid_synth_mask & valid_swact_mask
    elif step == "synth":
        all_valid_mask = valid_gener_mask & valid_synth_mask
    elif step == "gener":
        all_valid_mask = valid_gener_mask

    all_valid_design_numbers = set(valid_designs_df["design_number"][all_valid_mask])
    non_already_valid_design_numbers = existing_design_numbers - all_valid_design_numbers
    reduced_valid_designs_df = valid_designs_df[
        valid_designs_df["design_number"].isin(non_already_valid_design_numbers)
    ]

    # Missing rows
    missing_design_numbers = non_already_valid_design_numbers - set(reduced_valid_designs_df["design_number"])
    missing_dn_data = []
    for design_number in missing_design_numbers:
        missing_dn_data.append(
            {
                "design_number": design_number,
                "valid_power": False,
                "valid_gener": False,
                "valid_synth": False,
                "valid_swact": False,
                "valid_cmplx": False,
            }
        )
    reduced_valid_designs_df = pd.concat([reduced_valid_designs_df, pd.DataFrame(missing_dn_data)], ignore_index=True)

    if len(non_already_valid_design_numbers) > 0:
        logger.info(f"Checking {len(non_already_valid_design_numbers)} new designs ")
        # Check all non already valid design numbers
        if global_vars.get("debug"):  # Disable mutliprocessing
            checked_designs = [
                __check_valid_design(
                    root_dirpath,
                    step,
                    row,
                    bulk_flow_dirname,
                )
                for idx, row in reduced_valid_designs_df.iterrows()
            ]
        else:
            checked_designs = process_pool_helper(
                func=__check_valid_design,
                func_args_gen=(
                    (root_dirpath, step, row, bulk_flow_dirname) for idx, row in reduced_valid_designs_df.iterrows()
                ),
                error_message="Validity could not be checked for some designs.",
            )

        # Concatenate new db with previous one and overwrites previous values with new values
        new_valid_designs_df = pd.DataFrame(checked_designs)
        new_valid_designs_df = utils.add_new_df_to_df(new_valid_designs_df, valid_designs_df)

        logger.info(f"Valid Designs database file updated with {len(non_already_valid_design_numbers)} new designs:")
        logger.info(valid_design_db_filepath)

    else:
        new_valid_designs_df = valid_designs_df

    try:
        new_valid_designs_df.to_parquet(valid_design_db_filepath, index=False)
    except Exception:
        logger.error(f"Could not update valid design DB file {valid_design_db_filepath}")

    end_time0 = time()
    logger.debug(f"Updating valid design DB took {(end_time0 - start_time) / 60:2f}min")

    return new_valid_designs_df


def __get_do_get_numbers(return_types: list[str] | str):
    return "numbers" in return_types or "numbers" == return_types


def __get_do_get_filepaths(return_types: list[str] | str):
    return "filepaths" in return_types or "filepaths" == return_types


def __get_do_get_dirpaths(return_types: list[str] | str):
    return "dirpaths" in return_types or "dirpaths" == return_types


def __build_all_designs(
    root_dirpath: Path,
    step: str,
    valid_design_numbers: set[str],
    return_types: list[str] | str,
    get_df: bool = False,
    bulk_flow_dirname: str | None = None,
) -> list[str | Path] | pd.DataFrame:
    """This function uses multiprocessing to return a full list of valid design numbers."""
    start_time = time()
    number_els = len(return_types) if isinstance(return_types, list) else 1
    if return_types == "numbers":
        logger.info(f"Returning all valid {step}ed design numbers paths ... ")
        return valid_design_numbers

    if valid_design_numbers:
        logger.info(f"Building and returning all design paths ... ")

        do_get_numbers = __get_do_get_numbers(return_types)
        do_get_filepaths = __get_do_get_filepaths(return_types)
        do_get_dirpaths = __get_do_get_dirpaths(return_types)

        assert any([do_get_numbers, do_get_filepaths, do_get_dirpaths])

        df = pd.DataFrame(list(valid_design_numbers), columns=["design_number"])

        if do_get_numbers:
            df["design_path"] = df["design_number"]
        elif do_get_filepaths:
            df["design_path"] = df["design_number"].map(
                lambda x: __get_valid_file_path(x, str(root_dirpath), step, bulk_flow_dirname)
            )
        elif do_get_dirpaths:
            df["design_path"] = df["design_number"].map(
                lambda x: __get_res_dir_path(x, str(root_dirpath), step, bulk_flow_dirname)
            )

        end_time = time()
        logger.debug(f"Building all paths took {(end_time - start_time) / 60:2f}min")

        if get_df:
            return df
        else:
            return df["design_path"].tolist()
    else:
        # No valid design have been found, return empty list/dataframe
        if number_els > 1:
            return pd.DataFrame(columns=["filepaths", "dirpaths", "numbers"])
        else:
            return []


# def __yield_all_designs(root_dirpath:Path, step:str, valid_design_numbers:set[str], return_types:list[str]|str) -> Generator[str|Path|pd.DataFrame,None,None]:
#     """ Same as __build_all_designs but yields instead """
#     function_args, _ = __prepare_args_for_build(root_dirpath, step, valid_design_numbers, return_types)
#     for design_number in valid_design_numbers:
#         yield __build_valid_design(design_number, *function_args)


def _filter_list_of_valid_designs(
    dir_config: object | Path,
    step: str,
    return_types: list[str] | str = "numbers",
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
):
    """This function parses the synth output dir and get all design_numbers of already existing designs."""

    logger.info(f"Reading out list of valid {step}ed design {return_types} ... ")

    assert step in ["gener", "cmplx", "synth", "swact", "power"]
    assert not (global_vars["synth_only"] and step == "swact"), (
        f"Please make sure that step == `swact` is not called when global_vars['synth_only'] is True"
    )
    assert not (global_vars["synth_only"] and step == "power"), (
        f"Please make sure that step == `power` is not called when global_vars['synth_only'] is True"
    )

    if isinstance(dir_config, Path):
        root_dirpath = dir_config
    else:
        root_dirpath = dir_config.root_output_dir

    # Step 1:
    # Update the database of valid designs
    new_valid_designs_df = __get_valid_designs_db(root_dirpath, step, dir_config.bulk_flow_dirname)

    # Step 2:
    # Filter out all valid designs based on the list of design numbers to filter (provided by user)
    valid_design_numbers = set(new_valid_designs_df["design_number"][new_valid_designs_df[f"valid_{step}"]])
    if filter_design_numbers is not None:
        nb_filter_designs = len(filter_design_numbers)
        if nb_filter_designs > 0:
            _filter_design_numbers = set(filter_design_numbers)
            if filter_mode == "include":
                valid_design_numbers = valid_design_numbers.intersection(_filter_design_numbers)
            elif filter_mode == "exclude":
                valid_design_numbers = valid_design_numbers - _filter_design_numbers
            else:
                raise ValueError(
                    f"`filter_mode` argument should be among [`include`,`exclude`], received: {filter_mode}"
                )
        else:
            if filter_mode == "include":
                # Include nothing
                valid_design_numbers = []
            elif filter_mode == "exclude":
                # Exclude nothing
                valid_design_numbers = valid_design_numbers
    else:
        nb_filter_designs = 0

    logger.info(
        f"There are {len(valid_design_numbers)} valid {step}ed design numbers ({str(filter_mode)[:-1]}ing {nb_filter_designs} filtered designs)"
    )

    return root_dirpath, valid_design_numbers


def _get_list_of_valid_designs(
    dir_config: object | Path,
    step: str,
    return_types: list[str] | str = "numbers",
    filter_design_numbers: list[str] = None,
    filter_mode: str | None = None,
    get_df: bool = False,
) -> list[str | Path] | pd.DataFrame:
    """
    This function calls the filter valid design function and then call the list builder to return a list or a datafrmae of designs path or number
    """
    root_dirpath, valid_design_numbers = _filter_list_of_valid_designs(
        dir_config, step, return_types, filter_design_numbers, filter_mode
    )
    return __build_all_designs(
        root_dirpath,
        step,
        valid_design_numbers,
        return_types,
        get_df=get_df,
        bulk_flow_dirname=dir_config.bulk_flow_dirname,
    )


def get_genered_design_dir_path(generation_out_dir: Path, design_number: str) -> Path:
    """Returns the path to the generated design associated with the design_number"""
    return generation_out_dir / f"res_{design_number}/hdl"


def get_genered_design_file_path(generation_out_dir: Path, design_number: str) -> Path:
    """Returns the path to the generated design associated with the design_number"""
    return get_genered_design_dir_path(generation_out_dir, design_number) / "mydesign_comb.v"


def extract_design_number_from_path(path: Path) -> str:
    """This funciton extracts the design number from the path"""

    if path.is_file():
        _path = path.parent
    else:
        _path = path

    if _path.is_dir() or not _path.exists():
        if _path.name.startswith("res_"):
            return extract_int_string_from_string(_path.name)

        for i in range(4):
            if _path.parents[i].name.startswith("res_"):
                return extract_int_string_from_string(_path.parents[i].name)

    else:
        logger.error(f"Received path {path}")
        raise NotImplementedError


def read_all_existing_encoding(
    root_existing_designs_path: Path, curr_root_output_path: Path | None = None, type: str = "analyzer"
) -> dict[str, dict[int, str]]:
    """
    This function reads all encoding dictionnaries of the provided encoder generation directory.
    To accelerate the process, it also stores and/or load the encoder encodings dictionnary and check validity of pre-existing data.
    """

    logger.info(f"Reading out all existing encoding dictionnaries in:")
    logger.info(root_existing_designs_path)
    genered_encoder_dir_list, genered_encoder_number_list = _get_gener_file_list(
        root_existing_designs_path / "generation_out", return_numbers=True
    )
    zipped_encoder_lists = zip(genered_encoder_dir_list, genered_encoder_number_list)

    # Compare with pre-saved files
    do_read_encoders = True
    if type == "analyzer":
        known_encoders_filepath = curr_root_output_path / "associated_encoders.npz"
        if known_encoders_filepath.exists():
            known_encoders_dict = utils.load_serialized_data(known_encoders_filepath)
            if len(genered_encoder_dir_list) == len(known_encoders_dict.keys()):
                # Open 5 randomly chosen encoders and verify that the encoding dicts are the same
                encoder_samples = random.choices(list(zipped_encoder_lists), k=10)
                for genered_encoder_dir, encoder_number in encoder_samples:
                    encoder_lut_path = genered_encoder_dir / "mydesign_comb.v"
                    encoder_encodings_dict = extract_encodings(encoder_lut_path)
                    if known_encoders_dict[encoder_number] == encoder_encodings_dict["output"]:
                        continue
                    else:
                        break
                # If all sampled encoders have not changed, we can safely assume that the encoder files are still valid.
                # We can thus skip reading all  encoder files and simply open the existing .npz-stored dictionnary
                do_read_encoders = False

    if do_read_encoders:
        known_encoders_dict = dict()
        # Read all encoding from encoder design verilog files directly
        for genered_encoder_dir, encoder_number in zipped_encoder_lists:
            encoder_lut_path = genered_encoder_dir / "mydesign_comb.v"
            encoder_encodings_dict = extract_encodings(encoder_lut_path)

            if type == "analyzer":
                known_encoders_dict[encoder_number] = encoder_encodings_dict["output"]
            elif type == "generator":
                known_encoders_dict[encoder_number] = encoder_encodings_dict

        if type == "analyzer":
            # Save the data to accelerate next Analyzer run
            utils.save_serialized_data(known_encoders_filepath, known_encoders_dict)
            logger.info(f"Saved all parsed encoders encoding dictionnaries to {known_encoders_filepath}.")

    return known_encoders_dict


def _extract_encodings(root_output_path: Path, design_number: str) -> dict:
    # Get Path
    gener_filepath = get_genered_design_file_path(root_output_path / "generation_out", design_number)

    # Read Encoding
    encodings_dict = extract_encodings(gener_filepath)

    # Create data dict
    try:
        return {
            "design_number": design_number,
            "enc_dict_str": str(encodings_dict),
            "in_enc_dict": str(encodings_dict["input"]),
            "out_enc_dict": str(encodings_dict["output"]),
        }
    except KeyError:
        logger.error(f"Reading out encoding dictionnary of design number {design_number} generated an error.")
        return {
            "design_number": design_number,
            "enc_dict_str": "error",
            "in_enc_dict": "error",
            "out_enc_dict": "error",
        }


def read_all_existing_encodings_v2(
    root_output_path: Path | None = None, bulk_flow_dirname: str | None = None
) -> pd.DataFrame:
    """
    This function reads all encoding dictionnaries of the provided encoder generation directory.
    To accelerate the process, it also stores and/or load the encoder encodings dictionnary and check validity of pre-existing data.
    """

    logger.info(f"Reading out all existing encoding dictionaries in:")
    logger.info(root_output_path)
    # genered_encoder_dir_list, genered_encoder_number_list = _get_gener_file_list(root_output_path / "generation_out", return_numbers=True)
    # zipped_encoder_lists = zip(genered_encoder_dir_list, genered_encoder_number_list)

    # Open already parsed encodings
    known_encoders_filepath = root_output_path / "encodings_dicts.db.pqt"
    if known_encoders_filepath.exists():
        encoding_dicts_df = pd.read_parquet(known_encoders_filepath)
    else:
        encoding_dicts_df = pd.DataFrame()
    if not encoding_dicts_df.empty:
        known_design_numbers = set(encoding_dicts_df["design_number"])
    else:
        known_design_numbers = set()

    # Check the encodings tha are not yet known
    dir_dict = {
        "bulk_flow_dirname": bulk_flow_dirname,
        "root_output_dir": root_output_path,
    }
    config = SimpleNamespace(**dir_dict)
    valid_design_numbers = get_list_of_gener_designs_number(config)
    remaining_design_numbers = set(valid_design_numbers) - known_design_numbers

    logger.info(f"Extracting {len(remaining_design_numbers)} encodings from generated design LUT files ...")

    data_list = []
    with tqdm(total=len(remaining_design_numbers), desc=f"x128| Reading out design encodings") as pbar:  # Progress bar
        data_list = process_pool_helper(
            func=_extract_encodings,
            func_args_gen=((root_output_path, design_number) for design_number in remaining_design_numbers),
            error_message="encoding_read_errors has occurred!!!",
            pbar=pbar,
            max_workers=128,
        )

    new_encoding_dicts_df = pd.DataFrame(data_list)
    if encoding_dicts_df.empty:
        encoding_dicts_df = new_encoding_dicts_df
    else:
        encoding_dicts_df = pd.concat([encoding_dicts_df, new_encoding_dicts_df], ignore_index=True)

    encoding_dicts_df.to_parquet(known_encoders_filepath)
    logger.info(f"Updated known encodings database with {len(encoding_dicts_df) - 1} new designs.")

    return encoding_dicts_df


def _filter_design_path(path: Path, design_number_set: set[str], mode: str):
    if path.name == "hdl":
        folder_name = path.parent.name
    else:
        folder_name = path.name

    design_number = extract_int_string_from_string(folder_name)

    is_in_target_list = design_number in design_number_set

    if mode == "include" and is_in_target_list:
        return path
    elif mode == "exclude" and not is_in_target_list:
        return path
    else:
        return None


def filter_design_path_list(
    design_path_list: list[Path], target_design_number_list: list[str], mode: str = "include"
) -> list[Path]:
    """
    Return the list of design paths present in `design_path_list` that (do not) have the design_number in the `target_design_number_list` when mode is `include (`exclude`).
    Args:
        design_path_list (list[Path]): List of design paths to filter.
        target_design_number_list (list[str]): List of design numbers to filter.
        mode (str, optional): "include" or "exclude". Defaults to "include".
            Whether or not to inlcude or exclude the designs numbers of `target_design_number_list`.
            If `include`, the design numbers of target_design_number_list will be kept.
    """
    assert mode in ["include", "exclude"], "mode must be either 'include' or 'exclude'."
    logger.info(f"Filtering designs path list ... (mode = {mode})")
    start_time = time()

    # design_number_set = set(target_design_number_list)

    todo_design_numbers = [extract_design_number_from_path(path) for path in design_path_list]
    todo_design_numbers_set = set(todo_design_numbers)
    target_design_number_set = set(target_design_number_list)

    if mode == "inlcude":
        todo_design_numbers_set = todo_design_numbers_set.intersection(target_design_number_set)
        logger.info("include set")

    elif mode == "exclude":
        todo_design_numbers_set = todo_design_numbers_set - target_design_number_set
        logger.info("exclude set")

    remaining_paths = []
    for path in design_path_list:
        design_number = extract_design_number_from_path(path)
        if design_number in todo_design_numbers_set:
            remaining_paths.append(path)

    end_time = time()
    logger.info(f"Filtered list of design from {len(design_path_list)} to {len(remaining_paths)} designs. ")
    logger.info(f"It took {(end_time - start_time) / 60:.2f}min.")
    return remaining_paths


def open_file(*args):
    return FileCompressionHandler.open_file(*args)


def read(*args):
    return FileCompressionHandler.read(*args)


def read_lines(*args):
    return FileCompressionHandler.read_lines(*args)


def get_encoding_dict_from_file(design_filepath: Path):
    if design_filepath.suffix == ".v" or design_filepath.suffixes == [".v", ".bz2"]:
        encoding_dict = extract_encodings(design_filepath)
    elif design_filepath.suffix == ".json":
        # We can also receive an encoding dictionnary as a json file directly.
        try:
            with open(design_filepath, "r") as f:
                encoding_dict = json.load(f)
        except Exception:
            encoding_dict = eval(open(design_filepath).read().strip())

    else:
        raise ValueError("The design file must be a .v or .json file.")

    lut_build_dict = {}

    # Clean up exp config if needed
    if "input" in encoding_dict.keys():
        lut_build_dict["in_enc_dict"] = encoding_dict["input"]
    elif "encoding" in encoding_dict.keys():
        lut_build_dict["in_enc_dict"] = {
            int(k): v for k, v in encoding_dict["encoding"]["data"]["input"]["data"].items()
        }
    else:
        logger.error(f"Missing 'input' key in the the encoding of the design file.")

    if "output" in encoding_dict.keys():
        lut_build_dict["out_enc_dict"] = encoding_dict["output"]
    elif "encoding" in encoding_dict.keys():
        lut_build_dict["out_enc_dict"] = {
            int(k): v for k, v in encoding_dict["encoding"]["data"]["output"]["data"].items()
        }
    else:
        logger.error(f"Missing 'output' key in the the encoding of the design file.")

    return lut_build_dict
