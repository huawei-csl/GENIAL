# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

from pathlib import Path
from typing import Any
import pandas as pd
from genial.utils.utils import process_pool_helper
from loguru import logger
from copy import copy
import numpy as np
from tqdm import tqdm
import re
import json
import os
from time import time
import traceback

from functools import reduce

import genial.experiment.plotter as plotter
import genial.experiment.file_parsers as file_parsers
from swact import (
    apply_weights_to_switch_vectors as swact_apply_weights_to_switch_vectors,
    get_swact_table,
    reformat_result_databases as swact_reformat_result_databases,
)
from swact.netlist import format_synthed_design
from swact.netlist import get_fanout_wires_n_depth, get_wire_list
from genial.utils import utils
from genial.config.logging import Logging as logging
from genial.config.config_dir import ConfigDir
from genial.config.arg_parsers import analyzer_parser
from genial.experiment.task_generator import DesignGenerator
from swact.gates_configuration import GatesConfig
from genial.experiment.loop_module import LoopModule
from genial.globals import global_vars


class Analyzer(LoopModule):
    """
    Class to operate the analysis of all tasks.
    """

    __existing_steps__ = ["cmplx", "synth", "swact", "power", "gener"]

    def __init__(
        self,
        dir_config: ConfigDir | None = None,
        reset_logs: bool = False,
        skip_log_init: bool = False,
        read_only: bool = False,
    ) -> None:
        super().__init__()

        logger.info(f"Setting up Analyzer ...")

        # Initialize the experiment configuration
        if dir_config is None:
            self.args_dict = Analyzer.parse_args()
            self.dir_config = ConfigDir(is_analysis=True, **self.args_dict)
        else:
            self.args_dict = dir_config.args_dict
            self.dir_config = dir_config

        # Setup some important variables
        self.is_debug = self.args_dict.get("debug", False)
        self.read_only = read_only  # Enable to not raise warnings when loading missing databases
        global_vars["synth_only"] = self.args_dict.get("synth_only", False)

        # Outputs
        self.analysis_out_dir = self.dir_config.analysis_out_dir
        self.analysis_out_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = self.analysis_out_dir / "plots"
        if not self.plot_dir.exists():
            self.plot_dir.mkdir(parents=True)

        # Analysis Configuration
        self.skip_fullsweep_analysis = self.args_dict.get("skip_fullsweep_analysis", False)
        self.skip_tests_list = self.args_dict.get("skip_tests_list", [])
        if not skip_log_init:
            self.init_log_files(reset_logs)

        self.special_designs_dict = self.dir_config.read_special_designs(self.dir_config)
        self.special_designs_ids = self.special_designs_dict["design_numbers"]

        # Setup success or failures report
        self.status_report_dict = {key: [] for key in Analyzer.__existing_steps__}

        # Setup some outputs
        true_once = False
        for step in Analyzer.__existing_steps__:
            value: list[tuple[Path, str]] = []
            setattr(self, f"failed_{step}_analysis", value)
            value: list[str] = []
            setattr(self, f"incomplete_{step}", value)
            value: list[Path] = []
            setattr(self, f"successful_{step}_analysis", value)
            value = self.analysis_out_dir / f"{step}_analysis.db.pqt"
            setattr(self, f"output_{step}_db_path", value)

            if self.args_dict.get(f"{step}_only", False):
                if true_once:
                    err_msg = (
                        "Several `{steps}_only` arguemnts have been received by the Analyzer. Please use only one."
                    )
                    logger.error(err_msg)
                    raise ValueError(err_msg)
                logger.info(
                    f"Only analysis of step `{step}` will be performed because `{step}_only` has been received."
                )
                global_vars[f"{step}_only"] = True
                true_once = True

        # Open databases if needed
        do_format = False
        for step in Analyzer.__existing_steps__:
            load_db_cond = self.args_dict.get(f"skip_{step}", False) or not self.args_dict.get("rebuild_db", False)

            if load_db_cond:
                df = self._load_database(
                    getattr(self, f"output_{step}_db_path"),
                    force=(
                        self.dir_config.is_bulk_flow_mode
                        or self.args_dict.get(f"continue")
                        or self.read_only
                        or self.args_dict.get("ignore_user_prompts", False)
                    ),
                )
                if not df.empty:
                    df = self._add_special(df)  # Make sure special list is up to date
                    setattr(self, f"{step}_df", df)
                    logger.info(f"{step}ed analysis database loaded from pre-existing file:")
                    logger.info(getattr(self, f"output_{step}_db_path"))
                    do_format = True
                else:
                    setattr(self, f"{step}_df", pd.DataFrame())
                    logger.info(f"{step}ed analysis database has been loaded but is empty.")

            else:
                setattr(self, f"{step}_df", pd.DataFrame())
                logger.info(f"{step}ed analysis database has not been loaded.")

        # Format opened databases
        if do_format:
            self.format_databases()

        # Get the number of different tests available in the dataset
        self.test_type_names = self.dir_config.exp_config["do_tests_list"]
        self.test_type_nb = len(self.test_type_names)
        if not self.swact_df.empty:
            if self.test_type_names != list(self.swact_df["test_type"].unique()):
                self.test_type_names = list(self.swact_df["test_type"].unique())
                self.test_type_nb = len(self.test_type_names)

        # Setup design number list
        self.is_user_design_number_list = (
            self.args_dict.get("design_number_list", None) is not None
        ) or self.args_dict.get("continue", False)
        if self.is_user_design_number_list:
            logger.info(f"Design number list has been received.")

        _user_design_list = self.args_dict.get("design_number_list", None)
        if _user_design_list is None:
            if self.args_dict.get("continue", False):
                logger.info(f"Updating list of designs to analyze for continuing the analysis...")
                _user_design_list = self._get_continue_todo_design_numbers()
                if _user_design_list is None:
                    logger.warning("No new designs to be analyzed have been found. ")
                    logger.info("Nothing will happen.")
                else:
                    logger.info(
                        f"Analysis will continue with list of designs to analyze set to {len(_user_design_list)} designs."
                    )

        self.user_design_number_list = _user_design_list
        if self.user_design_number_list is not None:
            if isinstance(self.user_design_number_list, list):
                logger.warning(f"List of designs to analyze have set and contains {len(self.user_design_number_list)}")
            if isinstance(self.user_design_number_list, dict):
                pass

        # Setup filepaths todo
        if not read_only:
            self._setup_lists_of_designs_todo()

        # Nb Workers
        nb_workers = self.args_dict.get("nb_workers", None)
        self.nb_workers = 128 if nb_workers is None else nb_workers
        if self.is_debug:
            self.nb_workers = 1
        logger.opt(colors=True).info(f"Running tasks with <red>{self.nb_workers}</red> workers in parallel.")

        # Experiment configuration
        self.exp_config = self.dir_config.get_experiment_configuration()
        self.synth_instance_names = self.exp_config["top_synth_instance_names"]
        self.experiment_name = self.args_dict.get("experiment_name")

        logger.info(f"Analyzer initialized.\n")

    def _get_continue_todo_design_numbers(self) -> list[str]:
        """
        This function is called during __init__ if args contais continue, or by switch_to_iter_mode.
        It returns the list of designs that should be run by the analyzer.
        It first look at all valid (synthed + swacted) designs.
        And from them, it removes the designs whose design IDs are already present in the analyzer's databases.
        """
        logger.info(f"Evaluating the lists of designs to analyze")

        _user_design_list = {}

        for step in Analyzer.__existing_steps__:
            if not self.args_dict.get(f"skip_{step}", False):
                # Get list of `step`ed design that have already been analyzed:
                step_df = getattr(self, f"{step}_df")
                if "design_number" in step_df.columns:
                    _valid_step_design_numbers = set(step_df["design_number"])
                    to_analyze_step_design_numbers = set(
                        eval(
                            f"file_parsers.get_list_of_{step}_designs_number(self.dir_config, filter_design_numbers=_valid_step_design_numbers, filter_mode='exclude')"
                        )
                    )
                else:
                    to_analyze_step_design_numbers = set(
                        eval(f"file_parsers.get_list_of_{step}_designs_number(self.dir_config)")
                    )
                _user_design_list[step] = to_analyze_step_design_numbers
                logger.info(
                    f"List of {step}ed designs to analyze contains {len(to_analyze_step_design_numbers)} designs."
                )

        return _user_design_list

    def _setup_lists_of_designs_todo(self) -> None:
        """
        This function prepare the lists of designs filepaths that will be run during the analyzer main call.
        If is_user_design_number_list: removes the design numbers from the databases and only keep their filepaths
        If is_debug: only keep two filepaths in the list of designs todo
        Else: keep all valid filepaths found in the synthed and swacted directories
        """

        for step in Analyzer.__existing_steps__:
            step_df = getattr(self, f"{step}_df")

            target_user_design_number_list = []
            # Get the list of designs to do
            if (self.is_user_design_number_list) and (self.user_design_number_list is not None):
                if isinstance(self.user_design_number_list, list):
                    target_user_design_number_list = self.user_design_number_list
                elif step in self.user_design_number_list.keys():
                    target_user_design_number_list = self.user_design_number_list[step]

            # If list of design has been specified, remove them from pre-exisiting databases to make sure they will be analyzed again
            if len(target_user_design_number_list) > 0:
                if not step_df.empty:
                    # Remove all existing lines associated to design numbers to be evaluated
                    existing_indexes = step_df[step_df["design_number"].isin(target_user_design_number_list)].index
                    step_df = step_df.drop(index=existing_indexes)
                    logger.warning(
                        f"Removed specified designs from pre-existing {step} database. They will be re-analyzed during this run."
                    )
                    setattr(self, f"{step}_df", step_df)

            # For all designs that need to be analyzed, add their path to the list of designs todo
            logger.info(f"Updating list of {step}ed designs to analyze ...")
            if not self.args_dict.get(f"skip_{step}"):
                if self.is_user_design_number_list:
                    logger.info(f"Getting only required {step} designs based on reducing list")
                    step_designs_dir_paths_list = eval(
                        f"file_parsers.get_list_of_{step}_designs_dirpath(self.dir_config, filter_design_numbers=target_user_design_number_list, filter_mode='include')"
                    )
                else:
                    logger.info(f"Getting all {step} designs")
                    step_designs_dir_paths_list = eval(
                        f"file_parsers.get_list_of_{step}_designs_dirpath(self.dir_config)"
                    )
                if self.is_debug:
                    logger.debug(
                        f"DEBUG IS ON: list of {step}ed designs to analyze has been reduced to 2 out the the {len(step_designs_dir_paths_list)} original elements"
                    )
                    step_designs_dir_paths_list = step_designs_dir_paths_list[:2]
            else:
                if self.args_dict.get(f"skip_{step}"):
                    logger.info(
                        f"skip_{step} is True, list of {step}ed designs to analyze has been set to an empty list"
                    )

                step_designs_dir_paths_list = []

            setattr(self, f"{step}_designs_dir_paths_list", step_designs_dir_paths_list)

            if len(step_designs_dir_paths_list) == 0:
                logger.warning(f"List of {step}ed designs to analyze is empty!")
            else:
                logger.info(
                    f"List of {step}ed designs to analyze has been updated and contains {len(step_designs_dir_paths_list)} paths."
                )

        return None

    def init_log_files(self, reset_logs: bool):
        """This function initialize all log files"""

        if self.args_dict.get("skip_synth", False):
            logfile_mode = "analyzer_swact"
        elif self.args_dict.get("skip_swact", False):
            logfile_mode = "analyzer_synth"
        else:
            logfile_mode = "analyzer_full"

        self.logdir = self.dir_config.root_output_dir / "logs"

        logging().init_logging(log_dirpath=self.logdir, mode=logfile_mode, reset=reset_logs)

    def extend_special_designs(self, design_number: str, legend: str) -> None:
        """Extend the special designs with the specified design."""
        special_design_filepath = self.dir_config.special_designs_filepath
        if design_number in self.special_designs_dict["design_numbers"]:
            logger.warning(
                f"Tried to update special design list, but design_number {design_number} already found. Skipping."
            )
            return None
        elif legend in self.special_designs_dict["legend"]:
            existing_design_number = self.special_designs_dict["design_numbers"][
                self.special_designs_dict["legend"].index(legend)
            ]
            logger.warning(
                f"Tried to update special design list, but legend {design_number} already found. Associated with design_number: {existing_design_number}. Skipping."
            )
            return None

        self.special_designs_dict["design_numbers"].append(design_number)
        self.special_designs_dict["legend"].append(legend)
        json.dump(self.special_designs_dict, open(special_design_filepath, "w"))
        logger.info(f"Special Design file {special_design_filepath} has been updated with design {design_number}")

        return None

    @staticmethod
    def _load_database(db_path: Path, force: bool = False) -> pd.DataFrame:
        """Wrapper for loading a database."""

        if not db_path.exists() and not db_path.with_suffix(".pqt").exists():
            logger.error(f"Something went wrong when loading {db_path}.")
            logger.info(f"Database file {db_path} does not exists.")
            logger.error(
                "Maybe you forgot the command line argument `--rebuild_db` ... (By default it is not set, which avoids accidentally erasing the pre-existing databases.)"
            )
            if not force:
                answer = input("Do you know what you are doing? Answer [Y]es or [N]o.")
                if answer.lower().startswith("y"):
                    df = pd.DataFrame()
                    return df
                else:
                    exit(1)
            else:
                df = pd.DataFrame()
                return df

        df = utils.load_database(db_path)

        return df

    def get_all_number_of_cells(self):
        """This function redirects the worker function to the correct synthesis paths."""

        logger.info(f"Parsing all synthesis reports to get number of cells ... ")
        try:
            db_size_before = len(self.synth_df)
        except Exception:
            db_size_before = 0

        start = time()
        data_list = []
        if self.args_dict.get("rebuild_db", False) or self.is_user_design_number_list:
            desc = f"x{self.nb_workers}| {self.experiment_name} | Parse synthesis reports"
            with tqdm(total=len(self.synth_designs_dir_paths_list), desc=desc) as pbar:  # Progress bar
                data_list = utils.process_pool_helper(
                    func=file_parsers.get_cell_count_area_trans,
                    func_args_gen=(
                        (design_dir_path, self.dir_config.technology)
                        for design_dir_path in self.synth_designs_dir_paths_list
                    ),
                    pbar=pbar,
                )

            if data_list:
                new_synth_df = pd.DataFrame(data_list)
            else:
                new_synth_df = pd.DataFrame()

            self.synth_df = utils.add_new_df_to_df(new_synth_df, self.synth_df)

        try:
            db_size_after = len(self.synth_df)
        except Exception:
            db_size_after = 0

        logger.info(f"Analyzer.get_all_number_of_cells method took {round(time() - start)} seconds")
        logger.info(f"Finished parsing synthesized designs to get number of cells")

        if len(data_list) == 0:
            logger.warning(f"0 designs have been analyzed.")
        else:
            logger.info(f"A total of {len(data_list)} synthesis reports were parsed and added to the database")
        logger.info(f"Synth DF was {db_size_before} | is now {db_size_after}")

    @staticmethod
    def add_input_wires(wire_list: list[str], exp_config: dict, synth_version: int) -> list[str]:
        """Extends (inline) the input wire list of the list of splitted input wires of the design."""

        wire_list_extension = []

        # Build the list of input operand from list of port names
        operands_list = []
        for input_port_name in exp_config["input_ports"]:
            operands_list.append(f"{input_port_name.replace('input', 'operand')}_i")

        # Duplicate all operand by total number of bits
        in_bitwidth = int(exp_config["input_bitwidth"])
        for operand in operands_list:
            for i in range(in_bitwidth):
                input_wire_name = f"{operand}_{str(i).zfill(len(str(in_bitwidth)))}_"
                wire_list_extension.append(input_wire_name)

        wire_list.extend(wire_list_extension)
        return wire_list

    @staticmethod
    def add_output_wires(wire_list: list[str], exp_config: dict, synth_version: int) -> list[str]:
        """Extends (inline) the input wire list of the list of splitted input wires of the design."""

        wire_list_extension = []

        # Build the list of output operand from list of port names
        operands_list = []
        for input_port_name in exp_config["output_ports"]:
            operands_list.append(f"{input_port_name.replace('output', 'result')}_o")

        # Duplicate all operand by total number of bits
        out_bitwidth = int(exp_config["output_bitwidth"])
        out_bitwidth_len = len(str(out_bitwidth))
        for operand in operands_list:
            for i in range(out_bitwidth):
                output_wire_name = f"{operand}_{str(i).zfill(out_bitwidth_len)}_"
                wire_list_extension.append(output_wire_name)

        wire_list.extend(wire_list_extension)
        return wire_list

    @staticmethod
    def reduce_wire_list(wire_list: list[str], wire_type: str, return_index: bool = False) -> list[str]:
        """Take a wire list in input and reduce the wire list based on the sepcified filter type."""
        assert wire_type in ["internal", "io"]

        internal_pattern = re.compile(r"^_[0-9]*_")
        if wire_type == "internal":
            reduced_wire_list = np.array(wire_list)[
                np.vectorize(lambda x: re.match(internal_pattern, x) is not None)(wire_list)
            ]
        elif wire_type == "io":
            reduced_wire_list = np.array(wire_list)[
                np.vectorize(lambda x: re.match(internal_pattern, x) is None)(wire_list)
            ]

        if not return_index:
            return reduced_wire_list
        else:
            wire_indexes = [wire_list.index(wire) for wire in reduced_wire_list]
            return reduced_wire_list, wire_indexes

    @staticmethod
    def _get_fanout_wires_n_depth(
        synthed_design_dir_path: Path,
        exp_config: dict,
        synth_version: int = 0,
        swact_version: int = 0,
        technology_name: str = "notech_yosys",
        cell_cost_mode: str = "transistor",
    ) -> pd.DataFrame:
        """[Worker function] Parses the synthesis designs and extract the fanout for every single wire."""

        try:
            synthed_design_path = synthed_design_dir_path / "mydesign_yosys.v"

            design_number = file_parsers.extract_design_number_from_path(synthed_design_dir_path)

            # Cleanup, just in case ...
            format_synthed_design(synthed_design_path, technology_name)

            # Extract list of wires from synthed design netlist
            wire_list, (in_wire_list, out_wire_list) = get_wire_list(
                synthed_design_path=synthed_design_path, remove_dangling_wires=True
            )

            ### Measure fanout for all wires
            # Prepare cost model according to user configuration
            model_key_map = {
                "transistor": "transistor_count",
                "capacitance": "capacitance",
                "capacitance_calibrated": "capacitance_calibrated",
            }
            if cell_cost_mode == "none":
                cell_cost_model = None
            else:
                try:
                    key = model_key_map[cell_cost_mode]
                    cell_cost_model = GatesConfig.configs[technology_name][key]
                except KeyError as exc:
                    raise ValueError(
                        f"Cell cost model '{cell_cost_mode}' not available for technology '{technology_name}'"
                    ) from exc

            # Analyse fanout and wires
            wire_fanout_dict, wires_graph, cells_graph = get_fanout_wires_n_depth(
                synthed_design_path=synthed_design_path,
                wire_list=wire_list,
                cell_cost_model=cell_cost_model,
                technology_name=technology_name,
                clean_output_pin_names=True,
                fix_zeros=True,
            )

            # Store results in synth folder
            wire_fanout_dict["design_number"] = design_number
            json.dump(wire_fanout_dict, open(synthed_design_dir_path / "wire_fanout.json", "w"))

            # Do some quick analysis to add to synth_df
            internal_wire_list = Analyzer.reduce_wire_list(wire_list=wire_list, wire_type="internal")
            io_wire_list = Analyzer.reduce_wire_list(wire_list=wire_list, wire_type="io")

            fanout_stats_dict = {}
            if len(internal_wire_list) == 0:
                fanout_stats_dict["max_fanout_internal"] = 0
                fanout_stats_dict["min_fanout_internal"] = 0
            else:
                fanout_stats_dict["max_fanout_internal"] = max(np.vectorize(wire_fanout_dict.get)(internal_wire_list))
                fanout_stats_dict["min_fanout_internal"] = min(np.vectorize(wire_fanout_dict.get)(internal_wire_list))

            fanout_stats_dict["max_fanout_io"] = max(np.vectorize(wire_fanout_dict.get)(io_wire_list))
            fanout_stats_dict["min_fanout_io"] = min(np.vectorize(wire_fanout_dict.get)(io_wire_list))
            fanout_stats_dict["design_number"] = design_number

            # Do some quick analysis to get depth of all cells in cells_graph
            wire_depth_dict = utils.find_maximum_depth(wires_graph)
            cell_depth_dict = {}
            max_depth = 0
            for cell_name, cell_output_wires in cells_graph.items():
                if len(cell_output_wires) > 0:
                    cell_depth_dict[cell_name] = max([wire_depth_dict[wire] for wire in cell_output_wires])
                    max_depth = max(max_depth, cell_depth_dict[cell_name])
                else:
                    cell_depth_dict[cell_name] = None

            # Store results in synth folder
            wire_depth_dict["design_number"] = design_number
            json.dump(wire_depth_dict, open(synthed_design_dir_path / "wire_depth_dict.json", "w"))
            cell_depth_dict["design_number"] = design_number
            json.dump(cell_depth_dict, open(synthed_design_dir_path / "cell_depth_dict.json", "w"))

            # Store results for synth_df database
            fanout_stats_dict["max_cell_depth"] = max_depth

            return fanout_stats_dict

        except Exception as e:
            logger.debug(design_number)
            logger.debug(e)
            raise e

    def get_all_fanout_wires_n_depth(self):
        """This function redirects the worker function to the correct synthesis paths."""

        logger.info(f"Parsing all synthesized designs to get wire fanout and maximum depth ...")
        res_list = []
        if self.args_dict.get("rebuild_db", False) or self.is_user_design_number_list:
            desc = f"x{self.nb_workers}|{self.experiment_name} | Get fanout wires"
            with tqdm(total=len(self.synth_designs_dir_paths_list), desc=desc) as pbar:  # Progress bar
                res_list = process_pool_helper(
                    func=Analyzer._get_fanout_wires_n_depth,
                    func_args_gen=(
                        (
                            synthed_design_dir_path,
                            self.exp_config,
                            self.dir_config.synth_ver,
                            self.dir_config.swact_ver,
                            self.dir_config.technology,
                            self.dir_config.cell_cost_model,
                        )
                        for synthed_design_dir_path in self.synth_designs_dir_paths_list
                    ),
                    pbar=pbar,
                )

            if res_list:
                new_synth_df = pd.DataFrame(res_list)
                self.synth_df = utils.add_new_df_to_df(new_synth_df, self.synth_df)

        logger.info(f"Finished parsing {len(res_list)} synthesized designs to get wire fanout and depth")
        logger.info(
            f"All fanout data have been written to their respective design synthesis folder in {self.dir_config.synth_out_dir}"
        )

    def get_all_design_complexities(self):
        """This function redirects the worker function to the correct generation paths."""

        logger.info(f"Parsing all cmplxed designs to get LUT complexities ...")

        res_list = []
        if self.args_dict.get("rebuild_db", False) or self.is_user_design_number_list:
            desc = f"x{self.nb_workers}|{self.experiment_name} | Get LUT complexities"
            with tqdm(total=len(self.cmplx_designs_dir_paths_list), desc=desc) as pbar:  # Progress bar
                res_list = utils.process_pool_helper(
                    func=file_parsers.get_design_complexity,
                    func_args_gen=(
                        (cmplx_design_filepath,) for cmplx_design_filepath in self.cmplx_designs_dir_paths_list
                    ),
                    max_workers=self.nb_workers,
                    pbar=pbar,
                )

            if len(res_list) > 0:
                new_cmplx_df = pd.DataFrame(res_list)
                self.cmplx_df = utils.add_new_df_to_df(new_cmplx_df, self.cmplx_df)

        logger.info(f"Finished parsing {len(res_list)} cmplxed designs to get LUT complexities")

    @staticmethod
    def _get_design_encodings(design_lut_path: Path) -> dict[str, str]:
        """Returns the input and output encoding for the specified design LUT path"""
        encodings_dict = file_parsers.extract_encodings(design_lut_path)
        design_number = file_parsers.extract_design_number_from_path(design_lut_path)
        res_dict = {
            "design_number": design_number,
            "encodings_input": str(encodings_dict["input"]),
            "encodings_output": str(encodings_dict["output"]),
        }
        return res_dict

    def get_all_design_encodings(self):
        """This function redirects the worker function to the correct generation paths."""

        logger.info(f"Parsing all generated designs to get design encodings ...")
        genered_design_filepath_list = [
            (self.dir_config.generation_out_dir / gener_dirpaths.name / "hdl" / "mydesign_comb.v")
            for gener_dirpaths in self.gener_designs_dir_paths_list
        ]

        res_list = []
        if self.args_dict.get("rebuild_db", False) or self.is_user_design_number_list:
            desc = f"x{self.nb_workers}|{self.experiment_name} | Get design encodings dictionnaries"
            with tqdm(total=len(genered_design_filepath_list), desc=desc) as pbar:  # Progress bar
                res_list = utils.process_pool_helper(
                    func=Analyzer._get_design_encodings,
                    func_args_gen=(
                        (genered_design_filepath,) for genered_design_filepath in genered_design_filepath_list
                    ),
                    max_workers=self.nb_workers,
                    pbar=pbar,
                )

            if len(res_list) > 0:
                new_synth_df = pd.DataFrame(res_list)
                self.gener_df = utils.add_new_df_to_df(new_synth_df, self.gener_df)

        logger.info(f"Finished parsing {len(res_list)} generated designs to get encoding dictionnaries")

    @staticmethod
    def _analyze_swact(
        res_dict_template: dict,
        test_type: str,
        swact_count: pd.DataFrame | np.ndarray,
        wire_names: list[str],
        stimuli_count: int,
        suffix: str,
    ) -> dict:
        result_dict = copy(res_dict_template)
        result_dict["test_type"] = test_type
        nb_wires = len(wire_names)
        if isinstance(swact_count, pd.DataFrame):
            result_dict[f"swact{suffix}_total"] = swact_count.sum(axis=1).sum()
            result_dict[f"swact{suffix}_average"] = swact_count.sum(axis=1).mean()
            result_dict[f"per_wire_swact{suffix}_stddev"] = swact_count.sum(axis=0).std()
            result_dict[f"per_wire_min_swact{suffix}"] = swact_count.sum(axis=0).min()
            result_dict[f"per_wire_min_swact{suffix}_wire_name"] = wire_names[swact_count.sum(axis=0).argmin()]
            result_dict[f"per_wire_max_swact{suffix}"] = swact_count.sum(axis=0).max()
            result_dict[f"per_wire_max_swact{suffix}_wire_name"] = wire_names[swact_count.sum(axis=0).argmax()]

        else:
            result_dict[f"swact{suffix}_total"] = np.sum(swact_count)
            result_dict[f"swact{suffix}_average"] = result_dict[f"swact{suffix}_total"] / stimuli_count

            # suffix = "_weighted" if weighted else ""
            result_dict[f"per_wire_swact{suffix}_average"] = result_dict[f"swact{suffix}_average"] / nb_wires
            result_dict[f"per_wire_swact{suffix}_stddev"] = np.std(swact_count)
            result_dict[f"per_wire_min_swact{suffix}"] = np.min(swact_count)
            result_dict[f"per_wire_min_swact{suffix}_wire_name"] = wire_names[np.argmin(swact_count)]
            result_dict[f"per_wire_max_swact{suffix}"] = np.max(swact_count)
            result_dict[f"per_wire_max_swact{suffix}_wire_name"] = wire_names[np.argmax(swact_count)]

        return result_dict

    @staticmethod
    def clean_output_pin_names(text: str) -> str:
        """Remove the unsafe patterns from the output pin names"""
        return file_parsers._clean_output_pin_names(text)

    @staticmethod
    def clean_output_pin_names_in_db(db_path: Path, force: bool = False) -> None:
        """Open the db_path and change the names with bad escape sequences"""
        done_token_path = db_path.parent / "db_cleaned.token"
        if not done_token_path.exists():
            try:
                text = db_path.read_text()
                text = Analyzer.clean_output_pin_names(text)
                db_path.write_text(text)
            except UnicodeDecodeError:
                db = pd.read_parquet(db_path)
                db = db.rename(file_parsers._clean_output_pin_names, axis="columns")
                db.to_parquet(db_path)

        else:
            pass

        return None

    @staticmethod
    def apply_weights_to_switch_vectors(db_switch_vectors: pd.DataFrame, wire_fanout_dict: dict[str:float]):
        """Apply wire weights to switching vectors using :mod:`swact`.

        This wrapper delegates the heavy lifting to
        :func:`swact.apply_weights_to_switch_vectors` while keeping the
        original return type for backward compatibility.
        """

        weighted, missing = swact_apply_weights_to_switch_vectors(
            db_switch_vectors, wire_fanout_dict, return_missing=True
        )
        return weighted, missing

    @staticmethod
    def apply_weights_to_switch_vectors_legacy(db_switch_vectors: pd.DataFrame, wire_fanout_dict: dict[str:float]):
        """Legacy implementation retained for regression tests."""
        missing_wires = []
        for wire_name in wire_fanout_dict.keys():
            if (
                wire_name != "design_number"
                and "result_o" not in wire_name
                and wire_name
                and "clk" not in wire_name
                and "rst" not in wire_name
            ):
                try:
                    db_switch_vectors[wire_name] = db_switch_vectors[wire_name] * float(wire_fanout_dict[wire_name])
                except KeyError:
                    missing_wires.append(wire_name)

        return db_switch_vectors, missing_wires

    @staticmethod
    def _make_design_swact_analysis_outdir(analysis_out_dir: Path, design_number: str) -> Path:
        """Make the analysis output directory for the specified design
        Used for worst case swact, histograms, ...
        """
        out_dir_path = analysis_out_dir / "swact_analysis" / f"res_{design_number}"
        out_dir_path.mkdir(parents=True, exist_ok=True)

        return out_dir_path

    @staticmethod
    def _find_encoder_with_same_encoding(
        encodings_dict: dict[int, str], all_encoder_encoding_dicts: dict[str, dict[int, str]] | None
    ) -> None:
        """Explore the list of encoder encoding dictionnaries and return the encoder number that has the same encoding."""
        for encoder_number, encoder_encodings_dict in all_encoder_encoding_dicts.items():
            if encoder_encodings_dict == encodings_dict["input"]:
                return encoder_number
        return None

    @staticmethod
    def _correlate_swact_with_encoding_cost(
        all_switching_vectors_df: pd.DataFrame,
        isolated_representation_db: pd.DataFrame,
        all_wire_names: list[str],
        encoder_number: str,
        encoder_out_dir: Path,
        design_number: str,
        curr_design_swact_outdir: Path,
        input_ports: list[str],
        experiment_name: str,
        test_type_name: str,
        is_weighted: bool = True,
    ) -> None:
        """This function evaluates the cost of switching activity with added cost of encoding."""

        # Step 1 open the swact total for current design
        all_swact_design_df = Analyzer.evaluate_swact_counts(
            all_switching_vectors_df, isolated_representation_db, all_wire_names
        )

        # Step 2.1 open the swact total cost of the encoder
        all_swact_encoder_filepath = (
            encoder_out_dir
            / "analysis_out/swact_analysis"
            / f"res_{encoder_number}"
            / f"all_swact_overview_weighted_dn{encoder_number}_db.pqt"
        )
        try:
            all_swact_encoder_df = utils.load_database(all_swact_encoder_filepath)
        except UnicodeDecodeError:
            all_swact_encoder_df = pd.read_parquet(all_swact_encoder_filepath)
        all_swact_encoder_df_before = all_swact_encoder_df.shift(1)
        # Step 2.2 create the switch pattern dataframe
        encoder_switch_cost_df = pd.DataFrame()
        encoder_switch_cost_df["encoder_switch_pattern_output"] = (
            all_swact_encoder_df_before["output_rep"] + all_swact_encoder_df["output_rep"]
        )  # Create a series where each value corresponds to `str_before + str_after`
        encoder_switch_cost_df["encoder_io_swact_cost"] = all_swact_encoder_df["io_swact_count"].astype(
            float
        )  # Alignement has been checked, you can trust it
        encoder_switch_cost_df["encoder_internal_swact_cost"] = all_swact_encoder_df["internal_swact_count"].astype(
            float
        )
        encoder_switch_cost_df["encoder_total_swact_cost"] = all_swact_encoder_df["total_swact_count"].astype(float)
        # Step 2.3 add all no switch patterns
        all_switch_df_list = [
            encoder_switch_cost_df,
        ]
        for rep in all_swact_encoder_df["output_rep"].unique():
            no_switch_dict = {
                "encoder_switch_pattern_output": rep + rep,
                "encoder_io_swact_cost": 0.0,
                "encoder_internal_swact_cost": 0.0,
                "encoder_total_swact_cost": 0.0,
            }
            all_switch_df_list.append(pd.DataFrame([no_switch_dict]))

        encoder_switch_cost_df = pd.concat(all_switch_df_list, ignore_index=True)
        # Drop duplicatesotherwise the merge will add some rows in the merged table as all valid encoder_switch_pattern_output will be used during the merge
        encoder_switch_cost_df = encoder_switch_cost_df.drop_duplicates(subset="encoder_switch_pattern_output")

        # Step 3.1 Build the combined cost database
        all_swact_design_df_before = all_swact_design_df.shift(1)
        combined_switch_cost_df = pd.DataFrame()
        # Add design counts
        combined_switch_cost_df["design_io_swact_cost"] = all_swact_design_df["io_swact_count"].astype(float)
        combined_switch_cost_df["design_internal_swact_cost"] = all_swact_design_df["internal_swact_count"].astype(
            float
        )
        combined_switch_cost_df["design_total_swact_cost"] = all_swact_design_df["total_swact_count"].astype(float)
        # Step 3.2 add encoding costs for all input ports
        for input_port_name in input_ports:
            combined_switch_cost_df[f"design_switch_pattern_{input_port_name}"] = (
                all_swact_design_df_before[f"{input_port_name}_rep"] + all_swact_design_df[f"{input_port_name}_rep"]
            )

            # Perform the left join
            combined_switch_cost_df = pd.merge(
                left=combined_switch_cost_df,
                right=encoder_switch_cost_df,
                left_on=f"design_switch_pattern_{input_port_name}",
                right_on="encoder_switch_pattern_output",
                how="left",
            )

            # Rename the 'total_swact_cost' column to 'total_encoder_swact_cost'
            combined_switch_cost_df = combined_switch_cost_df.rename(
                columns={
                    "encoder_io_swact_cost": f"encoder_io_swact_cost_{input_port_name}",
                    "encoder_internal_swact_cost": f"encoder_internal_swact_cost_{input_port_name}",
                    "encoder_total_swact_cost": f"encoder_total_swact_cost_{input_port_name}",
                }
            )

            # Drop the 'switch_pattern' column (optional, if you don't need it anymore)
            combined_switch_cost_df = combined_switch_cost_df.drop(columns=["encoder_switch_pattern_output"])

        # Step 3 sum them (with several encoder reuse coefficient)
        for reuse in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            combined_switch_cost_df[f"dne_r{reuse}_io_swact_cost"] = combined_switch_cost_df["design_io_swact_cost"]
            combined_switch_cost_df[f"dne_r{reuse}_internal_swact_cost"] = combined_switch_cost_df[
                "design_internal_swact_cost"
            ]
            combined_switch_cost_df[f"dne_r{reuse}_total_swact_cost"] = combined_switch_cost_df[
                "design_total_swact_cost"
            ]
            for input_port_name in input_ports:
                combined_switch_cost_df[f"dne_r{reuse}_io_swact_cost"] += (
                    combined_switch_cost_df[f"encoder_io_swact_cost_{input_port_name}"] / reuse
                )
                combined_switch_cost_df[f"dne_r{reuse}_internal_swact_cost"] += (
                    combined_switch_cost_df[f"encoder_internal_swact_cost_{input_port_name}"] / reuse
                )
                combined_switch_cost_df[f"dne_r{reuse}_total_swact_cost"] += (
                    combined_switch_cost_df[f"encoder_total_swact_cost_{input_port_name}"] / reuse
                )

        # Step 4 save the data for plotting? - 1 plot per design - with multiple duplicate coefficient
        filepath = curr_design_swact_outdir / f"combined_switch_cost_{test_type_name}_db.pqt"
        plotter.plot_swact_cost_vs_encoder_design_coefficient(
            combined_switch_cost_df,
            curr_design_swact_outdir,
            design_number,
            encoder_number,
            experiment_name=experiment_name,
            test_type_name=test_type_name,
            is_weighted=is_weighted,
        )
        Analyzer._save_db_str(combined_switch_cost_df, filepath=filepath, title_name=None)

        return None

    @staticmethod
    def _get_switching_activity(
        dir_config: ConfigDir,
        tested_design_dir_path: Path,
        experiment_config: dict[str:str],
        analyze_fullsweep: bool,
        all_encoder_encoding_dicts: dict[str, dict[int, str]] | None,
        encoder_out_dir: Path,
        is_user_design_number_list: bool,
    ) -> pd.DataFrame:
        """This function asserts correctness of the synthesized design and returns the switching activity of a design."""
        try:
            # Setup design info
            design_number = file_parsers.extract_design_number_from_path(tested_design_dir_path)
            curr_design_swact_outdir = Analyzer._make_design_swact_analysis_outdir(
                dir_config.analysis_out_dir, design_number
            )
            design_lut_path = dir_config.generation_out_dir / f"res_{design_number}/hdl/mydesign_comb.v"

            # try:
            res_dict_template = {
                "design_number": design_number,
                "test_type": None,
                "nb_wires": None,
            }

            # Open test databases
            test_databases = []
            db_found = False
            idx = 0
            test_db_names_dict = {}
            for filename in os.listdir(tested_design_dir_path):
                # Check Whether the filename is ok
                filename_ok = False
                for ending_str in ["_db.csv", "_db.pqt", "_db.parquet"]:
                    filename_ok = filename_ok or ending_str in filename

                if filename_ok:
                    _test_type = filename.replace("results_", "").split(".")[0].replace("_db", "")
                    if _test_type == "db":
                        _test_type = "undefined"
                        # We can expect results of a flowy run
                        if dir_config.exp_config.get("synth_with_flowy", False):
                            # Get the data record done by flowy
                            data_record_filepath = tested_design_dir_path / "swact_data_record.json"
                            if data_record_filepath.exists():
                                json_data_dict = json.loads(data_record_filepath.read_text())["results_dict"]["data"]
                                json_data_dict.update(
                                    {
                                        "test_type": _test_type,
                                        "design_number": design_number,
                                    }
                                )
                                result_df = pd.DataFrame([json_data_dict])
                                # Rename the swact count column
                                result_df.rename(
                                    columns={
                                        "swact_count": "swact_average",
                                        "swact_count_weighted": "swact_weighted_average",
                                        "n_wires": "nb_wires",
                                    },
                                    inplace=True,
                                )
                                result_dict = {
                                    "row": result_df,
                                    "status": "success",
                                    "design_path": design_lut_path,
                                    "test_types_nb": len(test_databases),
                                }
                                return result_dict

                    if _test_type in experiment_config["do_tests_list"] or _test_type == "undefined":
                        db_path = tested_design_dir_path / filename
                        Analyzer.clean_output_pin_names_in_db(db_path)
                        try:
                            test_database = utils.load_database(db_path)
                        except UnicodeDecodeError:
                            test_database = pd.read_parquet(db_path)
                        if not len(test_database) == 0:
                            test_databases.append(test_database)
                            db_found = True
                            test_db_names_dict[idx] = _test_type
                        idx += 1

            # Assert that a database exists
            if not db_found:
                # __analyzer.incomplete_swact.append(tested_design_dir_path)
                # logger.info(f"Design {design_number} has not been tested (result databases not found)")
                return {"row": None, "status": "incomplete", "design_path": tested_design_dir_path, "test_types_nb": 0}

            # Check correctness of design
            for idx, test_db in enumerate(test_databases):
                test_type = test_db_names_dict[idx]

                try:
                    if not Analyzer.assert_exactness(
                        design_lut_path,
                        test_db,
                        experiment_config,
                        design_swact_oudir_path=curr_design_swact_outdir,
                        save_histogram=False,
                        test_type_name=test_type,
                    ):
                        # __analyzer.failed_swact_analysis.append((design_lut_path,"NOT_EXACT_SYNTH"))
                        logger.info(f"Design {design_number} has failed testing for test {test_type} (model not exact)")
                        return {"row": None, "status": "failed", "design_path": design_lut_path, "test_types_nb": 0}
                except Exception as e:
                    logger.error(e)
                    logger.error(
                        f"Design number {design_number} generated an error during assertion of exactness for test {test_type}. Please analyze."
                    )
                    logger.error(
                        f"Hint: there might be an issue with the encoding description in the generated design."
                    )
                    if design_number in [
                        "99999",
                    ]:
                        pass
                    else:
                        raise e

            # Get encodings and add them to output dictionary
            encodings_dict = file_parsers.extract_encodings(design_lut_path)
            res_dict_template["encodings_input"] = str(encodings_dict["input"])
            res_dict_template["encodings_output"] = str(encodings_dict["output"])

            # Split inputs into wire and isolate full db from representation and values db
            test_databases, isolated_representations_databases = Analyzer.reformat_result_databases(
                test_databases, experiment_config
            )
            # At this stage, test_databases contain only wire data (no more values nor full representations columns)

            # Get number of wires and add them to output dictionary
            wire_only_names = []
            notawire_names = isolated_representations_databases[0].columns.to_list()
            for wire in test_databases[0].columns.to_list():
                if wire not in notawire_names:
                    wire_only_names.append(wire)
                else:
                    pass

            nb_wires = len(wire_only_names)
            res_dict_template["nb_wires"] = nb_wires

            # Load Wire Fanout
            wire_fanout_dict = json.loads(
                (dir_config.synth_out_dir / f"res_{design_number}/wire_fanout.json").read_bytes()
            )
            assert all(wire_fanout_dict.values())

            # Find correlated encoder number
            if all_encoder_encoding_dicts is not None:
                encoder_number = Analyzer._find_encoder_with_same_encoding(encodings_dict, all_encoder_encoding_dicts)
            else:
                encoder_number = None

            # Do some analysis
            result_df = pd.DataFrame()
            for idx, (db, isolated_rep_db) in enumerate(zip(test_databases, isolated_representations_databases)):
                test_type = test_db_names_dict[idx]
                try:
                    db_switch_vectors, db_switch_vectors_count, wire_names = Analyzer.get_switching_vectors(
                        db, save_dbs=("fullsweep" in test_type)
                    )
                except Exception as e:
                    logger.error(f"design number{design_number}")
                    logger.error(e)

                # Store Non Weighted Total Switching Activity Values
                swact_counts_dict = db_switch_vectors_count.to_dict()
                json.dump(swact_counts_dict, open(tested_design_dir_path / "wire_swact_count.json", "w"))

                result_dict = Analyzer._analyze_swact(
                    res_dict_template,
                    test_type,
                    db_switch_vectors[wire_names],
                    wire_names,
                    stimuli_count=len(db_switch_vectors),
                    suffix="",
                )

                # Cleanup wire_fanout_dict wire names
                _wire_fanout_dict = dict()
                for key, value in wire_fanout_dict.items():
                    if "[" in key:
                        _key = key.replace("[", "_").replace("]", "_")
                    else:
                        _key = key
                    _wire_fanout_dict[_key] = int(value)
                wire_fanout_dict = _wire_fanout_dict

                db_switch_vectors_weighted, missing_wires = Analyzer.apply_weights_to_switch_vectors(
                    db_switch_vectors, wire_fanout_dict
                )
                # db_switch_vectors_count_weighted, _ = Analyzer.apply_weights_to_switch_vectors(
                #     db_switch_vectors_count, wire_fanout_dict
                # )
                # logger.info(db_switch_vectors_count_weighted)
                # print(db_switch_vectors_count_weighted)
                # db_switch_vectors_count_dict_weighted = db_switch_vectors_count_weighted.to_dict()
                # db_switch_vectors_dict_weighted = db_switch_vectors_count_weighted.to_dict()

                if len(missing_wires) > 0:
                    json.dump(missing_wires, open(tested_design_dir_path / "missing_wires_in_simulation.json", "w"))

                # Get Weighted Switching Activity Values
                # db_switch_vectors_count_dict_weighted = dict()
                # for wire_name in wire_only_names:
                #     swact = swact_counts_dict[wire_name]
                #     wire_fanout = wire_fanout_dict[wire_name]
                #     db_switch_vectors_count_dict_weighted[wire_name] = swact*wire_fanout
                # json.dump(swact_counts_dict, open(tested_design_dir_path/"wire_swact_count_weighted.json","w"))

                # Extract the patterns that generate the most switching activity
                if "fullsweep" in test_type and analyze_fullsweep:
                    swact_pattern_analysis_results = Analyzer.find_maximum_switching_pattern(
                        design_number,
                        curr_design_swact_outdir,
                        db_switch_vectors_weighted,
                        isolated_rep_db,
                        wire_names,
                        dir_config.experiment_name,
                        is_weighted=True,
                        do_plots=False,
                    )
                    result_dict.update(swact_pattern_analysis_results)
                    result_dict["nb_zero_internal_swact"] = Analyzer.find_number_of_zero_internal_swact(
                        db_switch_vectors_weighted, wire_names
                    )
                else:
                    # Required otherwise the format result function will throw an error
                    result_dict["max_io_swact"] = 0
                    result_dict["max_io_swact_count"] = 0
                    result_dict["max_internal_swact"] = 0
                    result_dict["max_internal_swact_count"] = 0
                    result_dict["nb_zero_internal_swact"] = 0

                # Find encoder with same encoding and correlate swact with encoding cost
                # TODO: Put this in another multiprocessing loop?
                if encoder_number is not None:
                    Analyzer._correlate_swact_with_encoding_cost(
                        db_switch_vectors_weighted,
                        isolated_rep_db,
                        wire_names,
                        encoder_number,
                        encoder_out_dir,
                        design_number,
                        curr_design_swact_outdir,
                        experiment_config["input_ports"],
                        dir_config.experiment_name,
                        test_type_name=test_type,
                        is_weighted=True,
                    )

                # Store Total Weighted Switching Activity Values
                # swact_count_weighted = np.vectorize(db_switch_vectors_dict_weighted.get)(wire_names)
                # swact_count_weighted = db_switch_vectors_dict_weighted[wire_names]
                result_dict = Analyzer._analyze_swact(
                    result_dict,
                    test_type,
                    db_switch_vectors_weighted[wire_names],
                    wire_names,
                    stimuli_count=len(db_switch_vectors_weighted),
                    suffix="_weighted",
                )

                # Store Internal Weighted Switching Activity Values
                intern_wire_list = Analyzer.reduce_wire_list(wire_list=wire_names, wire_type="internal")
                if len(intern_wire_list) == 0:
                    # It can happen that there is not internal witching activity
                    intern_wire_list = ["none"]
                result_dict = Analyzer._analyze_swact(
                    result_dict,
                    test_type,
                    db_switch_vectors_weighted[intern_wire_list],
                    intern_wire_list,
                    stimuli_count=len(db_switch_vectors_weighted),
                    suffix="_internal_weighted",
                )

                # Store IO Weighted Switching Activity Values
                io_wire_list = Analyzer.reduce_wire_list(wire_list=wire_names, wire_type="io")
                result_dict = Analyzer._analyze_swact(
                    result_dict,
                    test_type,
                    db_switch_vectors_weighted[io_wire_list],
                    io_wire_list,
                    stimuli_count=len(db_switch_vectors_weighted),
                    suffix="_io_weighted",
                )

                new_row_df = pd.DataFrame([result_dict])
                result_df = pd.concat([result_df, new_row_df], ignore_index=True)

            result_dict = {
                "row": result_df,
                "status": "success",
                "design_path": design_lut_path,
                "test_types_nb": len(test_databases),
            }

        except Exception:
            logger.error(f"There was an errror extracting witching activity for des {design_number}")
            logger.error(traceback.format_exc())

            result_dict = {
                "row": None,
                "status": "failed",
                "design_path": design_lut_path,
                "test_types_nb": len(test_databases),
            }

        # Append current design path to list of successful switching activity analysis
        # Analyzer.successful_swact_analysis.append(design_lut_path)
        return result_dict

    @staticmethod
    def find_number_of_zero_internal_swact(all_switching_vectors_df: pd.DataFrame, all_wire_names: list[str]):
        """ " This function takes the full database of switching vectors and returns the number of input patterns that generated zero internal switching activity."""

        intern_wire_list, internal_wire_indexes = Analyzer.reduce_wire_list(
            wire_list=all_wire_names, wire_type="internal", return_index=True
        )
        if len(intern_wire_list) == 0:
            return float("infinity")
        else:
            internal_swact_counts = all_switching_vectors_df[intern_wire_list].sum(axis=1)
            zero_args = Analyzer.extract_internal_zero_args_from_swact_counts(internal_swact_counts)
            return len(zero_args)

    @staticmethod
    def extract_internal_zero_args_from_swact_counts(internal_swact_counts_df: pd.DataFrame) -> np.ndarray:
        """Return the indexes at which the swact row is zero. The user should pass the internal switching swact counts as input argument."""

        # Extract zeros
        zero_args = np.argwhere(internal_swact_counts_df == 0)[:, 0]

        # Remove zeros coming from first and last line of test
        zero_args = np.delete(zero_args, np.argwhere(zero_args == 0)[:, 0])
        zero_args = np.delete(zero_args, np.argwhere(zero_args == (internal_swact_counts_df.shape[0] - 1))[:, 0])

        return zero_args

    @staticmethod
    def evaluate_swact_counts(
        all_switching_vectors_df: pd.DataFrame, isolated_representation_db: pd.DataFrame, all_wire_names: list[str]
    ) -> pd.DataFrame:
        # Get IO SwAct
        io_wire_list = Analyzer.reduce_wire_list(wire_list=all_wire_names, wire_type="io", return_index=False)
        io_all_switchvec = all_switching_vectors_df[io_wire_list]
        io_swact_counts = io_all_switchvec.sum(axis=1)

        # Get Internal SwAct
        intern_wire_list = Analyzer.reduce_wire_list(wire_list=all_wire_names, wire_type="internal", return_index=False)
        internal_all_switchvec = all_switching_vectors_df[intern_wire_list]
        internal_swact_counts = internal_all_switchvec.sum(axis=1)
        # At this stage, the row 0 corresponds to nothing, so there is now switching activity there

        ### Associate counts with binary representations
        # Store IO and Internal SwAct Counts along with binary representations and values of input output values
        # Note: the swact count displayed at row i corresponds to the switch from row i to row i+1
        isolated_representation_db["index"] = isolated_representation_db.index
        all_swact_overview_df = copy(isolated_representation_db)
        all_swact_overview_df["io_swact_count"] = io_swact_counts
        all_swact_overview_df["internal_swact_count"] = internal_swact_counts

        # Store total SwAct count as well
        all_swact_overview_df = all_swact_overview_df.astype({"internal_swact_count": int, "io_swact_count": int})
        all_swact_overview_df["total_swact_count"] = (
            all_swact_overview_df["io_swact_count"] + all_swact_overview_df["internal_swact_count"]
        )

        return all_swact_overview_df

    @staticmethod
    def find_maximum_switching_pattern(
        design_number: str,
        design_swact_oudir_path: Path,
        all_switching_vectors_df: pd.DataFrame,
        isolated_representation_db: pd.DataFrame,
        all_wire_names: list[str],
        experiment_name: str,
        is_weighted: bool,
        do_plots: bool = False,
    ) -> dict[str, float]:
        """
        Find the input value movements that generate the most switching activity
        Note: row i of all_switching_vectors array corresponds to switch from row i-1 to row i
        """

        # Prepare output dictionary
        results = dict()
        no_internal_wires = (
            len(Analyzer.reduce_wire_list(wire_list=all_wire_names, wire_type="internal", return_index=True)) == 0
        )

        ### Evaluate SwAct counts (total, per row)
        all_swact_overview_df = Analyzer.evaluate_swact_counts(
            all_switching_vectors_df, isolated_representation_db, all_wire_names
        )

        # Save All SwAct Count table
        weighted_suffix = "_weighted" if is_weighted else ""
        filepath = design_swact_oudir_path / f"all_swact_overview{weighted_suffix}_dn{design_number}_db.pqt"
        Analyzer._save_db_str(
            all_swact_overview_df, filepath=filepath, title_name=None
        )  # title_name="All SwAct Overview")

        if do_plots:
            plotter.plot_distribution_of_swact(
                all_swact_overview_df["io_swact_count"], swact_type="io", out_dir_path=design_swact_oudir_path
            )
            plotter.plot_distribution_of_swact(
                all_swact_overview_df["internal_swact_count"],
                swact_type="internal",
                out_dir_path=design_swact_oudir_path,
            )
            plotter.plot_internal_versus_io_swact_count(
                all_swact_overview_df,
                design_number=design_number,
                experiment_name=experiment_name,
                out_dir_path=design_swact_oudir_path,
            )

        ### Extract notable SwAct row indexes
        # Find rows with max SwAct for IO wires
        io_notable_args = np.argwhere(
            all_swact_overview_df["io_swact_count"] == np.amax(all_swact_overview_df["io_swact_count"])
        )[:, 0]
        # Remove extreme indices which actually have no real signification
        io_notable_args = io_notable_args[io_notable_args != 0]
        io_notable_args = io_notable_args[io_notable_args != (len(all_swact_overview_df) - 1)]
        # Extract and save maximum value
        max_io_swact, max_io_swact_count = np.unique(
            all_swact_overview_df["io_swact_count"].to_numpy()[io_notable_args], return_counts=True
        )
        results["max_io_swact"] = max_io_swact[0]
        results["max_io_swact_count"] = max_io_swact_count[0]

        # Find rows with max SwAct for internal wires
        internal_notable_args = np.argwhere(
            all_swact_overview_df["internal_swact_count"] == np.amax(all_swact_overview_df["internal_swact_count"])
        )[:, 0]
        # Remove extreme indices which actually have no real signification
        internal_notable_args = internal_notable_args[internal_notable_args != 0]
        internal_notable_args = internal_notable_args[internal_notable_args != (len(all_swact_overview_df) - 1)]
        # Extract and save maximum value
        max_internal_swact, max_internal_swact_count = np.unique(
            all_swact_overview_df["internal_swact_count"].to_numpy()[internal_notable_args], return_counts=True
        )
        results["max_internal_swact"] = max_internal_swact[0]
        results["max_internal_swact_count"] = max_internal_swact_count[0]

        # Find rows with zeros SwAct for internal wires only (no IO as there should not be zero IO)
        if no_internal_wires:
            # There should not be any notable internal SwAct row as there is not internal wires
            internal_notable_args = []
        else:
            # Extend the list of internal notable row indexes
            zero_args = Analyzer.extract_internal_zero_args_from_swact_counts(
                all_swact_overview_df["internal_swact_count"]
            )
            internal_notable_args = np.concatenate([internal_notable_args, zero_args])

        ### Create the DBs that illustrates all min and max SwAct patterns (IO and Internal)
        # Get initial representation pattern
        io_notable_switch_patterns_init = isolated_representation_db.loc[io_notable_args]
        # Get next representation pattern
        io_notable_switch_patterns_end = isolated_representation_db.loc[io_notable_args + 1]
        # Merge: Reduce index of next by 1 to align the databases for merging
        io_notable_switch_patterns_end["index"] = io_notable_switch_patterns_end["index"].apply(lambda x: x - 1)
        io_notable_switch_patterns = pd.merge(
            io_notable_switch_patterns_init,
            io_notable_switch_patterns_end,
            on="index",
            suffixes=["_io_init", "_io_end"],
        )
        # Remove useless index column
        io_notable_switch_patterns.drop("index", axis="columns", inplace=True)
        # Get values of notable indexes
        io_notable_switch_patterns["io_swact_count"] = all_swact_overview_df["io_swact_count"].to_numpy()[
            io_notable_args
        ]
        # Save max switching sums
        filepath = design_swact_oudir_path / f"notable_swact_patterns_io_dn{design_number}_db.pqt"
        Analyzer._save_db_str(
            io_notable_switch_patterns, filepath=filepath, title_name=None
        )  # title_name="IO Notable SwAct Patterns Summary")

        # Do the same for internal wires
        filepath = design_swact_oudir_path / f"notable_swact_patterns_internal_dn{design_number}_db.pqt"
        if no_internal_wires:
            filepath.write_text("No internal wires in the design.\nDB is empty.")
        else:
            internal_notable_switch_patterns_init = isolated_representation_db.loc[internal_notable_args]
            internal_notable_switch_patterns_end = isolated_representation_db.loc[internal_notable_args + 1]
            internal_notable_switch_patterns_end["index"] = internal_notable_switch_patterns_end["index"].apply(
                lambda x: x - 1
            )
            internal_notable_switch_patterns = pd.merge(
                internal_notable_switch_patterns_init,
                internal_notable_switch_patterns_end,
                on="index",
                suffixes=["_internal_init", "_internal_end"],
            )
            internal_notable_switch_patterns.drop("index", axis="columns", inplace=True)
            internal_notable_switch_patterns["internal_swact_count"] = all_swact_overview_df[
                "internal_swact_count"
            ].to_numpy()[internal_notable_args]
            Analyzer._save_db_str(
                internal_notable_switch_patterns, filepath=filepath, title_name=None
            )  # title_name="Internal Notable SwAct Patterns Summary")

        return results

    def get_all_switching_activity(self):
        """This function redirects the worker function to the correct synthesis paths."""

        # Check if encoders are available
        # TODO: Continue this
        if self.dir_config.encoder_out_dir is not None:
            all_encoder_encoding_dicts = file_parsers.read_all_existing_encoding(
                self.dir_config.encoder_out_dir, self.dir_config.root_output_dir, type="analyzer"
            )
        else:
            all_encoder_encoding_dicts = None

        logger.info("Getting all switching activity")
        max_workers = max(self.nb_workers // 2, 1)
        if self.args_dict.get("rebuild_db", False) or self.is_user_design_number_list:
            desc = f"x{max_workers}|{self.experiment_name} | Parsing switching activity results"
            with tqdm(total=len(self.swact_designs_dir_paths_list), desc=desc) as pbar:
                all_results_list = utils.process_pool_helper(
                    func=Analyzer._get_switching_activity,
                    func_args_gen=(
                        (
                            self.dir_config,
                            test_design_path,
                            self.exp_config,
                            not self.skip_fullsweep_analysis,
                            all_encoder_encoding_dicts,
                            self.dir_config.encoder_out_dir,
                            self.is_user_design_number_list,
                        )
                        for test_design_path in self.swact_designs_dir_paths_list
                    ),
                    max_workers=max_workers,
                    pbar=pbar,
                )

            # Analyze the list of results
            nb_tests_types = []
            results_rows = []

            logger.info(f"Consolidating results ...")
            if len(all_results_list) > 0:
                for result in all_results_list:
                    if result["status"] == "success":
                        self.successful_swact_analysis.append(result["design_path"])
                        results_rows.append(copy(result["row"]))
                        # results_rows_weighted.append(copy(result["row_weighted"]))
                    elif result["status"] == "failed":
                        self.failed_swact_analysis.append(result["design_path"])
                    elif result["status"] == "incomplete":
                        self.incomplete_swact.append(result["design_path"])

                    if result["test_types_nb"] != 0:
                        nb_tests_types.append(result["test_types_nb"])

                new_swact_df = pd.concat(results_rows, ignore_index=True)

                if self.is_user_design_number_list:  # We are analyzing specified design numbers
                    self.swact_df = pd.concat([self.swact_df, new_swact_df], ignore_index=True)
                else:
                    # Concatenate all results into the test database
                    self.swact_df = new_swact_df

                # Remove test types that are not present for all tests
                if len(np.unique(nb_tests_types)) != 1:
                    # Count the number of design that have the test_type for each test type
                    counts = self.swact_df.test_type.value_counts()
                    # Remove tests that had issue with some tests
                    # This is useful when some design were not tested on specific tests
                    incomplete_test_type_names = counts[counts != len(all_results_list)]
                    for incomplete_test_type_name in list(incomplete_test_type_names.keys()):
                        self.swact_df = self.swact_df.drop(
                            self.swact_df[self.swact_df["test_type"] == incomplete_test_type_name].index
                        )
                    logger.warning(f"Removed incomplete tests: {list(incomplete_test_type_names.keys())}")
                self.test_type_names = list(self.swact_df["test_type"].unique())
                self.test_type_nb = len(self.test_type_names)

                logger.info("Finished getting all switching activity")
                total_nb_designs = (
                    len(self.successful_swact_analysis) + len(self.failed_swact_analysis) + len(self.incomplete_swact)
                )
                logger.info("-------------------------------------")
                success_percent = len(self.successful_swact_analysis) / total_nb_designs * 100
                logger.info(
                    f"Number of successfully analyzed designs: {len(self.successful_swact_analysis)} ({success_percent:.1f}%)"
                )
                incomplete_percent = len(self.incomplete_swact) / total_nb_designs * 100
                logger.info(
                    f"Number of not yet tested designs: {len(self.incomplete_swact)} ({incomplete_percent:.1f}%)"
                )
                failed_percent = len(self.failed_swact_analysis) / total_nb_designs * 100
                logger.info(
                    f"Number of failed tested or analyzed designs: {len(self.failed_swact_analysis)} ({failed_percent:.1f}%)"
                )

                failed_designs_log = self.dir_config.analysis_out_dir / "failed_designs.txt"
                with open(failed_designs_log, "w") as f:
                    f.writelines(self.failed_swact_analysis)
                if len(self.failed_swact_analysis) > 0:
                    logger.warning(f"All failed designs path have been written to {failed_designs_log}")

            else:
                logger.info(f"No resutls were obtained, noting to be done.")

    def get_all_power(self):
        """This function redirects the worker function to the correct synthesis paths."""

        logger.info(f"Parsing all power reports to get power data ... ")
        try:
            db_size_before = len(self.power_df)
        except Exception:
            db_size_before = 0

        start = time()
        data_list = []
        if self.args_dict.get("rebuild_db", False) or self.is_user_design_number_list:
            desc = f"x{self.nb_workers}| {self.experiment_name} | Parse power reports"
            with tqdm(total=len(self.power_designs_dir_paths_list), desc=desc) as pbar:  # Progress bar
                data_list = utils.process_pool_helper(
                    func=file_parsers.parse_power_n_delay_reports,
                    func_args_gen=((design_dir_path,) for design_dir_path in self.power_designs_dir_paths_list),
                    pbar=pbar,
                )

            if data_list:
                new_power_df = pd.DataFrame(data_list)
            else:
                new_power_df = pd.DataFrame()

            if self.is_user_design_number_list:  # We are analyzing specified design numbers
                self.power_df = pd.concat([self.power_df, new_power_df], ignore_index=True)
            else:
                self.power_df = new_power_df

        try:
            db_size_after = len(self.power_df)
        except Exception:
            db_size_after = 0

        logger.info(f"Analyzer.get_all_power method took {round(time() - start)} seconds")
        logger.info(f"Finished parsing power reports to get number of cells")

        if len(data_list) == 0:
            logger.warning(f"0 designs have been analyzed.")
        else:
            logger.info(f"A total of {len(data_list)} power reports were parsed and added to the database")
        logger.info(f"Power DF was {db_size_before} | is now {db_size_after}")

    @staticmethod
    def get_switching_vectors(db: pd.DataFrame, save_dbs: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-wire switching vectors using :mod:`swact`.

        The original implementation is retained as
        :func:`get_switching_vectors_legacy` for regression tests.
        """

        switch_df, _, wire_names = get_swact_table(db, {}, return_wire_names=True)
        return switch_df, switch_df.sum(), wire_names

    @staticmethod
    def get_switching_vectors_legacy(
        db: pd.DataFrame, save_dbs: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Legacy implementation of :func:`get_switching_vectors`."""

        db.dropna(axis=0, inplace=True)
        db_int = db.astype(int)

        db_shift = db_int.shift(1)
        db_diff = (db_int != db_shift).astype(int)
        db_diff.iloc[0, :] = 0

        return db_diff, db_diff.sum(), list(db_diff.columns)

    @staticmethod
    def reformat_result_databases(
        databases: list[pd.DataFrame], experiment_config: dict, set_reset_at_zero: bool = False
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """Delegate operand splitting to :mod:`swact`.

        The legacy implementation is available as
        :func:`reformat_result_databases_legacy`.
        """

        reformatted: list[pd.DataFrame] = []
        isolated: list[pd.DataFrame] = []
        for db in databases:
            new_db, iso = swact_reformat_result_databases(
                db,
                set_reset_at_zero=set_reset_at_zero,
                input_ports=experiment_config["input_ports"],
                output_ports=experiment_config["output_ports"],
            )
            reformatted.append(new_db)
            isolated.append(iso)
        return reformatted, isolated

    @staticmethod
    def reformat_result_databases_legacy(
        databases: list[pd.DataFrame], experiment_config: dict, set_reset_at_zero: bool = False
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """Original operand splitting routine kept for regression tests."""

        notawire_names = []
        input_rep_names = []
        port_names_dict = {
            "input_ports": experiment_config["input_ports"],
            "output_ports": experiment_config["output_ports"],
        }
        for key, port_names_list in port_names_dict.items():
            for port_name in port_names_list:
                notawire_names.extend([f"{port_name}_val", f"{port_name}_rep"])
                if key == "input_ports":
                    input_rep_names.append(f"{port_name}_rep")

        isolated_representations = []
        for db_idx, db in enumerate(databases):
            isolated_representations.append(copy(db[notawire_names]))
            rep_arrays = [
                np.roll(utils.from_binstr_list_to_int_array(db[column_name].to_list()), -1, axis=0)
                for column_name in input_rep_names
            ]
            nb_values, bitwidth = rep_arrays[0].shape
            new_col_names = []
            for tgt_name in input_rep_names:
                _tgt_name = tgt_name.replace("_rep", "_i").replace("input", "operand")
                new_col_names.extend([f"{_tgt_name}_{i}_" for i in range(bitwidth - 1, -1, -1)])
            bit_arrays = []
            for rep_array in rep_arrays:
                bit_arrays.extend(np.hsplit(rep_array, bitwidth))
            for col_name, bit_array in zip(new_col_names, bit_arrays):
                databases[db_idx][col_name] = bit_array

        databases = [db.drop(columns=notawire_names) for db in databases]

        if set_reset_at_zero:
            for db_idx in range(len(databases)):
                databases[db_idx].loc[0] = 0

        return databases, isolated_representations

    @staticmethod
    def assert_exactness(
        design_lut_path: Path,
        test_db: pd.DataFrame | list[pd.DataFrame],
        experiment_config: dict,
        design_swact_oudir_path: Path,
        save_histogram: bool = False,
        test_type_name: str | None = None,
    ) -> bool:
        #
        design_type = experiment_config["design_type"]
        input_rep_name_list = [f"{port_name}_rep" for port_name in experiment_config["input_ports"]]
        output_rep_name_list = [f"{port_name}_rep" for port_name in experiment_config["output_ports"]]

        # Get design number
        design_number = utils.extract_int_string_from_string(design_lut_path.parents[1].name)

        # Get encoding
        repr_to_val_dict = file_parsers.extract_encodings(design_lut_path, reverted=True)

        # Pick random lines in a database and verify correctness
        is_ok_per_db = []

        # Pick random lines
        # random_lines = test_db.sample(n=min(len(test_db), 3000))
        random_lines = test_db
        if len(test_db) < 3000:
            report_filepath = design_lut_path.parents[3] / "logs/unfinished_tests.log"
            with open(report_filepath, "a") as f:
                f.write(
                    f"Design number {design_number} did not entirely finish its test {test_type_name}. Available number of stimuli: {len(test_db)}\n"
                )

        # Get list of arrays of representations and values
        input_rep_arrays = []
        input_val_arrays = []
        for input_rep_name in input_rep_name_list:
            # Get list of representations
            rep_array = random_lines[input_rep_name].to_numpy()
            input_rep_arrays.append(rep_array)
            # Get list of values
            val_array = np.vectorize(repr_to_val_dict["input"].get)(rep_array)
            input_val_arrays.append(val_array)

        output_rep_arrays = []
        output_val_arrays = []
        for output_rep_name in output_rep_name_list:
            # Get list of representations
            rep_array = random_lines[output_rep_name].to_numpy()
            output_rep_arrays.append(rep_array)
            # Get list of values
            val_array = np.vectorize(repr_to_val_dict["output"].get)(rep_array)
            output_val_arrays.append(val_array)

        # Generate the target output vector
        if design_type == "adder":
            assert len(output_val_arrays) == 1, (
                f"Found more than one output port in adder experiment. That should not be."
            )

            # Compute element-wise addition
            c_sum = np.array(input_val_arrays).sum(axis=0)
            c_vals = np.clip(
                c_sum, a_min=-4, a_max=3
            )  # Values must be clipped as we decided that the adder lut simulates +inf and -inf

            # Ensure all values match the target value
            is_ok_per_db.append(np.all((output_val_arrays[0] - c_vals) == 0))

        elif design_type == "multiplier":
            assert len(output_val_arrays) == 1, (
                f"Found more than one output port in multiplier experiment. That should not be."
            )

            # Compute element-wise multiplication
            c_vals = np.prod(np.array(input_val_arrays), axis=0)

            # Ensure all values match the target value
            is_ok_per_db.append(np.all((output_val_arrays[0] - c_vals) == 0))

        elif design_type == "encoder":
            assert len(input_val_arrays) == 1, (
                f"Found more than one input port in encoder experiment. That should not be."
            )
            assert len(output_val_arrays) == 1, (
                f"Found more than one output port in encoder experiment. That should not be."
            )

            # Output values should be equal to input values
            c_vals = input_val_arrays[0]

            # Ensure all values match the target value
            is_ok_per_db.append(np.all((output_val_arrays[0] - c_vals) == 0))

        # Store histogram of value for later stage validation of test type
        if save_histogram:
            test_val_hist_dict = dict()
            for val_array, port_name in zip(
                input_val_arrays + output_val_arrays,
                experiment_config["input_ports"] + experiment_config["output_ports"],
            ):
                vals_unique, vals_count = np.unique(val_array, return_counts=True)

                test_val_hist_dict[port_name] = {
                    "vals": vals_unique,
                    "count": vals_count,
                }

            utils.save_serialized_data(
                npz_filepath=design_swact_oudir_path / f"test_val_hist_{test_type_name}.npz", data=test_val_hist_dict
            )

        return np.all(is_ok_per_db)

    @staticmethod
    def _save_db_str(db: pd.DataFrame, filepath: Path, title_name: str | None = "") -> None:
        """This function converts all columns of a dataframe to string and save it as parquet file."""

        # Convert paths and dicts to string
        for el in db.columns:
            db[el] = db[el].apply(lambda x: str(x))

        # Save as csv file
        db.to_parquet(filepath.with_suffix(".pqt"), index=False)

        if title_name is not None:
            logger.opt(colors=True).info(f"<yellow>Database {title_name}</yellow> saved at {filepath}")

    def get_topk_designs(self, k: int = 10) -> dict[str, pd.DataFrame]:
        """This function gets the design names of the top k design with lowest switching activity."""

        if self.synth_df.empty:
            logger.info("No synthesis results found to get the top k designs. Please run the synthesis first.")
            return {}

        # Get the top k designs with lowest nb_cells
        topk_df_dict: dict[str, pd.DataFrame] = {}
        col_front = [
            "design_number",
            "nb_transistors",
            "is_special",
            "special_names",
            "encodings_input",
            "encodings_output",
        ]

        if self.swact_df is not None and not self.swact_df.empty:
            if "swact_weighted_total" in self.swact_df.keys():
                swact_metric = "swact_weighted_total"
            else:
                swact_metric = "swact_weighted_average"
            col_front += [
                "nb_wires",
                swact_metric,
            ]

            for test_type_name in self.test_type_names:
                sub_swact_df = self.swact_df[self.swact_df["test_type"] == test_type_name]
                topk_design_numbers = sub_swact_df.sort_values(by=swact_metric).head(k)["design_number"]
                topk_swact_df = sub_swact_df[sub_swact_df["design_number"].isin(topk_design_numbers)]
                topk_synth_df = self.synth_df[self.synth_df["design_number"].isin(topk_design_numbers)]

                # Combine topk databases
                topk_full_df = pd.merge(topk_swact_df, topk_synth_df, on="design_number", suffixes=("", "_synth"))
                topk_full_df = topk_full_df.loc[:, ~topk_full_df.columns.duplicated()]

                topk_full_df = topk_full_df.sort_values(by=[swact_metric, "nb_transistors", "nb_wires"])
                topk_df_dict[test_type_name] = topk_full_df

        else:
            topk_design_numbers = self.synth_df.sort_values(by="nb_transistors").head(k)["design_number"]
            topk_full_df = self.synth_df[self.synth_df["design_number"].isin(topk_design_numbers)]
            sort_values = ["nb_transistors"]
            topk_full_df = topk_full_df.sort_values(by=sort_values)
            topk_df_dict["no_test"] = topk_full_df

        for test_type_name, topk_full_df in topk_df_dict.items():
            # Reorder the table
            topk_full_df = topk_full_df.reindex(
                columns=col_front + [col for col in topk_full_df.columns if col not in col_front]
            )

            # Save the table
            self._save_db_str(
                topk_full_df,
                self.analysis_out_dir / f"topk_designs_{test_type_name}.pqt",
                title_name=f"topk_{test_type_name}",
            )

        return topk_df_dict

    def remove_skip_tests_list(self) -> None:
        """Checks the list of test type to skip and remove them from the"""

        if len(self.swact_df) > 0:
            for test_type in self.skip_tests_list:
                self.test_type_nb -= 1
                self.swact_df.drop(self.swact_df[self.swact_df["test_type"] == test_type].index, inplace=True)

        return None

    def _plot_swact_vs_ncells(self) -> None:
        """This function only plots weighted total swact vs ncells. It should be used to simplify launching interactive graph."""

        # Update the types of values in databases
        self.format_databases()

        # Reorder databases based on the design number
        self.align_databases()

        # Remove test types that one does not want to plot
        self.remove_skip_tests_list()

        # Plot the
        is_interactive = int(os.environ.get("MPL_INTERACTIVE", 0)) != 0
        plotter.plot_swact_type_versus_area(
            self,
            swact_type="total",
            weighted=True,
            marker_size_type="nb_zero_swact",
            colorbar_type="max_depth",
            interactive=is_interactive,
        )

    def align_databases(self) -> None:
        """Reorder the synthesis db based on the design numbers of swact"""

        all_dfs = []
        if not self.synth_df.empty:
            all_dfs.append(self.synth_df)
        if not self.swact_df.empty:
            all_dfs.append(self.swact_df)
        if not self.power_df.empty:
            all_dfs.append(self.power_df)
        if not self.cmplx_df.empty:
            all_dfs.append(self.cmplx_df)

        if len(all_dfs) < 2:
            logger.warning("Did not receive enough databases to align.")
            return

        # Step 1: Find common sorted values in "design_number"
        common_elements = reduce(lambda x, y: x & y, [set(df["design_number"]) for df in all_dfs])
        common_elements = sorted(common_elements)

        # Step 2: Filter and sort each DataFrame by "design_number"
        def align_df(df):
            if not df.empty:
                return df[df["design_number"].isin(common_elements)].sort_values("design_number").reset_index(drop=True)
            else:
                return df

        synth_df = align_df(self.synth_df)
        swact_df = align_df(self.swact_df)
        power_df = align_df(self.power_df)
        cmplx_df = align_df(self.cmplx_df)

        logger.info(f"All databases aligned. All databases are now of lenght {len(self.synth_df)}")

        self.synth_df = synth_df
        self.swact_df = swact_df
        self.power_df = power_df
        self.cmplx_df = cmplx_df

        return None

    def format_databases(self, which_db: str = None) -> None:
        """Format some columns in the Analyzer db to simplify data analysis afterward."""
        if len(self.gener_df) or which_db == "gener":
            gener_mapping_dict = {
                "encodings_input": str,
                "encodings_output": str,
            }

            # Enforce data types on self.synth_df
            gener_df_columns = self.gener_df.columns
            missing_columns = []
            for key in gener_mapping_dict.keys():
                if key in gener_df_columns:
                    self.gener_df[key] = self.gener_df[key].map(gener_mapping_dict[key], na_action="ignore")
                else:
                    missing_columns.append(key)
            if missing_columns:
                logger.warning(f"Columns {missing_columns} were expected to be found in self.gener_df, but were not!")

        if len(self.synth_df) or which_db == "synth":
            synth_mapping_dict = {
                "is_special": utils.str_to_int_to_bool,  # solves an issue with direct conversion to bool datatypes
                "nb_cells": int,
                "nb_transistors": int,
                "tot_cell_area": float,
                "max_cell_depth": float,
                "design_number": str,
                "max_fanout_internal": float,
                "min_fanout_internal": float,
                "max_fanout_io": float,
                "min_fanout_io": float,
                "encodings_input": str,
                "encodings_output": str,
                "special_names": str,
                "$_ANDNOT_": float,
                "$_AND_": float,
                "$_DFF_PN0_": float,
                "$_MUX_": float,
                "$_NAND_": float,
                "$_NOR_": float,
                "$_NOT_": float,
                "$_ORNOT_": float,
                "$_OR_": float,
                "$_XNOR_": float,
                "$_XOR_": float,
            }

            # Enforce data types on self.synth_df
            synth_df_columns = self.synth_df.columns
            missing_columns = []
            for key in synth_mapping_dict.keys():
                if key in synth_df_columns:
                    self.synth_df[key] = self.synth_df[key].map(synth_mapping_dict[key], na_action="ignore")
                else:
                    missing_columns.append(key)
            if missing_columns:
                logger.warning(f"Columns {missing_columns} were expected to be found in self.synth_df, but were not!")

        if len(self.swact_df) or which_db == "swact":
            swact_mapping_dict = {
                "is_special": utils.str_to_int_to_bool,
                "nb_wires": int,
                "design_number": str,
                "max_io_swact": float,
                "max_io_swact_count": float,
                "max_internal_swact": float,
                "max_internal_swact_count": float,
                "nb_zero_internal_swact": float,
            }

            for suffix in ["", "_weighted", "_internal_weighted", "_io_weighted"]:
                swact_mapping_dict[f"per_wire_swact{suffix}_average"] = float
                swact_mapping_dict[f"per_wire_swact{suffix}_stddev"] = float
                swact_mapping_dict[f"per_wire_min_swact{suffix}"] = float
                swact_mapping_dict[f"per_wire_min_swact{suffix}_wire_name"] = str
                swact_mapping_dict[f"per_wire_max_swact{suffix}"] = float
                swact_mapping_dict[f"per_wire_max_swact{suffix}_wire_name"] = str
                swact_mapping_dict[f"swact{suffix}_total"] = float
                swact_mapping_dict[f"swact{suffix}_average"] = float

            for k, v in swact_mapping_dict.items():
                if k in self.swact_df.columns:
                    self.swact_df[k] = self.swact_df[k].map(v)

        if len(self.power_df) or which_db == "power":
            swact_mapping_dict = {
                "is_special": utils.str_to_int_to_bool,
                "p_comb_dynamic": float,
                "p_comb_static": float,
                "design_number": str,
            }

            for k, v in swact_mapping_dict.items():
                if k in self.swact_df.columns:
                    self.swact_df[k] = self.swact_df[k].map(v)

        if len(self.cmplx_df) or which_db == "cmplx":
            swact_mapping_dict = {
                "is_special": utils.str_to_int_to_bool,
                "p_comb_dynamic": float,
                "p_comb_static": float,
                "design_number": str,
            }

            for k, v in swact_mapping_dict.items():
                if k in self.swact_df.columns:
                    self.swact_df[k] = self.swact_df[k].map(v)

        return None

    def _add_special(self, db: pd.DataFrame) -> pd.DataFrame:
        """Adds columns for is_special and special_names to the specified database if there is a list of special designs given to the Analyzer (via the json file)."""
        if self.special_designs_ids is not None and len(self.special_designs_ids) > 0:
            # Add is_special booleans columns to the database
            db["is_special"] = db["design_number"].isin(self.special_designs_ids).astype(int)

            # Add spacial_names str columns to the database
            special_names = db["is_special"].to_numpy(dtype=str)
            for design_number, special_name in zip(self.special_designs_ids, self.special_designs_dict["legend"]):
                special_names[db["design_number"] == design_number] = special_name
            special_names[db["is_special"] == 0] = ""  # Empty string for non-special designs
            db["special_names"] = special_names
        else:
            # Add a column where all special is false
            db["is_special"] = False
            db["special_names"] = ""

        db.astype({"is_special": "bool"})
        return db

    def save_database(self, db_type: str) -> None:
        """This function saves the pandas databases as a parquet file"""

        if self.args_dict.get("rebuild_db", False) or self.is_user_design_number_list:
            if (not self.args_dict.get(f"skip_{db_type}", False)) or self.is_user_design_number_list:
                db_path = getattr(self, f"output_{db_type}_db_path")
                if self.is_debug:
                    db_path = db_path.with_suffix(".debug.pqt")
                df = getattr(self, f"{db_type}_df")
                if not df.empty:
                    df = self._add_special(df)
                    setattr(self, f"{db_type}_df", df)
                    self.format_databases(db_type)
                    df = getattr(self, f"{db_type}_df")
                    df.to_parquet(db_path.with_suffix(".pqt"), index=False)
                    logger.opt(colors=True).info(f"<yellow>{db_type} Database</yellow> saved at {db_path}")
                else:
                    logger.info(f"No {db_type} Database to save, skipping save.")

    def analyze_results(self) -> None:
        """This function does various plots."""

        # Check that there is at least a database to do some plots
        any_not_empty = False
        for step in Analyzer.__existing_steps__:
            if step == "gener":
                continue
            if not getattr(self, f"{step}_df").empty:
                any_not_empty = True
                break
        if not any_not_empty:
            logger.warning(f"No database to plot, exiting ...")
            return None

        # Update the types of values in databases
        self.format_databases()

        # Reorder databases based on the design number
        self.align_databases()

        # Remove test types that one does not want to plot
        self.remove_skip_tests_list()

        # Do all the plots
        if not self.args_dict.get("skip_plots", False):
            if not self.cmplx_df.empty:
                plotter.plot_cmplx_distribution(self)
            if not self.args_dict.get("fast_plots", False):
                if not self.args_dict.get("skip_synth", False):
                    plotter.plot_max_fanout_versus_max_depth(self)
                    plotter.plot_cell_count_distribution(self, type="nb_transistors")
                    plotter.plot_cell_count_distribution(self, type="nb_cells")

                if not self.args_dict.get("skip_power", False):
                    if self.args_dict.get("interactive"):
                        import genial.plot_apps.plotter_dash as plotter_dash
                        from genial.training.elements.score_tools import ScoreComputeHelper

                        merged_df = ScoreComputeHelper.merge_data_df(synth_df=self.synth_df, swact_df=self.swact_df)
                        plotter_dash.launch_dash_scatter_plot(
                            merged_df,
                            self.power_df,
                            info_dict={
                                "experiment_name": self.dir_config.experiment_name,
                                "output_dir_name": self.dir_config.output_dir_name,
                                "technology": self.dir_config.technology,
                            },
                        )
                        exit()
                    else:
                        if not self.swact_df.empty:
                            plotter.plot_swact_versus_power(self)
                        plotter.plot_power_versus_area(self)
                        plotter.plot_power_distribution(self)

                # Get the designs with lowest switching activity count
                if not self.args_dict.get("skip_swact", False):
                    for weighted in [True]:
                        for swact_type in ["total"]:  # , "average", "minimum", "maximum"]:
                            for marker_size_type in ["max_depth"]:  # ["nb_zero_swact", "max_depth"]:
                                for colorbar_type in ["max_depth", "total_worst_swact"]:
                                    for x_axis_type in [
                                        "nb_transistors",
                                        "tot_cell_area",
                                    ]:  # ["nb_cells", "nb_transistors", "tot_cell_area"]:
                                        plotter.plot_swact_type_versus_area(
                                            self,
                                            swact_type=swact_type,
                                            x_axis_type=x_axis_type,
                                            weighted=weighted,
                                            marker_size_type=marker_size_type,
                                            colorbar_type=colorbar_type,
                                        )
                                        plotter.plot_swact_type_versus_area(
                                            self,
                                            swact_type=swact_type,
                                            x_axis_type=x_axis_type,
                                            weighted=weighted,
                                            marker_size_type=marker_size_type,
                                            colorbar_type=colorbar_type,
                                            dataset="weighted",
                                            relative=True,
                                        )
                        plotter.plot_min_max_swact_(self, weighted=weighted)

                    # Get the designs with the smallest gate count
                    topk_df_dict = self.get_topk_designs(k=10)
                    plotter.plot_encoding_heatmaps(self, topk_df_dict)
                    if not self.skip_fullsweep_analysis:
                        plotter.plot_test_type_value_histogram(self, self.test_type_names)

            else:
                if not self.args_dict.get("skip_swact", False):
                    weighted = True
                    swact_type = "total"
                    marker_size_type = "max_depth"
                    colorbar_type = "max_depth"
                    for x_axis_type in ["nb_transistors", "tot_cell_area"]:
                        plotter.plot_swact_type_versus_area(
                            self,
                            swact_type=swact_type,
                            x_axis_type=x_axis_type,
                            weighted=weighted,
                            marker_size_type=marker_size_type,
                            colorbar_type=colorbar_type,
                        )
                        plotter.plot_swact_type_versus_area(
                            self,
                            swact_type=swact_type,
                            x_axis_type=x_axis_type,
                            weighted=weighted,
                            marker_size_type=marker_size_type,
                            colorbar_type=colorbar_type,
                            dataset="weighted",
                            relative=True,
                        )
                else:
                    plotter.plot_cell_count_distribution(self, type="nb_transistors")
                    plotter.plot_cell_count_distribution(self, type="nb_cells")
                    if self.dir_config.is_bulk_flow_mode:
                        if self.dir_config.bulk_flow_dirname == "power_out":
                            plotter.plot_power_versus_area(self)
                            plotter.plot_power_distribution(self)

        logger.info(f"All analysis and plots done.")

    def merge_design_encodings_in_synth_df(
        self,
    ):
        """
        Function used for retro-compatibility where all generation information was stored in the synth_df.
        """
        if not self.synth_df.empty:
            self.synth_df = utils.add_new_df_to_df(self.gener_df, self.synth_df, keep_only_preexisting=True)

    def main(self) -> None:
        start_time = time()
        logger.info(f"Starting Analyzer.main ...")

        # Add info to experiment configuration
        self.find_classic_encoding_design_number(write_back=True)
        self._setup_lists_of_designs_todo()
        self.get_all_design_encodings()
        self.save_database(db_type="gener")

        # Analyse synthesis results (SYNTH)
        if not self.args_dict.get("skip_synth", False):
            self.get_all_number_of_cells()
            self.get_all_fanout_wires_n_depth()
            self.merge_design_encodings_in_synth_df()
            self.save_database(db_type="synth")
        else:
            logger.info(f"Synthesis analysis has been skipped.")

        # Analyse complexity results (CMPLX)
        if not self.args_dict.get("skip_cmplx", False):
            self.get_all_design_complexities()
            self.save_database(db_type="cmplx")
        else:
            logger.info(f"Complexities analysis has been skipped.")

        # Analyze test results (SWACT)
        if not self.args_dict.get("skip_swact", False):
            self.get_all_switching_activity()
            self.save_database(db_type="swact")
        else:
            logger.info(f"Switching activity analysis has been skipped.")

        # Analyze Power results (POWER)
        if not self.args_dict.get("skip_power", False):
            self.get_all_power()
            self.save_database(db_type="power")
        else:
            logger.info(f"Power extraction analysis has been skipped.")

        # Analyze the results and to the plots
        self.analyze_results()

        end_time = time()

        logger.info(
            f"Analyzer.main done in {(end_time - start_time) / 60 / 60:.2f}h or {int(end_time - start_time)}s.\n"
        )
        return None

    def find_classic_encoding_design_number(self, write_back: bool = True, force: bool = False) -> str:
        """Parses the list of generated designs and finds the design number that has the classic encoding"""

        logger.info(f"Looking for the classic encoding in existing generated designs ...")

        if self.is_debug:
            logger.debug(f"self.special_designs_dict:{self.special_designs_dict}")

        if "classic_encoding" in self.special_designs_dict["legend"] and not force:
            design_number = self.special_designs_dict["design_numbers"][
                self.special_designs_dict["legend"].index("classic_encoding")
            ]
            return design_number

        classic_encoding = DesignGenerator._get_all_design_configuration_dictionnaries(
            nb_to_generate=1, exp_config_dict=self.exp_config, permute_in=False, permute_out=False
        )
        # The first (and only) design should be the design without any permutation
        classic_encoding = classic_encoding[0]
        classic_encoding_subdict_str = str(
            {"input": str(classic_encoding["in_enc_dict"]), "output": str(classic_encoding["out_enc_dict"])}
        )

        # Find the design with the same encoding
        encoding_dicts_df = file_parsers.read_all_existing_encodings_v2(
            self.dir_config.root_output_dir, self.dir_config.bulk_flow_dirname
        )
        cond = encoding_dicts_df["enc_dict_str"] == classic_encoding_subdict_str
        found = False
        if cond.sum():
            found = True
            design_number = encoding_dicts_df[cond].iloc[0]

        if found:
            # Save the classic encoding to a file
            comment = DesignGenerator._extend_text_with_encoding_comment("", classic_encoding)
            classic_encoding_filepath = self.dir_config.root_output_dir / "classic_encoding.v"
            classic_encoding_filepath.write_text(comment)
            logger.info(f"Stored classic encoding in {classic_encoding_filepath}")

            # Write back design number to special designs
            if write_back:
                self.extend_special_designs(design_number=design_number, legend="classic_encoding")

        else:
            logger.warning("Design with classic encoding could not be found in all generated designs.")
            design_number = None

        return design_number

    @staticmethod
    def parse_args() -> dict[str, Any]:
        args_dict = analyzer_parser()

        return args_dict

    def switch_to_iter_mode(self, config_dict: dict[str, Any], **kwargs) -> None:
        """
        Update the analyzer state so that it runs in iteration mode.
        Returns:
            True if somthing went wrong.
            False otherwise.
        """
        self.is_user_design_number_list = True
        self.args_dict["continue"] = True
        self.args_dict["rebuild_db"] = False

    def prepare_next_iter(self) -> bool:
        """Prepare the next iteration nby updating the list of designs todo."""
        return self.update_todo_design_list()

    def update_todo_design_list(self) -> None:
        """Update the list of designs todos by comparing te already done designs"""
        _todo_design_list = self._get_continue_todo_design_numbers()
        if _todo_design_list is None:
            logger.warning(f"No new desgins to analyze have been found. Returning.")
            return True

        self.user_design_number_list = _todo_design_list
        return False


def main_cli() -> Analyzer:
    start_time = time()
    # try:
    analyzer = Analyzer(reset_logs=True)
    analyzer.main()

    status = "Success"
    error_msg = ""
    # except:
    #     status="Failed"
    #     error_msg = traceback.format_exc()

    logger.info(error_msg)
    analyzer.send_email(
        config_dict=analyzer.args_dict,
        start_time=start_time,
        status=status,
        error_message=error_msg,
        calling_module="Analyzer",
        root_output_dir=analyzer.dir_config.root_output_dir,
    )

    logger.info(f"Analyzer's `main_cli` exited with status: {status}.")
    if status == "Failed":
        logger.error(error_msg)

    return analyzer


def main_find_classic_encoding_design_number() -> None:
    args_dict = Analyzer.parse_args()

    args_dict["rebuild_db"] = True
    args_dict["skip_swact"] = False
    args_dict["skip_synth"] = False

    dir_config = ConfigDir(is_analysis=True, **args_dict)

    analyzer = Analyzer(
        dir_config=dir_config,
        reset_logs=False,
    )

    design_number = analyzer.find_classic_encoding_design_number(force=True)

    logger.opt(colors=True).info(f"For experiment {args_dict.get('experiment_name')}")
    logger.opt(colors=True).info(f"The LUT-based design with <red>classic encoding is {design_number}</red>")


def main_plot_swact_vs_ncells() -> None:
    analyzer = Analyzer()

    analyzer._plot_swact_vs_ncells()


if __name__ == "__main__":
    analyzer = main_cli()
    # main_find_classic_encoding_design_number()

    logger.info("DONE")
