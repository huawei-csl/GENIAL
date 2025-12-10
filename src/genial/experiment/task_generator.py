# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

import shutil
import argparse
from string import Template
from typing import Any, Literal
from pathlib import Path
import itertools
from copy import copy
from loguru import logger
import traceback
from time import time

import math
import numpy as np
import pandas as pd


from tqdm import tqdm

from genial.utils import utils
from genial.config.config_dir import ConfigDir
from genial.config.arg_parsers import get_default_parser

import genial.experiment.plotter as plotter
import genial.experiment.binary_operators as bin_ops
import genial.experiment.file_parsers as file_parsers

from genial.config.logging import Logging as logging
from genial.globals import global_vars

from genial.experiment.file_parsers import process_pool_helper
from genial.experiment.loop_module import LoopModule
from genial.experiment.encoding_distance import get_all_col_permuted_list

import re
import json

from swact.file_compression_handler import FileCompressionHandler

from genial.ext_plugins.flowy.hdl_generator import AdaptedFlowyHdlGenerator


class DesignGenerator(LoopModule):
    def __init__(self, dir_config: ConfigDir | None = None, **kwargs) -> None:
        super().__init__()

        logger.info(f"Setting up DesignGenerator ...")

        # Initialize the experiment configuration
        if dir_config is None:
            self.args_dict = DesignGenerator.parse_args()
            self.dir_config = ConfigDir(is_analysis=False, **self.args_dict)
        else:
            gener_args_dict = DesignGenerator.parse_args()
            self.dir_config = dir_config
            for key in gener_args_dict.keys():
                if key not in dir_config.args_dict.keys():
                    dir_config.args_dict[key] = gener_args_dict[key]
            self.args_dict = dir_config.args_dict
            dir_config.update_experiment_configuration(args_dict=dir_config.args_dict)

        logging().init_logging(self.dir_config.root_output_dir, mode="generator_main")

        self.args_dict.update(**kwargs)

        self.gener_templates_dir = self.dir_config.exp_templates_dir / "generation"

        self.template_verilog_filepaths = self.dir_config.find_all_temp_files(step="generation")

        self.design_templates = self.init_templates()

        if self.args_dict.get("debug", False):
            self.is_debug = True
            self.nb_design = 2
            self.nb_workers = 1
            # logger.debug(f"Debug mode is ON. Only {self.nb_design} designs will be generated.")
        else:
            self.is_debug = False
            self.nb_design = None
            nb_workers = self.args_dict.get("nb_workers", 32)
            self.nb_workers = 128 if nb_workers is None else nb_workers

        self.user_design_number_list = self.args_dict.get("design_number_list", None)

        self.plot_dir = self.dir_config.analysis_out_dir / "generation/plots"
        if not self.plot_dir.exists():
            self.plot_dir.mkdir(parents=True)

        self.exp_config_dict = self.dir_config.get_experiment_configuration()

        # Setup columns for truth table generation
        in_n_out_port_names = self.exp_config_dict["input_ports"] + self.exp_config_dict["output_ports"]
        port_val_names = []
        port_rep_names = []
        for port_name in in_n_out_port_names:
            port_val_names.append(f"{port_name}_val")
            port_rep_names.append(f"{port_name}_rep")

        self._db_columns = port_val_names + port_rep_names

        # Prepare the number of zeros for the design number
        self.zfill_len = None

        self.existing_encodings_path = self.args_dict.get("existing_encodings_path", None)
        if self.exp_config_dict.get("design_type") == "encoder":
            assert self.existing_encodings_path is not None, (
                f"For generating the encoders, the user must pass the list of original design (not encoders) paths as `--existing_encodings_path` argument ."
            )
        if self.existing_encodings_path is not None:
            self.existing_encodings_path = Path(self.existing_encodings_path)
            if not self.existing_encodings_path.exists():
                logger.error(f"Specified path to existing design for reading encodings do not exist.")
                logger.error(f"Specified path is {self.existing_encodings_path}")
                raise ValueError

        self.seed = self.args_dict.get("seed", 1)

        logger.info(f"DesignGenerator initialized.\n\n")

    def init_templates(self) -> list[tuple[str, Template]]:
        return [
            (template_path.name, Template(template_path.read_text()))
            for template_path in self.template_verilog_filepaths
        ]

    @staticmethod
    def get_truth_table(design_config_dict: dict[str, dict[int, str] | int | str], columns: list[str]) -> pd.DataFrame:
        # Read useful elements from design configuration dictionary
        in_enc_dict = design_config_dict["in_enc_dict"]
        out_enc_dict = design_config_dict["out_enc_dict"]
        design_type = design_config_dict["design_type"]
        default_repr = design_config_dict["input_default"]
        is_mixed = design_config_dict["in_enc_type"] == "mixed"
        if is_mixed:
            key_index = list(in_enc_dict.keys())

        variables = {col: str() for col in columns}
        df = pd.DataFrame(variables, index=[])

        all_lines = []
        in_min_val = design_config_dict["min_val_in_included"]
        in_max_val = design_config_dict["max_val_in_included"]
        if design_type == "adder":
            for a in range(in_min_val, in_max_val + 1):
                for b in range(in_min_val, in_max_val + 1):
                    c = a + b
                    a_rep = in_enc_dict[a]
                    b_rep = in_enc_dict[b]
                    if c > in_max_val:
                        c_rep = out_enc_dict[in_max_val]
                    elif c < in_min_val:
                        c_rep = out_enc_dict[in_min_val]
                    else:
                        c_rep = out_enc_dict[c]

                    new_data = [a, b, c, a_rep, b_rep, c_rep]
                    new_df = {col: [val] for col, val in zip(df.columns, new_data)}
                    new_df = pd.DataFrame(new_df, columns=df.columns)
                    all_lines.append(new_df)

        elif design_type == "multiplier":
            for a in range(in_min_val, in_max_val + 1):
                for b in range(in_min_val, in_max_val + 1):
                    c = a * b
                    a_rep = in_enc_dict[a]
                    if is_mixed:
                        b_rep = default_repr[key_index.index(b)]
                    else:
                        b_rep = in_enc_dict[b]

                    c_rep = out_enc_dict[c]
                    new_data = [a, b, c, a_rep, b_rep, c_rep]
                    new_df = {col: [val] for col, val in zip(df.columns, new_data)}
                    new_df = pd.DataFrame(new_df, columns=df.columns)
                    all_lines.append(new_df)

        elif design_type == "encoder":
            for a, a_rep in in_enc_dict.items():
                c = a
                c_rep = out_enc_dict[c]
                new_data = [a, c, a_rep, c_rep]
                new_df = {col: [val] for col, val in zip(df.columns, new_data)}
                new_df = pd.DataFrame(new_df, columns=df.columns)
                all_lines.append(new_df)

        else:
            raise ValueError(f"Design type: {design_type} has not been implemented")

        # Recompose truth table as a single DataFrame
        df = pd.concat(all_lines, ignore_index=True)

        return df

    @staticmethod
    def convert_to_verilog(truth_table: pd.DataFrame) -> str:
        """
        This function takes a truth table database (columns = binary representation of each single operand and output),
        And formats it to the equivalent verilog code.
        """

        output_str = "  always @(sel) begin\n"
        output_str += "    unique case(sel)\n"

        # logger.info(f"Length of truth table {len(truth_table)}")
        # Find all input operands that will be concatenated into the selection signal
        input_names = []
        for name in list(truth_table.columns):
            if name.split("_")[-1] == "rep":
                if name.split("_")[0] == "input":
                    input_names.append(name)

        for idx, row in truth_table.iterrows():
            output_str += "      "
            case_selection_string = ""
            for in_name in input_names:
                case_selection_string += row[in_name]
            result_string = row["output_rep"]
            output_str += f"{len(case_selection_string):.0f}'b{case_selection_string} : out = {len(result_string):.0f}'b{result_string};\n"
        # output_str += f"        default : out = {'0'*self._bitwidth};\n"
        output_str += "    endcase\n"
        output_str += "  end\n"

        return output_str

    @staticmethod
    def generate_local_params(design_config_dict) -> str:
        """
        This function generate the list of local params associated with the design config file.
        Used for: FSM,
        """

        if design_config_dict.get("design_type") == "fsm":
            in_bitwidth = design_config_dict.get("in_bitwidth")
            # output_str = "// State encodings (localparams)\n"
            output_str = ""
            for val, repr in design_config_dict.get("in_enc_dict").items():
                output_str += f"    localparam S{val} = {in_bitwidth}'b{repr}; // State {val}\n"
        else:
            raise NotImplementedError("Only FSM design type is supported for now.")

        return output_str

    def __prepare_all_gener_dir_path(
        self, design_numbers_list: list[str], output_dir_path: Path | None = None
    ) -> list[Path]:
        """Prepare all output design directories"""

        assert design_numbers_list is not None, ValueError(f"Either design_numbers_list must be passed as argument.")
        shared_files = self.dir_config._get_shared_files(step="generation")

        # Setup output directory path
        if output_dir_path is not None:
            # If specified, the output_dir_path given as argument has precedence over the default Generator output dir path
            _output_dir_path = output_dir_path
        else:
            _output_dir_path = self.dir_config.generation_out_dir

        nb_designs = len(design_numbers_list)
        logger.info(f"Generating {nb_designs} in {_output_dir_path} ... ")
        with tqdm(
            total=nb_designs, desc=f"x{self.nb_workers}|Output design directories preparation"
        ) as pbar:  # Progress bar
            gener_filepaths = utils.process_pool_helper(
                func=DesignGenerator._prepare_gener_dir_path,
                func_args_gen=(
                    (design_number, _output_dir_path, shared_files, self.template_verilog_filepaths)
                    for design_number in design_numbers_list
                ),
                max_workers=self.nb_workers,
                pbar=pbar,
            )

        output_path_list = gener_filepaths

        return output_path_list

    @staticmethod
    def _prepare_gener_dir_path(
        design_number: str,
        generation_out_dir_path: Path,
        shared_files: list[Path],
        template_verilog_filepaths: list[Path],
    ) -> list[Path]:
        """Prepare the generated design directory and returns its of paths."""

        design_dir_name = f"res_{design_number}"
        design_dir_path = generation_out_dir_path / design_dir_name / "hdl"
        design_dir_path.mkdir(parents=True, exist_ok=True)

        for template_file in shared_files:
            if template_file not in template_verilog_filepaths:
                # print(template_verilog_filepaths)
                shutil.copy(template_file, design_dir_path)

        return design_dir_path

    def generate_more_designs(
        self,
        nb_designs: int | None = None,
        output_dir_path: Path | None = None,
        ignore_already_existing: bool = False,
        do_encodings_only: bool = False,
    ) -> list[dict]:
        """
        This function generates more encodings. For each encoding generated, it makes sure it's not already existing in the output directory.

        Args:
         - nb_designs: number of designs to generate. If None, the function will generate all possible designs.
         - output_dir_path: path to the output directory. If None, the function will use the default output directory.
         - do_encodings_only: whether to make the design configurations, but not actually buid and save all folders and files
         - ignore_already_existing: whether to ignore already existing designs in the output directory. If True, the function will generate new designs even if their encoding already exists in the output folder.

        Returns:
         - list of dictionaries, each dictionary containing the design number and the encodings used for the design.
        """

        start_time = time()

        # Correct nb_of_designs definition
        if self.args_dict.get("standard_encoding_only", False):
            logger.warning(f"`standard_encoding_only` is True, generating only the design with standard encoding.")
            nb_designs = 1
        elif nb_designs is None:
            logger.warning(f"Number of designs to generate was not provided, factorial(6) new designs.")
            nb_designs = math.factorial(6)

        # Specif the correct output directory path
        if output_dir_path is None:
            output_dir_path = self.dir_config.generation_out_dir

        # Get all existing original encodings
        known_encodings_dict_df = file_parsers.read_all_existing_encodings_v2(
            self.dir_config.root_output_dir, self.dir_config.bulk_flow_dirname
        )
        nb_existing_designs = len(known_encodings_dict_df)

        # Remove non generated designs from the list of known encoding dicts
        special_designs_dict = self.dir_config.read_special_designs(self.dir_config)
        standard_design_idx = None
        for idx, design_type in enumerate(special_designs_dict["legend"]):
            if "standard" in design_type:
                standard_design_idx = idx
                break
        if standard_design_idx is not None:
            drop_idx = known_encodings_dict_df[
                known_encodings_dict_df["design_number"] == special_designs_dict["design_numbers"][standard_design_idx]
            ].index
            known_encodings_dict_df.drop(drop_idx, inplace=True)

        # Instatiate new design numbers
        if nb_existing_designs > 0:
            existings_design_numbers = known_encodings_dict_df["design_number"].astype(np.int64).to_numpy()
        else:
            existings_design_numbers = np.array([-1], dtype=np.int64)
        max_existing_design_number = np.max(existings_design_numbers)

        # Prepare new design numbers
        avail_new_design_numbers = np.arange(1, int(nb_designs) + 1, 1) + max_existing_design_number
        avail_new_design_numbers = list(avail_new_design_numbers.astype(str))
        min_number = avail_new_design_numbers[0]
        max_number = avail_new_design_numbers[-1]

        # Generate nb_designs designs
        new_config_dicts_out = []
        final_paths_out = []
        new_design_count = 0

        timeout = 3  # Timeout to exit the loop when the number of design to generate is too high with respect to the remaining designs generable
        is_done = False
        while new_design_count < nb_designs:
            if is_done:
                break

            timeout -= 1
            if timeout < 0:
                logger.warning(f"Timeout reached while generating new designs.")
                break

            _nb_designs = nb_designs
            seed_offset = (
                1 + self.current_iter_nb + new_design_count + max_existing_design_number
            )  # Try to randomize as much as possible

            """
            Build the list of design configs. A design config is a dictionnary that contains those keys:
            {
                'design_type' -> multiplier, adder ...
                'in_bitwidth'
                'out_bitwidth'
                'in_enc_type' -> two's_comp, one_hot ...
                'out_enc_type' -> two's_comp, one_hot ...
                'in_enc_dict' -> encoding dictionnary
                'out_enc_dict' -> encoding dictionnary
            }
            """
            if do_encodings_only:
                _, design_config_list = self.generate_all_designs(
                    nb_to_generate=_nb_designs,
                    design_numbers=avail_new_design_numbers,
                    skip_files_generation=True,
                    seed_offset=seed_offset,
                )
            else:
                gener_filepaths, design_config_list = self.generate_all_designs(
                    nb_to_generate=_nb_designs,
                    design_numbers=avail_new_design_numbers,
                    output_dir_path=output_dir_path,
                    skip_files_generation=False,
                    seed_offset=seed_offset,
                )

            # for new_design_path, new_design_config_dict in new_design_paths_n_dict:
            if not known_encodings_dict_df.empty:
                known_encodings_str_set = set(known_encodings_dict_df["enc_dict_str"].to_list())
            else:
                known_encodings_str_set = set()

            for idx, new_design_config_dict in enumerate(design_config_list):
                if len(avail_new_design_numbers) == 0:
                    # The job is done, all new design shave been generated, we can exit the loop
                    is_done = True
                    break

                new_encoding_dict_str = str(
                    {
                        "input": new_design_config_dict["in_enc_dict"],
                        "output": new_design_config_dict["out_enc_dict"],
                    }
                )
                if ignore_already_existing or (new_encoding_dict_str not in known_encodings_str_set):
                    # `ignore_already_existing` is for debug purpose mostly
                    # It enforces generation of new designs even if they already exist in list of generated designs
                    new_design_count += 1

                    # Extend the list of new designs and associated encoding dictionnaries
                    new_config_dicts_out.append(new_design_config_dict)

                    # Add the encoding dictionnary to list of known encoding to avoid generating the same design twice
                    known_encodings_str_set.add(new_encoding_dict_str)

                    # Add the design encoding and associated final path to output lists
                    if do_encodings_only:
                        new_design_number = avail_new_design_numbers.pop()
                        new_design_number = new_design_number.zfill(self.zfill_len)
                        final_path = output_dir_path / f"res_{new_design_number}"
                        final_paths_out.append(final_path)
                    else:
                        final_paths_out.append(gener_filepaths[idx].parents[1])

                    # If we do the copy now, do it
                    # if not do_encodings_only:
                    # to_copy_list.append((new_design_path.parents[1], final_path))

        end_time = time()
        logger.info(
            f"Successfully generated a total of {len(final_paths_out)} new designs in {(end_time - start_time) / 60:.2f}min."
        )
        logger.info(
            f"Design numbers range is: {str(min_number).zfill(self.zfill_len)}-{str(max_number).zfill(self.zfill_len)}"
        )

        if self.current_iter_nb != 0:
            logger.info(f"Iteration number {self.current_iter_nb}")

        return final_paths_out, new_config_dicts_out

    # def _perform_generation_copy(self,filepath_lists):
    #     return utils.perform_parallel_copy(filepath_lists, nb_workers=self.nb_workers)

    def perform_design_files_generation(
        self,
        all_design_configs_list: list[dict[str, Any]],
        design_numbers: list[str],
        output_dir_path: Path | None = None,
    ) -> tuple[list[Path], list[dict[str, Any]]] | list[Path]:
        """This function generates all the"""

        nb_config_dicts = len(all_design_configs_list)

        # Setup the list of design numbers to use
        if nb_config_dicts < len(design_numbers):
            logger.warning(
                f"The list of configuration dictionnaries is smaller than the list of design numbers provided."
            )
        _design_numbers_to_make = []
        for design_number in design_numbers[:nb_config_dicts]:
            _design_numbers_to_make.append(design_number.zfill(self.zfill_len))

        # Prepare all output directories
        gener_filepath_list = self.__prepare_all_gener_dir_path(_design_numbers_to_make, output_dir_path)
        assert nb_config_dicts == len(gener_filepath_list), f"{nb_config_dicts} != {len(gener_filepath_list)}"

        # Generate all output verilog files (substitute template)
        logger.info(f"Generating all designs from template files {self.template_verilog_filepaths}")
        with tqdm(
            total=nb_config_dicts, desc=f"x{self.nb_workers}|Design generation from template"
        ) as pbar:  # Progress bar
            gener_filepaths_list = utils.process_pool_helper(
                func=DesignGenerator._substitute_template_with_design_configuration,
                func_args_gen=(
                    (design_config_dict, copy(self.design_templates), gener_dir_path, self._db_columns)
                    for design_config_dict, gener_dir_path in zip(all_design_configs_list, gener_filepath_list)
                ),
                max_workers=self.nb_workers,
                pbar=pbar,
            )

        return gener_filepaths_list, all_design_configs_list

    def generate_all_designs(
        self,
        nb_to_generate: int,
        design_numbers: list[str | int] = None,
        output_dir_path: Path | None = None,
        skip_files_generation: bool = False,
        existing_encodings_dicts: list[dict[int, str]] | None = None,
        seed_offset: int = 1,
    ) -> tuple[list[Path] | None, list[dict[str, Any]]]:
        """
        This function 1) generate new design configurations dictionnaries (i.e. encodings) 2) prepare the generation directories and copy the verilog files and substituted templates (LUTs, etc.)
        Args:
         - nb_to_generate: number of designs to generate
         - design_numbers: the list of design_number that will be associate to the list of generated encodings. These values will be used as directory names `res_<design_number>`
         - output_dir_path: where to prepare the directories. This path should be a directory and will be populated with all generated designs.
         - skip_files_generation: whether to not do file generation and simply return the list of configuration dictionnary
         - seed_offset: seed offset used when randomly permuting a sample.
        """

        if nb_to_generate is None:
            raise ValueError(
                "`nb_to_generate` should be defined at this stage. It should be set as standalone, or should match the lengths of either the `design_numbers` or `existing_encodings_dicts` lists."
            )
            # nb_to_generate = math.factorial(6)

        logger.info(
            f"Generating {nb_to_generate} representation permutations with file_generation set to: {not skip_files_generation}"
        )

        # Setup which representation  permuted
        if self.args_dict.get("standard_encoding_only", False):
            # Do not permute anything because we want the standard encoding
            permute_in = False
            permute_out = False
        else:
            if self.exp_config_dict["design_type"] == "encoder":
                # Encoders take standard encoding in input and return custom encodings in output
                if self.exp_config_dict["output_encoding_type"] == "twos_comp":
                    permute_in = False
                    permute_out = True
            elif self.exp_config_dict["design_type"] == "decoder":
                # Decoders take custom encodings in input and return standard encoding in output
                if self.exp_config_dict["input_encoding_type"] == "one_hot_msbl":
                    permute_in = False
                    permute_out = False
                    raise NotImplementedError()
            elif self.exp_config_dict["design_type"] in ["adder", "multiplier"]:
                if (
                    self.exp_config_dict["input_encoding_type"] == "twos_comp_both"
                    and self.exp_config_dict["output_encoding_type"] == "twos_comp_both"
                ):
                    # Build the design so that both the input and output encodings are permuted (full custom design)
                    permute_in = True
                    permute_out = True
                else:
                    permute_in = True
                    permute_out = False
            else:
                permute_in = True
                permute_out = False

        # Prepare all encodings dictionaries
        all_design_configs_list = self._get_all_design_configuration_dictionnaries(
            nb_to_generate=nb_to_generate,
            exp_config_dict=self.exp_config_dict,
            permute_in=permute_in,
            permute_out=permute_out,
            existing_encodings_path=self.existing_encodings_path,
            existing_encoding_dicts=existing_encodings_dicts,
            seed=self.seed + seed_offset,
        )

        # Update zfill length
        self.zfill_len = DesignGenerator._get_zfill_len(
            exp_config=self.exp_config_dict, permute_in=permute_in, permute_out=permute_out
        )
        logger.info(f"zfill length has been set to {self.zfill_len}")

        # Setup the number of directories to generate
        # if self.is_debug:
        # end = self.nb_design #TODO: change this?? (now that we can give nb_design to generate design command)
        # all_design_configs_list = all_design_configs_list[:end]

        if not skip_files_generation:
            logger.info(
                f"{len(all_design_configs_list)} design configurations have been generated. File generation will start now."
            )
            return self.perform_design_files_generation(all_design_configs_list, design_numbers, output_dir_path)
        else:
            logger.info(
                f"{len(all_design_configs_list)} design configurations have been generated. File generation have been skipped."
            )
            return None, all_design_configs_list

    @staticmethod
    def _extend_text_with_encoding_comment(text: str, design_config_dict: dict):
        """Type should be `input` or `output`"""
        if "input" not in design_config_dict.keys() and "output" not in design_config_dict.keys():
            keys = ["in_enc_dict", "out_enc_dict"]
        else:
            keys = ["input", "output"]

        text += "\n"
        for type in keys:
            text += f"// {type} Value Encoding\n"
            for value, encoding in design_config_dict[type].items():
                text += f"// {value} -> {encoding}\n"
        return text

    @staticmethod
    def _substitute_template_with_design_configuration(
        design_config_dict: dict[str, dict[int, str]],
        design_templates: list[tuple[str, Template]],
        gener_dir_path: Path,
        columns: list[str],
    ):
        # Get all the keys present in the design templates
        all_keys = set()
        for filename, design_template in design_templates:
            keys = re.findall(design_template.pattern, design_template.template)
            keys = [key[1] for key in keys]
            all_keys.update(keys)

        files_to_write = []
        if "look_up_table" in all_keys:
            truth_table = DesignGenerator.get_truth_table(design_config_dict=design_config_dict, columns=columns)

            verilog_table, complexity_info_dict = AdaptedFlowyHdlGenerator(truth_table).main_genial(do_optimize=False)

            if verilog_table is None:
                verilog_table = DesignGenerator.convert_to_verilog(truth_table)
                assert isinstance(complexity_info_dict, dict)

            for filename, design_template in design_templates:
                verilog_code = design_template.substitute(
                    {
                        "look_up_table": verilog_table,
                        "in_bitwidth": design_config_dict["in_bitwidth"],
                        "out_bitwidth": design_config_dict["out_bitwidth"],
                    }
                )

                if "mydesign_comb.v.temp" == filename:
                    # Add encoding as comment in verilog code
                    verilog_code = DesignGenerator._extend_text_with_encoding_comment(verilog_code, design_config_dict)

                design_filepath = (gener_dir_path / filename).with_suffix(".v")
                files_to_write.append((design_filepath, verilog_code))

                if complexity_info_dict is not None:
                    design_filepath = gener_dir_path.parent / "lut_complexity_info.json"
                    json.dump(complexity_info_dict, design_filepath.open("w"))

        elif "localparams" in all_keys:
            localparams = DesignGenerator.generate_local_params(design_config_dict=design_config_dict)

            for filename, design_template in design_templates:
                verilog_code = design_template.substitute(
                    {
                        "localparams": localparams,
                        "in_bitwidth": design_config_dict["in_bitwidth"],
                        # "out_bitwidth": design_config_dict["out_bitwidth"]
                    }
                )

                if "mydesign_comb.v.temp" == filename:
                    # Add encoding as comment in verilog code
                    verilog_code = DesignGenerator._extend_text_with_encoding_comment(verilog_code, design_config_dict)

                design_filepath = (gener_dir_path / filename.partition(".")[0]).with_suffix(".v")
                files_to_write.append((design_filepath, verilog_code))

        for design_filepath, verilog_code in files_to_write:
            FileCompressionHandler.write_compressed_file(design_filepath, verilog_code)

        return design_filepath

    # @staticmethod
    # def _substitute_template_with_design_configuration(
    #     design_config_dict: dict[str, dict[int, str]],
    #     design_templates: list[tuple[str, Template]],
    #     gener_dir_path: Path,
    #     columns: list[str],
    # ):

    #     try:
    #         truth_table = DesignGenerator.get_truth_table(design_config_dict=design_config_dict, columns=columns)
    #         generator =
    #     except:
    #         return DesignGenerator._substitute_template_with_design_configuration_noflowy(*args, **kwargs)

    @staticmethod
    def _get_all_permutations(input_list: list[str], nb: int, seed: int) -> list[str]:
        """This function generates all possible permutations of an input list of elements"""

        if math.factorial(len(input_list)) == nb:
            # Generate all permutations
            all_perms = list(
                itertools.permutations(
                    input_list,
                )
            )
        else:
            # Generate nb permutations
            randon_gen = np.random.default_rng(seed)

            count = 0
            all_perms = []
            all_perms_str = set()
            timeout = 1000
            while count < nb:
                new_permutation = randon_gen.permutation(
                    input_list,
                )
                new_perm_string = "".join(new_permutation)

                # Do some checks
                is_not_already_present = new_perm_string not in all_perms_str

                is_ok = is_not_already_present

                # Add the permutation to the list of permutations
                if is_ok:
                    all_perms_str.add(new_perm_string)
                    all_perms.append(new_permutation)
                    count += 1

                else:
                    timeout -= 1
                    if timeout == 0:
                        break
                    continue
            # all_perms.append(np.random.permutation(input_array,))

        return all_perms

    @staticmethod
    def make_one_hot_encodings(
        all_values: list[int], bitwidth: int, enc_type: str, **kwargs
    ) -> tuple[list[int], list[int]]:
        """Generates the one-hot encoding for the given value. The one-hot is centered around zero."""
        available_one_hot = ["one_hot_msbl", "one_hot_msbr"]
        assert enc_type in available_one_hot

        if enc_type == "one_hot_msbr":
            raise NotImplementedError(f"Encoding type {enc_type} has not been implemented yet.")

        # TODO: add msbr
        # add zero encodings capabilities where zero is encoded as full zeros (requires one bit less)
        # add possibility to truncate values based on bitwidth

        _values = []
        _representations = []
        if bitwidth == len(all_values):
            zeros_chars = ["0" for i in range(bitwidth)]
            for idx, value in enumerate(all_values):
                out_string = copy(zeros_chars)
                out_string[idx] = "1"
                out_string = "".join(out_string)

                _representations.append(out_string)
                _values.append(value)

        else:
            raise NotImplementedError(f"Ensure bitwidth is the same as the number of possible values.")

        return _values, _representations

    # def make_twos_comp_encodings(all_values=out_values_unique_sorted, bitwidth=out_bitwidth, enc_type=out_enc_type)

    @staticmethod
    def get_binary_representations(
        values_sorted: list[int], enc_type: str, bitwidth: int | str, port: Literal["input", "output"]
    ):
        """
        Returns the list of binary representations associated with the list of `values_sorted` provided in input.
        """

        if isinstance(bitwidth, str):
            if str.isdigit(bitwidth):
                bitwidth = int(bitwidth)
            else:
                raise TypeError(
                    f"`bitwidth` should be of type int or str but should ultimately represent an integer value."
                )

        valid_rep_count = 0
        if enc_type in ["twos_comp", "twos_comp_both", "unsigned_int", "unsigned_int_both", "mixed"]:
            representations = []
            _values = []
            for value in values_sorted:
                if enc_type in ["twos_comp", "twos_comp_both", "mixed"]:
                    representation = bin_ops.twos_complement(value=value, bitwidth=bitwidth)
                elif enc_type in ["unsigned_int", "unsigned_int_both"]:
                    representation = bin_ops.unsigned_integer(value=value, bitwidth=bitwidth)
                _values.append(value)

                if representation is not None:
                    representations.append(representation)
                    valid_rep_count += 1
                else:
                    logger.warning(f"A generated representation was None! for port {port}.")
                    logger.warning(
                        f"Probably because the value bitwidth could not be represented on the specified bitwidth?"
                    )
                    representations.append(None)
            # TODO: values are added anyway, so there will be an issue when the output encoding width is too small to encode all the values.
            # values_unique_sorted = _values

        elif enc_type in ["one_hot_msbl", "one_hot_msbr"]:
            values_unique_sorted, representations = DesignGenerator.make_one_hot_encodings(
                all_values=values_sorted, bitwidth=bitwidth, enc_type=enc_type
            )

        # Inform the user about the bit usage of the configuration
        # Note: twos_complement or unsined has the same available range length
        # Using here tc function because of legacy
        # min_val_included = bin_ops.min_value_tc(bitwidth)
        # max_val_included = bin_ops.max_value_tc(bitwidth)

        # This plot cannot be done anymore as it happens inside a loop
        # TODO: check why that is the case
        # _range = max_val_included - min_val_included
        # usage_ratio = valid_rep_count / (_range+1)

        return representations

    @staticmethod
    def generate_standard_encoding(exp_config_dict: dict[str, Any]) -> dict[str, Any]:
        """
        This function generation the desgin configuration for the classic encoding

        Args:
         - exp_config_dict: experiment configuration dictionnary. Must specify: design_type, encodings_types, bitwidths
        """

        classic_enc_config_dict = DesignGenerator._get_all_design_configuration_dictionnaries(
            nb_to_generate=1, exp_config_dict=exp_config_dict, permute_in=False, permute_out=False
        )

        return classic_enc_config_dict[0]

    @staticmethod
    def get_all_sorted_unique_values(
        in_enc_type: str, in_bitwidth: int | str, design_type: str, input_only: bool = False
    ) -> tuple[list[int]]:
        """
        This function returns the soted list of unique valid input and output values of the design simply based on the design's input bitwidth.
        Args:
         - in_enc_type: type of encodings
         - in_bitwidth:
         - design_type: which operator is the design
        """

        if isinstance(in_bitwidth, str):
            if str.isdigit(in_bitwidth):
                in_bitwidth = int(in_bitwidth)
            else:
                raise TypeError(
                    f"`in_bitwidth` should be of type int or str but should ultimately represent an integer value."
                )

        # Get the input values
        if in_enc_type in ["twos_comp", "twos_comp_both", "mixed"]:
            min_val_in_included = bin_ops.min_value_tc(in_bitwidth)
            max_val_in_included = bin_ops.max_value_tc(in_bitwidth)
        elif in_enc_type in [
            "unsigned_int",
            "unsigned_int_both",
        ]:
            min_val_in_included = bin_ops.min_value_uint(in_bitwidth)
            max_val_in_included = bin_ops.max_value_uint(in_bitwidth)
        else:
            raise NotImplementedError(f"Input encoding type {in_enc_type} not implemented.")

        in_values_unique_sorted = list(range(min_val_in_included, max_val_in_included + 1))

        # Get the output values
        if not input_only:
            out_values = []
            for a in in_values_unique_sorted:
                for b in in_values_unique_sorted:
                    if design_type == "adder":
                        out_values.append(a + b)
                    elif design_type == "multiplier":
                        out_values.append(a * b)
                    elif design_type == "encoder":
                        break
                    elif design_type == "decoder":
                        break

                if design_type == "encoder":
                    out_values.append(a)
                if design_type == "decoder":
                    out_values.append(a)

            out_values_unique_sorted = np.sort(np.unique(out_values)).tolist()
        else:
            out_values_unique_sorted = []

        values_dict = {
            "input": in_values_unique_sorted,
            "output": out_values_unique_sorted,
        }

        return values_dict

    @staticmethod
    def generate_permuted_representations(
        exp_config_dict: dict[str, Any],
        nb_to_generate: int,
        values_dict: dict[str, list[int]],
        permute_in: bool = True,
        permute_out: bool = False,
        seed: int = 1,
    ):
        """
        This function generates the list of binary representation (input and output), permuted when permute_<in/out> is True.
        The number of different permutations generated is defined by nb_to_generate.

        Returns:
            _type_: _description_
        """

        in_enc_type = exp_config_dict["input_encoding_type"]
        out_enc_type = exp_config_dict["output_encoding_type"]
        in_bitwidth = int(exp_config_dict["input_bitwidth"])
        out_bitwidth = int(exp_config_dict["output_bitwidth"])

        # Get the list of classic representations for both input and output values
        # NOTE: is some representations are None, it's because the value cannot be represented with the specified bitwidth and encoding type
        default_in_representations = DesignGenerator.get_binary_representations(
            values_sorted=values_dict["input"], enc_type=in_enc_type, bitwidth=in_bitwidth, port="input"
        )

        # Setup the number of designs to generate
        if nb_to_generate is None:
            max_nb_to_generate = math.factorial(8)
        else:
            if permute_in and not permute_out:
                max_nb_to_generate = min(nb_to_generate, math.factorial(2**in_bitwidth))
            elif permute_out and not permute_in:
                max_nb_to_generate = min(nb_to_generate, math.factorial(2**out_bitwidth))
            elif permute_out and permute_in:
                if exp_config_dict.get("permute_out_mode", None) is not None:
                    if exp_config_dict.get("permute_out_mode") in ["keep_in_repr", "random"]:
                        max_nb_to_generate = min(nb_to_generate, math.factorial(2**in_bitwidth))
                else:
                    # To avoid generating square designs, reduce number to generate per port type by squareroot
                    max_nb_to_generate = int(
                        math.sqrt(min(nb_to_generate, math.factorial(2**out_bitwidth), math.factorial(2**in_bitwidth)))
                    )
            else:
                max_nb_to_generate = 1
            logger.info(f"{max_nb_to_generate} designs will be generated.")

        # Generate all permutations of the input
        if permute_in:
            if exp_config_dict.get("permute_in_mode", None) is not None:
                logger.warning(
                    f"Generation of input encodings with `permute_in_mode`=={exp_config_dict.get('permute_in_mode')}"
                )
                if exp_config_dict.get("permute_in_mode") == "col_permutation_from_standard":
                    # Create only permutations of column-permuted standard encoding
                    input_repr_perms = get_all_col_permuted_list(default_in_representations)

                    # Sample `max_nb_to_generate` from input_repr_perms
                    np_rng = np.random.default_rng(global_vars.get("seed", 0))
                    input_repr_perms = np_rng.choice(input_repr_perms, size=max_nb_to_generate, replace=False)
                    input_repr_perms = [list(perm) for perm in input_repr_perms]
                elif exp_config_dict.get("permute_in_mode") == "classic_encoding":
                    input_repr_perms = [default_in_representations]
                else:
                    raise NotImplementedError(
                        f"Experiment configuration `permute_in_mode` {exp_config_dict.get('permute_in_mode')}"
                    )

            else:
                input_repr_perms = DesignGenerator._get_all_permutations(
                    default_in_representations, nb=max_nb_to_generate, seed=seed
                )

        else:
            input_repr_perms = [default_in_representations]

        # Generate all all possible binary representations of the output
        default_out_representations = DesignGenerator.get_binary_representations(
            values_sorted=values_dict["output_full"], enc_type=out_enc_type, bitwidth=out_bitwidth, port="output"
        )
        if permute_out:
            if exp_config_dict.get("permute_out_mode", None) is not None:
                output_repr_perms = []
                if exp_config_dict.get("permute_out_mode") == "keep_in_repr":
                    for input_repr in input_repr_perms:
                        # Include input encodings in the output encodings
                        output_repr_perms.append(
                            DesignGenerator._get_matched_output_representations(values_dict, input_repr)
                        )

                elif exp_config_dict.get("permute_out_mode") == "random":
                    # Generate purely random output encodings
                    output_repr_perms = DesignGenerator._get_random_output_permutation(
                        values_dict, default_out_representations, nb=len(input_repr_perms), seed=seed
                    )
                else:
                    raise NotImplementedError(
                        f"Experiment configuration `permute_out_mode` {exp_config_dict.get('permute_out_mode')} is not implemented."
                    )
            else:
                output_repr_perms = DesignGenerator._get_all_permutations(
                    default_out_representations, nb=max_nb_to_generate, seed=seed
                )
        else:
            default_out_representations = DesignGenerator.get_binary_representations(
                values_sorted=values_dict["output"], enc_type=out_enc_type, bitwidth=out_bitwidth, port="output"
            )
            output_repr_perms = [default_out_representations]

        permuted_representations = {
            "input": input_repr_perms,
            "output": output_repr_perms,
            "input_default": default_in_representations,
        }

        return permuted_representations

    @staticmethod
    def _get_matched_output_representations(
        values_dict: dict[str, list[int]], input_repr_list: list[str], fill_with: str = "0", repr_bit_width: int = None
    ):
        """
        This function generates the output representation that matches with the input representation. i.e. values which are share have the same representation + fill-in 0s or 1s.
        """

        if repr_bit_width is None:
            required_bitwidth = math.ceil(math.log2(len(values_dict["output"])))

        # Value lists
        input_values = set(values_dict["input"])
        output_values = set(values_dict["output"])

        # Find values which are shared
        shared_values = input_values.intersection(output_values)

        # Configuration for completing the input representation
        input_bitwidth = len(input_repr_list[0])
        nb_missing_bits = required_bitwidth - input_bitwidth
        shift_config = np.random.choice(list(range(nb_missing_bits + 1)), size=1).item()

        output_repr_set = set()
        output_repr_dict = dict()
        for value in shared_values:
            # Find the index of the shared value in the input representation
            input_repr_index = values_dict["input"].index(value)
            input_repr = input_repr_list[input_repr_index]

            # Create the output representation for this value
            output_repr = "".join(np.random.choice(["0", "1"], size=required_bitwidth, replace=True, p=[0.5, 0.5]))
            output_repr = output_repr[:shift_config] + input_repr + output_repr[shift_config + input_bitwidth :]

            output_repr_dict[value] = output_repr
            # assert
            output_repr_set.add(output_repr)

        # Complete the dictionary with remaining values, randomly generated
        for value in output_values - shared_values:
            is_ok = False
            while not is_ok:
                output_repr = "".join(np.random.choice(["0", "1"], size=required_bitwidth, replace=True, p=[0.5, 0.5]))
                if output_repr not in output_repr_set:
                    output_repr_dict[value] = output_repr
                    output_repr_set.add(output_repr)
                    is_ok = True

        # Sort output_repr_dict based on its keys
        output_repr_dict = dict(sorted(output_repr_dict.items(), key=lambda item: item[0]))

        # Return the values of the dictionary
        return list(output_repr_dict.values())

    @staticmethod
    def _get_random_output_permutation(
        values_dict: dict[str, list[int]], full_output_repr_list: list[str], nb: int = 1, seed: int = 0
    ):
        """
        This function generates a purely random output representation by permuting the given full_output_repr_list and taking as many vectors as there are values.
        """

        # Value lists
        nb_output_repr = len(set(values_dict["output"]))

        # Setup random generator
        random_gen = np.random.default_rng(seed)

        # Pick up a random selection of binary strings
        permutations = process_pool_helper(
            func=random_gen.choice,
            func_args_gen=((full_output_repr_list, nb_output_repr, False) for _ in range(nb)),
            pbar=None,
        )

        # Return the values of the dictionary
        return permutations

    @staticmethod
    def _get_zfill_len(exp_config: dict[str, Any], permute_in: bool, permute_out: bool) -> int:
        # Get the lists of valid and unique input and output values
        values_dict = DesignGenerator.get_all_sorted_unique_values(
            in_enc_type=exp_config["input_encoding_type"],
            in_bitwidth=int(exp_config["input_bitwidth"]),
            design_type=exp_config["design_type"],
        )

        # Get associated zfill_len
        if permute_in:
            tot_possible_designs = math.factorial(len(values_dict["input"]))
        else:
            tot_possible_designs = 1

        if permute_out:
            tot_possible_designs *= math.factorial(len(values_dict["output"]))
        else:
            tot_possible_designs *= 1

        zfill_len = min(len(str(tot_possible_designs)), 15)

        return zfill_len

    @staticmethod
    def _get_all_design_configuration_dictionnaries(
        exp_config_dict: dict[str, Any],
        nb_to_generate: int,
        permute_in: bool = True,
        permute_out: bool = False,
        existing_encodings_path: Path | None = None,
        existing_encoding_dicts: list[dict[int, str]] | None = None,
        seed: int = 1,
    ) -> list[dict[str, dict[int, str] | int]]:
        """
        This function returns the list of all possible input and output configurations.

        If `existing_encodings_path:Path` is not None, it will generate the designs configurations from the list of encodings that can be found in exsiting designs under `existing_encodings_path/generation_out` directory.
        If `existing_encoding_dicts:dict[str,dict]` is not None and contains the right keys `{"in_enc_dict", "out_enc_dict"}`, it will take these dictionnaries and build the design configuration dictionaries from them.

        The output list contains dictionaries.
        This dictionary represents the design configuration, which includes some information about bitwidth, encoding type, etc. but also the encoding dictionnaries as a whole.
        Keys of returned dictionnaries:
        [
            'design_type'
            'in_bitwidth'
            'out_bitwidth'
            'in_enc_type'
            'out_enc_type'
            'in_enc_dict'
            'out_enc_dict'
        ]
        """

        design_configuration_dict_template = {
            "design_type": exp_config_dict["design_type"],
            "in_enc_type": exp_config_dict["input_encoding_type"],
            "out_enc_type": exp_config_dict["output_encoding_type"],
            "in_bitwidth": int(exp_config_dict["input_bitwidth"]),
            "out_bitwidth": int(exp_config_dict["output_bitwidth"]),
        }

        # Do essential checks
        available_encoding_types = ConfigDir.__valid_encoding_types__
        if permute_in:
            assert design_configuration_dict_template["in_enc_type"] in available_encoding_types, (
                f"Input encoding sould be among the following types: {available_encoding_types}"
            )
        if permute_out:
            assert design_configuration_dict_template["out_enc_type"] in available_encoding_types, (
                f"Output encoding sould be among the following types: {available_encoding_types}"
            )

        valid_design_types = ConfigDir.__valid_design_types__
        assert design_configuration_dict_template["design_type"] in valid_design_types, (
            f"Generation supports designs of type: {valid_design_types}"
        )

        values_dict = DesignGenerator.get_all_sorted_unique_values(
            in_enc_type=design_configuration_dict_template["in_enc_type"],
            in_bitwidth=design_configuration_dict_template["in_bitwidth"],
            design_type=design_configuration_dict_template["design_type"],
        )

        if existing_encoding_dicts is not None:
            if not isinstance(existing_encoding_dicts, list):
                logger.error(
                    f"`existing_encoding_dicts` should have been a list of dictionnaries of dictionnaries. Received:"
                )
                logger.error(existing_encoding_dicts)
                raise AssertionError()

            design_configuration_dict_template["min_val_in_included"] = min(
                existing_encoding_dicts[0]["in_enc_dict"].keys()
            )
            design_configuration_dict_template["max_val_in_included"] = max(
                existing_encoding_dicts[0]["in_enc_dict"].keys()
            )

            design_configuration_dict_template["input_default"] = DesignGenerator.get_binary_representations(
                values_sorted=values_dict["input"],
                enc_type=design_configuration_dict_template["in_enc_type"],
                bitwidth=design_configuration_dict_template["in_bitwidth"],
                port="input",
            )

        else:
            # Get the lists of valid and unique input and output values

            # Get the list of all possible output values with the encoding type and output bitwidth specified
            out_values_dict = DesignGenerator.get_all_sorted_unique_values(
                in_enc_type=design_configuration_dict_template["out_enc_type"],
                in_bitwidth=design_configuration_dict_template["out_bitwidth"],
                design_type=design_configuration_dict_template["design_type"],
                input_only=True,
            )
            values_dict["output_full"] = out_values_dict["input"]

            # Log information about encoding efficiency
            output_values_percentage = len(values_dict["output"]) / len(values_dict["output_full"]) * 100
            if output_values_percentage < 100:
                logger.info(
                    f"Encoding Density: The output values required to describe the possible results of all input values are using only "
                    f"{output_values_percentage:.2f}% of the total possible output values with this output bitwidth."
                )

            # Memorize some information
            design_configuration_dict_template["min_val_in_included"] = min(values_dict["input"])
            design_configuration_dict_template["max_val_in_included"] = max(values_dict["input"])

            design_configuration_dict_template["min_val_out_included"] = min(values_dict["output"])
            design_configuration_dict_template["max_val_out_included"] = max(values_dict["output"])

            if existing_encodings_path is not None:
                # Encodings have been specified, we want generate them, so we just generate one encoding configuration to serve as template
                default_representations = DesignGenerator.generate_permuted_representations(
                    exp_config_dict=exp_config_dict,
                    nb_to_generate=1,
                    values_dict=values_dict,
                    permute_in=False,
                    permute_out=False,
                    seed=seed,
                )
                design_configuration_dict_template["input_default"] = default_representations["input_default"]
            else:
                # If no encodings dictionnary is given in input, we need to generate them (and so we need to generate the representation permutations)
                logger.info(f"Generating permuted representations ...")
                permuted_representations = DesignGenerator.generate_permuted_representations(
                    exp_config_dict=exp_config_dict,
                    nb_to_generate=nb_to_generate,
                    values_dict=values_dict,
                    permute_in=permute_in,
                    permute_out=permute_out,
                    seed=seed,
                )
                design_configuration_dict_template["input_default"] = permuted_representations["input_default"]

        # Build all design configurations dictionaries
        design_configurations_list = []
        if existing_encodings_path is not None:  # Used for generating encoders
            # Here, we first read all the encoding dictionnaries present in the generated designs found in `existing_encodings_path`
            known_encodings_dicts = file_parsers.read_all_existing_encoding(existing_encodings_path, type="generator")
            for key, known_encoding_dict in known_encodings_dicts.items():
                design_configuration = copy(design_configuration_dict_template)
                if exp_config_dict["design_type"] == "encoder":
                    design_configuration.update(
                        {
                            "in_enc_dict": {
                                k: v for k, v in zip(values_dict["input"], default_representations["input"][0])
                            },
                            "out_enc_dict": known_encoding_dict["input"],
                        }
                    )
                    design_configurations_list.append(design_configuration)
                else:
                    # TODO
                    # Probably simply design_configuration.update(known_encoding_dict), maybe with a change of keys (from input to in_enc_dict)
                    raise NotImplementedError()

        elif existing_encoding_dicts is not None:
            # Here we simply expand the default configuraiton dictionnary with the existing encoding dictionnary
            valid_encoding_dict_keys = {"in_enc_dict", "out_enc_dict"}
            for encoding_dict in existing_encoding_dicts:
                design_configuration = copy(design_configuration_dict_template)
                if set(encoding_dict.keys()) != valid_encoding_dict_keys:
                    logger.error(
                        f"Expected encoding dictionary keys to be {valid_encoding_dict_keys}, received {set(encoding_dict.keys())}"
                    )
                    raise KeyError()
                else:
                    design_configuration.update(encoding_dict)
                    design_configurations_list.append(design_configuration)

        else:
            # Here we take the generate list of permuted representation and we build all associated encoding dictionnary
            for in_idx, input_repr_perm in enumerate(permuted_representations["input"]):
                if exp_config_dict.get("permute_out_mode", None) is not None:
                    if exp_config_dict.get("permute_out_mode") in ["keep_in_repr", "random"]:
                        # Here, we want to keep alignement between input encodings and output encodings
                        _output_permuted_reprs = [permuted_representations["output"][in_idx]]
                else:
                    _output_permuted_reprs = permuted_representations["output"]

                for output_repr_perm in _output_permuted_reprs:
                    # Instantiate encoding dictionaries
                    input_encoding_dict = {int(k): str(v) for k, v in zip(values_dict["input"], input_repr_perm)}
                    output_encoding_dict = {int(k): str(v) for k, v in zip(values_dict["output"], output_repr_perm)}

                    # Setup design configuration dictionnary and update list of design configurations
                    design_configuration = copy(design_configuration_dict_template)
                    design_configuration.update(
                        {
                            "in_enc_dict": input_encoding_dict,
                            "out_enc_dict": output_encoding_dict,
                        }
                    )
                    design_configurations_list.append(design_configuration)

        logger.info(f"Generated {len(design_configurations_list)} design configurations.")

        return design_configurations_list

    def plot_values_distribution(self, values: list[int]) -> None:
        """Plot the histogram of the value distribution"""

        plotter.plot_values_distribution(
            values=values,
            experiment_name=self.dir_config.experiment_name,
            plot_filepath=self.plot_dir / "output_values_distribution.png",
        )

    @staticmethod
    def parse_args() -> dict[str, Any]:
        default_args_dict = get_default_parser()

        arg_parser = argparse.ArgumentParser(description="Run the design generator")
        arg_parser.add_argument("--nb_new_designs", type=int, default=None, help="How many new designs to generate")
        arg_parser.add_argument(
            "--standard_encoding_only",
            action="store_true",
            help="Ensure that the design generator generates a single design with standard encoding.",
        )
        arg_parser.add_argument(
            "--permute_in_mode",
            type=str,
            help="Which modes to use during permutation stage of representations generation.",
        )
        arg_parser.add_argument(
            "--permute_out_mode",
            type=str,
            help="Which modes to use during permutation stage of representations generation.",
        )

        args = arg_parser.parse_known_args()
        args_dict = vars(args[0])

        args_dict.update(default_args_dict)

        return args_dict

    def main(self, nb_to_generate: int | None = None):
        """Generate designs"""

        if nb_to_generate is None:
            _nb_new_designs = self.args_dict.get("nb_new_designs", None)
        else:
            _nb_new_designs = nb_to_generate

        logger.info(f"Generator main was called with nb designs to generate: {_nb_new_designs}")
        gener_dirpath_list, design_config_list = self.generate_more_designs(nb_designs=_nb_new_designs)
        # else:
        # genered_designs_list = self.generate_all_designs(nb_to_generate=nb_to_generate)
        logger.opt(colors=True).info(
            f"<red>Generation</red> of {len(gener_dirpath_list)} designs is <red>finished</red>"
        )

        return gener_dirpath_list

    def switch_to_iter_mode(self, config_dict: dict[str, Any], **kwargs) -> None:
        """Does nothing."""
        pass


def main_cli():
    start_time = time()
    try:
        # Initialize the design generator
        design_generator = DesignGenerator()
        design_generator.main()

        status = "Success"
        error_msg = ""
    except Exception:
        status = "Failed"
        error_msg = traceback.format_exc()

    logger.info(error_msg)
    design_generator.send_email(
        config_dict=design_generator.args_dict,
        start_time=start_time,
        status=status,
        error_message=error_msg,
        calling_module="Generator",
        root_output_dir=design_generator.dir_config.root_output_dir,
    )

    logger.info("DesignGenerator's `main_cli` exited properly.")


if __name__ == "__main__":
    main_cli()
