# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

import argparse
from copy import copy
from typing import Any

from genial.globals import global_vars
from swact.gates_configuration import GatesConfig


def get_default_parser() -> dict[str, Any]:
    arg_parser = argparse.ArgumentParser(description="Default Parser")
    arg_parser.add_argument(
        "--output_dir_name",
        default=None,
        type=str,
        help="Name of the output dir where to put the results. By default, all results are stored in `$WORK_DIR/output/<output_dir_name>`",
    )
    arg_parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the directory containing the experiment template files, check all available names in `src/templates_and_launch_scripts`. It MUST match with one of these. It will also be used a parent of the output directory. Example: multiplier_2bi_4bo_permuti_allcells_notech_fullsweep_only.",
    )
    arg_parser.add_argument(
        "--debug", action="store_true", help="[Launcher:] run a single iteration (not all designs) | [Analyzer]: None"
    )
    arg_parser.add_argument("--nb_workers", type=int, default=16, help="Number of workers to launch jobs in parallel")
    arg_parser.add_argument(
        "--design_number_list",
        nargs="*",
        type=str,
        default=None,
        help="List of device numbers to run. Only these designs will be treated.",
    )
    arg_parser.add_argument(
        "--existing_encodings_path",
        type=str,
        default=None,
        help="[Generator Only] Path where the generator will load the encoding dictionnaries instead of generating them.",
    )
    arg_parser.add_argument(
        "--synth_version", type=int, default=0, help="Select synthesis version for running the synthesis."
    )
    arg_parser.add_argument(
        "--swact_version", type=int, default=0, help="Select switching activity version for running the evaluation."
    )
    arg_parser.add_argument(
        "--power_version", type=int, default=0, help="Select version for running the power extraction."
    )
    arg_parser.add_argument(
        "--cell_cost_model",
        type=str,
        default="transistor",
        choices=["transistor", "capacitance", "capacitance_calibrated", "none"],
        help="Model used to weight fanout: transistor count, input capacitance, calibrated capacitance or none.",
    )
    arg_parser.add_argument(
        "--synth_only",
        action="store_true",
        help="Analyzer and Launcher will only run synthesis jobs.",
    )
    arg_parser.add_argument(
        "--cmplx_only",
        action="store_true",
        help="Analyzer and Launcher will only run complexity jobs.",
    )
    arg_parser.add_argument("--skip_synth", action="store_true", help="Skip synthesis analysis")
    arg_parser.add_argument("--skip_swact", action="store_true", help="Skip test switching activity analysis")
    arg_parser.add_argument("--skip_power", action="store_true", help="Skip power extraction analysis")
    arg_parser.add_argument("--skip_cmplx", action="store_true", help="Skip complexity analysis")
    arg_parser.add_argument(
        "--send_email", action="store_true", help="Send an e-mail to notify the user when the job is done."
    )
    arg_parser.add_argument(
        "--keep_not_valid",
        action="store_true",
        help="When set, designs folders that are not valid will not be deleted when the file parser checks their validity.",
    )
    arg_parser.add_argument("--seed", default=0, type=int, help="Random seed for any module")
    arg_parser.add_argument(
        "--bulk_flow_dirname",
        default=None,
        type=str,
        help="Name of the directory in which the full list of output files is sotred when running a bulk flow script.",
    )
    arg_parser.add_argument(
        "--technology",
        default="notech_yosys",
        type=str,
        help="Name of technology in which the design has been synthesized (for analysis).",
        choices=GatesConfig.configs.keys(),
    )
    arg_parser.add_argument(
        "--ignore_user_prompts", action="store_true", help="Ignore input prompt to user whenever they should be raised."
    )

    # Other potential args
    arg_parser.add_argument(
        "--other_output_dirpath",
        type=str,
        help="Absolute path to another output_dir sharing the same design IDs which have already been synthesized (potentially using another version).",
    )

    args = arg_parser.parse_known_args()
    args_dict = vars(args[0])

    if args_dict["synth_only"]:
        args_dict["skip_swact"] = True
        args_dict["skip_power"] = True
        args_dict["skip_cmplx"] = True

    if args_dict["cmplx_only"]:
        args_dict["skip_swact"] = True
        args_dict["skip_power"] = True
        args_dict["skip_synth"] = True

    global_vars["debug"] = args_dict["debug"]
    global_vars["seed"] = args_dict["seed"]
    global_vars["keep_not_valid"] = args_dict["keep_not_valid"]
    global_vars["nb_workers"] = args_dict["nb_workers"]

    if args_dict["bulk_flow_dirname"] is not None:
        global_vars["is_bulk_flow_mode"] = True

    return args_dict


def analyzer_parser() -> dict[str, Any]:
    default_args_dict = get_default_parser()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--rebuild_db",
        action="store_true",
        help="Erase the existing database if it exists",
    )
    arg_parser.add_argument(
        "--continue",
        action="store_true",
        help="Extend the existing database with all the designs that have been both correctly synthesized and correctly tested.",
    )
    arg_parser.add_argument("--skip_plots", action="store_true", help="Skip plot realization")
    arg_parser.add_argument(
        "--fast_plots",
        action="store_true",
        help="Do only a few of the plots for quick check up. `skip_plots` option will be ignored",
    )
    arg_parser.add_argument(
        "--skip_fullsweep_analysis",
        action="store_true",
        help="Skip detailed SwAct analysis (worst cases and zero cases with fullsweep test)",
    )
    arg_parser.add_argument(
        "--skip_tests_list",
        nargs="*",
        type=str,
        default=[],
        help="Tests that should be skipped during plotting operations.",
    )
    arg_parser.add_argument("--interactive", action="store_true", help="Do plots in interactive mode.")

    args = arg_parser.parse_known_args()
    args_dict = vars(args[0])

    args_dict.update(default_args_dict)

    return args_dict


def double_analyzer_parser() -> dict[str, Any]:
    analyzer_args_dict = analyzer_parser()
    arg_parser = argparse.ArgumentParser()

    # arg_parser.add_argument("--experiment_name_0", type=str, default="adder_3b_permutations_allcells_notech", help="Folder name where the experiment files and scripts are stored.")
    arg_parser.add_argument(
        "--output_dir_name_0",
        default=None,
        type=str,
        help="Name of the output dir where to put the results. By default, all results are stored in `$WORK_DIR/output/<output_dir_name>`",
    )
    # arg_parser.add_argument("--experiment_name_1", type=str, default="adder_3b_permutations_allcells_notech", help="Folder name where the experiment files and scripts are stored.")
    arg_parser.add_argument(
        "--output_dir_name_1",
        default=None,
        type=str,
        help="Name of the output dir where to put the results. By default, all results are stored in `$WORK_DIR/output/<output_dir_name>`",
    )
    arg_parser.add_argument(
        "--rebuild_meta_db",
        action="store_true",
        help="When set, it enforces that the script builds the database use for meta data analysis.",
    )

    args = arg_parser.parse_known_args()
    args_dict = vars(args[0])
    args_dict.update(analyzer_args_dict)

    args_dict["rebuild_db"] = False  # Ensure analyzer DBs do not get overwritten

    args_dict.pop("output_dir_name")
    # args_dict.pop("experiment_name")

    args_0 = copy(args_dict)
    args_0.pop("output_dir_name_1")
    # args_0.pop("experiment_name_1")
    # args_0["experiment_name"] = args_0.pop("experiment_name_0")
    args_0["output_dir_name"] = args_0.pop("output_dir_name_0")

    args_1 = copy(args_dict)
    args_1.pop("output_dir_name_0")
    # args_1.pop("experiment_name_0")
    # args_1["experiment_name"] = args_1.pop("experiment_name_1")
    args_1["output_dir_name"] = args_1.pop("output_dir_name_1")

    return args_0, args_1, args_dict


def sampling_analyzer_parser() -> dict[str, Any]:
    analyzer_args_dict = analyzer_parser()
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--output_dir_name",
        default=None,
        type=str,
        help=(
            "Name of the output dir where to put the results. "
            "By default, all results are stored in `$WORK_DIR/output/<output_dir_name>`"
        ),
    )

    args = arg_parser.parse_known_args()
    args_dict = vars(args[0])
    args_dict.update(analyzer_args_dict)

    args_dict["rebuild_db"] = False  # Ensure analyzer DBs do not get overwritten

    return args_dict
