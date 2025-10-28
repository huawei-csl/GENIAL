from pathlib import Path
from typing import Any

from genial.experiment.file_parsers import get_encoding_dict_from_file
from genial.experiment.task_generator import DesignGenerator

from genial.config.config_dir import ConfigDir

from genial.ext_plugins.flowy.hdl_generator import AdaptedFlowyHdlGenerator

from loguru import logger

import argparse


import genial.experiment.binary_operators as bops


def get_complexity_dict_from_file(design_filepath: Path, exp_config: dict[str:Any], print_only: bool = False):
    """
    This function exploit the file parser to get the design encoding from a file.
    It then rebuild the LUT out of the encoding, realying on the design generator class methods.
    And using the adapted HDL generator to extract the complexity of the design.
    """

    lut_build_dict = get_encoding_dict_from_file(design_filepath)

    lut_build_dict["design_type"] = exp_config["design_type"]

    in_bitwidth = int(exp_config["input_bitwidth"])
    if exp_config["input_encoding_type"] == "twos_comp":
        lut_build_dict["min_val_in_included"] = bops.min_value_tc(in_bitwidth)
        lut_build_dict["max_val_in_included"] = bops.max_value_tc(in_bitwidth)
    elif exp_config["input_encoding_type"] == "unsigned":
        lut_build_dict["min_val_in_included"] = bops.min_value_uint(in_bitwidth)
        lut_build_dict["max_val_in_included"] = bops.max_value_uint(in_bitwidth)
    else:
        logger.error(f"Unknown input encoding type: {exp_config['input_encoding_type']}")

    columns = [
        "input_a_val",
        "input_b_val",
        "output_val",
        "input_a_rep",
        "input_b_rep",
        "output_rep",
    ]

    print(lut_build_dict["in_enc_dict"])
    lut = DesignGenerator.get_truth_table(lut_build_dict, columns)
    optimized_lut, complexity_dict = AdaptedFlowyHdlGenerator(lut).main_genial(do_optimize=False)

    if print_only:
        logger.info(f"Complexity dictionnary of file {design_filepath}:")
        logger.info(complexity_dict)
        return None
    else:
        return complexity_dict


def main_cli():
    dir_config = ConfigDir()
    exp_config = dir_config.exp_config

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        f"--design_filepath",
        type=str,
        help="Path to the design file containing the encoding dictionnary. (verilog, compressed verilog or json file).",
    )
    args = arg_parser.parse_known_args()[0]

    design_filepath = Path(args.design_filepath)

    get_complexity_dict_from_file(design_filepath, exp_config=exp_config, print_only=True)

    logger.info(f"Execution Successful.")


if __name__ == "__main__":
    main_cli()
