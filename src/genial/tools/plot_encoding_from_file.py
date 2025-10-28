from pathlib import Path
from typing import Any


from loguru import logger

import argparse


import genial.experiment.plotter as plotter
from genial.experiment.file_parsers import get_encoding_dict_from_file

import matplotlib.pyplot as plt


def get_and_plot_encoding_from_file(
    design_filepath: Path, bitwdith: dict[str:Any], print_only: bool = False, encoding_type: str = "input"
):
    """
    This function exploit the file parser to get the design encoding from a file.
    It then rebuild the LUT out of the encoding, realying on the design generator class methods.
    And using the adapted HDL generator to extract the complexity of the design.
    """

    lut_build_dict = get_encoding_dict_from_file(design_filepath)

    if encoding_type == "input":
        key = "in_enc_dict"
    elif encoding_type == "output":
        key = "out_enc_dict"
    else:
        raise ValueError(f"Encoding type {encoding_type} not supported")

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    plotter.plot_encoding_heatmap_solo(
        ax=axes,
        encoding_str=str(lut_build_dict[key]),
        design_number="__(^^)__",
        bitwidth=bitwdith,
        port_type="input",
        ax_title=f"",
    )
    filepath = design_filepath.parent / "encoding_representation.png"
    plt.savefig(filepath)
    plt.close()

    logger.info(f"Visualization of the {encoding_type} encoding saved in {filepath}")

    return None


def main_cli():
    # dir_config = ConfigDir()
    # exp_config = dir_config.exp_config

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        f"-f",
        f"--design_filepath",
        type=str,
        help="Path to the design file containing the encoding dictionnary. (verilog, compressed verilog or json file).",
    )
    arg_parser.add_argument(
        f"-t",
        f"--encoding_type",
        type=str,
        help="input or output",
        default="input",
    )
    arg_parser.add_argument(
        f"-b",
        f"--bitwidth",
        type=int,
        help="Bitwidth of the encoding to visualize",
        default=4,
    )
    args = arg_parser.parse_known_args()[0]

    design_filepath = Path(args.design_filepath)

    get_and_plot_encoding_from_file(
        design_filepath, bitwdith=args.bitwidth, print_only=True, encoding_type=args.encoding_type
    )

    logger.info(f"Execution Successful.")


if __name__ == "__main__":
    main_cli()
