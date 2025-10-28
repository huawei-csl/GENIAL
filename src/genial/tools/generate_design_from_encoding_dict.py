from genial.config.config_dir import ConfigDir
from genial.experiment.file_parsers import get_encoding_dict_from_file, get_list_of_gener_designs_number
from genial.experiment.task_generator import DesignGenerator

import argparse
from pathlib import Path

from loguru import logger


def main_cli():
    # Setup config dir object
    dir_config = ConfigDir()

    # Setup argument parser and parse args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--enc_dict_path",
        type=str,
        help=(
            "Path to the file containing the encoding dictionnary. It can be a file endin in '.json', '.v' or '.v.bz2'"
        ),
        required=True,
    )
    args = arg_parser.parse_known_args()[0]

    # Get the encoding dictionnary
    lut_build_dict = get_encoding_dict_from_file(Path(args.enc_dict_path))
    assert "in_enc_dict" in lut_build_dict, "Encoding dictionnary must contain an 'in_enc_dict' key"
    # assert "out_enc_dict" in lut_build_dict, "Encoding dictionnary must contain an 'out_enc_dict' key"
    if "out_enc_dict" not in lut_build_dict:
        values_dict = DesignGenerator.get_all_sorted_unique_values(
            in_enc_type=dir_config.exp_config["input_encoding_type"],
            in_bitwidth=int(dir_config.exp_config["input_bitwidth"]),
            design_type=dir_config.exp_config["design_type"],
        )
        out_values_dict = DesignGenerator.get_all_sorted_unique_values(
            in_enc_type=dir_config.exp_config["output_encoding_type"],
            in_bitwidth=int(dir_config.exp_config["output_bitwidth"]),
            design_type=dir_config.exp_config["design_type"],
            input_only=True,
        )
        values_dict["output_full"] = out_values_dict["input"]
        default_representations = DesignGenerator.generate_permuted_representations(
            dir_config.exp_config,
            nb_to_generate=1,
            values_dict=values_dict,
            permute_in=False,
            permute_out=False,
        )

        lut_build_dict["out_enc_dict"] = {
            k: v for k, v in zip(values_dict["output"], default_representations["output"][0])
        }

        lut_build_dict["in_enc_dict"] = {
            int(k): v for k, v in zip(values_dict["input"], default_representations["input"][0])
        }

    # Build the associted design
    design_generator = DesignGenerator(dir_config=dir_config)

    # Generate the designs
    design_numbers = get_list_of_gener_designs_number(dir_config)
    design_numbers_int = [int(design_number) for design_number in design_numbers]

    if len(design_numbers_int) == 0:
        max_known_design_number = -1
    else:
        max_known_design_number = max(design_numbers_int)

    nb_designs = 1
    new_design_numbers = [str(max_known_design_number + 1)]

    generated_design_paths_list, _ = design_generator.generate_all_designs(
        nb_to_generate=nb_designs,
        design_numbers=new_design_numbers,
        output_dir_path=dir_config.generation_out_dir,
        existing_encodings_dicts=[lut_build_dict],
    )

    # logger.info(f"Generated design with encoding {lut_build_dict}")
    logger.info(f"Generated design paths: {generated_design_paths_list}")
    logger.info(f"Done.")


if __name__ == "__main__":
    main_cli()
