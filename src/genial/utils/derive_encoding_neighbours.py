import argparse
from typing import Any
import pandas as pd
from itertools import combinations


from loguru import logger

from genial.config.config_dir import ConfigDir
from genial.experiment.task_analyzer import analyzer_parser
from genial.experiment import file_parsers
from genial.experiment.task_generator import DesignGenerator


def parse_args() -> dict[str, Any]:
    # Add arguments for this script
    arg_parser = argparse.ArgumentParser()
    help_message = "Number of top encodings to generate neighbours for."
    arg_parser.add_argument("--top_n", type=int, default=10, help=help_message)

    # Will be used for any kind of generation
    arg_parser.add_argument(
        "--dst_output_dir_name",
        type=str,
        default=None,
        help="If set, the designs will be copied to this output directory. A standard directory structure will be setup by instantiating a directory configuration object first.",
    )

    args_dict = vars(arg_parser.parse_known_args()[0])

    # Add standard analyzer arguments
    default_args_dict = analyzer_parser()
    args_dict.update(default_args_dict)

    return args_dict


def hamming_distance(a: int, b: int) -> int:
    """
    Returns the hamming distance between two integers.
    """
    return bin(a ^ b).count("1")


def hamming_2_neighbors(int_encodings_input_dict: dict[int, int]) -> list[dict]:
    keys = list(int_encodings_input_dict.keys())
    neighbors = []
    # Check all the possible combinations of encodings
    for i, j in combinations(range(len(keys)), 2):
        ki, kj = keys[i], keys[j]
        # Check if the hamming distance between two encodings is one.
        if hamming_distance(int_encodings_input_dict[ki], int_encodings_input_dict[kj]) == 1:
            # Swap values
            new_seq = int_encodings_input_dict.copy()
            new_seq[ki], new_seq[kj] = new_seq[kj], new_seq[ki]
            # Convert back to string
            # TODO: change 04b to bitwidth dependent value (add bitwidth as argument to function)
            new_seq = {k: format(v, "04b") for k, v in new_seq.items()}
            neighbors.append(new_seq)
    return neighbors


def derive_neighbours(dir_config: ConfigDir, top_n: int, derive_inverse=True):
    # Load analysis df.
    analysis_df = pd.read_parquet(dir_config.analysis_out_dir / "synth_analysis.db.pqt")

    # Sort by increasing transistor count.
    analysis_df = analysis_df.sort_values("nb_transistors").drop_duplicates("encodings_input").reset_index(drop=True)

    # Create a list to store the prototypes.
    proto_encoding_dicts_list = []
    # Loop through the top n encodings from the analysis df.
    for i, row in analysis_df.iloc[:top_n].iterrows():
        # Retrieve the input and output dictionaries.
        encodings_input_dict = eval(row["encodings_input"])
        encodings_output_dict = eval(row["encodings_output"])
        # Convert the string encoding to an integer representation for faster compute.
        int_encodings_input_dict = {k: int(v, 2) for k, v in encodings_input_dict.items()}
        # Derive all the neighbours with hamming distance of 2 to the input encoding sequence.
        new_neighbours_list = hamming_2_neighbors(int_encodings_input_dict)
        # Add all the encoding input found with the template output encoding
        proto_encoding_dicts_list += [
            {"in_enc_dict": d, "out_enc_dict": encodings_output_dict} for d in new_neighbours_list
        ]
        # To derive the inverse of the encoding and its neighbours
        if derive_inverse:
            # Derive the inverse of the encoding
            inverse_int_encodings_input_dict = {k: v ^ 0b1111 for k, v in int_encodings_input_dict.items()}
            # Derive all the neighbours with hamming distance of 2 to the inverse input encoding sequence.
            inverse_neighbours_list = hamming_2_neighbors(inverse_int_encodings_input_dict)
            # Add the inverse in the bit string format and the neighbours in the same list of dictionaries.
            # TODO: change 04b to bitwidth dependent value (add bitwidth as funciton argument)
            inverse_plus_neighbours_list = [
                *[{k: format(v, "04b") for k, v in inverse_int_encodings_input_dict.items()}],
                *inverse_neighbours_list,
            ]
            # Add the inverse encoding input and its neighbours with the template output encoding
            proto_encoding_dicts_list += [
                {"in_enc_dict": d, "out_enc_dict": encodings_output_dict} for d in inverse_plus_neighbours_list
            ]
    return proto_encoding_dicts_list


def main():
    # Parse args
    args_dict = parse_args()
    # Create dir config object
    dir_config = ConfigDir(is_analysis=True, **args_dict)

    # Update args dict for generation. Prototype data and generated design will be stored there.
    gener_args_dict = dir_config.args_dict
    if dir_config.args_dict.get("dst_output_dir_name") is not None:
        gener_args_dict.update({"output_dir_name": dir_config.args_dict.get("dst_output_dir_name")})

    # Setup Generation
    gener_dir_config = ConfigDir(is_analysis=False, **gener_args_dict)

    # Derive the neighbours in a list of dictionary format
    proto_encoding_dicts_list = derive_neighbours(dir_config=dir_config, top_n=gener_args_dict.get("top_n"))

    # Remove duplicates
    prototype_df = pd.DataFrame([{"proto_str": str(p)} for p in proto_encoding_dicts_list])
    index_to_keep = set(prototype_df.drop_duplicates().index.tolist())
    proto_encoding_dicts_list = [p for i, p in enumerate(proto_encoding_dicts_list) if i in index_to_keep]

    # Find which design numbers to use
    nb_protos = len(proto_encoding_dicts_list)
    design_numbers = file_parsers.get_list_of_gener_designs_number(gener_dir_config)
    design_numbers_int = [int(design_number) for design_number in design_numbers]

    if len(design_numbers_int) == 0:
        max_known_design_number = -1
    else:
        max_known_design_number = max(design_numbers_int)
    new_design_numbers = [str(i) for i in range(max_known_design_number + 1, max_known_design_number + 1 + nb_protos)]

    # Actually generate the prototypes
    design_generator = DesignGenerator(gener_dir_config)
    logger.info(f"Generating new designs, starting with design_number {max_known_design_number + 1}")
    generated_design_paths_list, generated_design_config_dicts_list = design_generator.generate_all_designs(
        nb_to_generate=nb_protos,
        design_numbers=new_design_numbers,
        output_dir_path=gener_dir_config.generation_out_dir,
        existing_encodings_dicts=proto_encoding_dicts_list,
    )


if __name__ == "__main__":
    main()
