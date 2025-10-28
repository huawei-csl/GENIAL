import argparse
from typing import Any
import pandas as pd
import torch
import time

from loguru import logger

from genial.config.config_dir import ConfigDir
from genial.experiment.task_analyzer import analyzer_parser
from genial.experiment import file_parsers
from genial.experiment.task_generator import DesignGenerator


def parse_args() -> dict[str, Any]:
    # Add arguments for this script
    arg_parser = argparse.ArgumentParser()
    help_message = "Number of initial samples to generate"
    arg_parser.add_argument("--n_samples", type=int, default=20_000, help=help_message)

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


def derive_diverse_initial_dataset(
    dir_config: ConfigDir, n_samples: int, device: str, batch_size: int = 5000, candidates_per_iter: int = 1000
):
    # Two's complement encoding dictionary to use as template
    # TODO: use the following:
    # classic_encoding = DesignGenerator._get_all_design_configuration_dictionnaries(
    #        nb_to_generate=1, exp_config_dict=dir_config.exp_config, permute_in=False, permute_out=False
    #    )[0]
    # tc_enc = classic_encoding["in_enc_dict"]
    tc_enc = {
        -8: "1000",
        -7: "1001",
        -6: "1010",
        -5: "1011",
        -4: "1100",
        -3: "1101",
        -2: "1110",
        -1: "1111",
        0: "0000",
        1: "0001",
        2: "0010",
        3: "0011",
        4: "0100",
        5: "0101",
        6: "0110",
        7: "0111",
    }
    # Extract the keys (as a list) and the values (as a torch boolean array)
    keys = list(tc_enc.keys())
    values = torch.stack([torch.tensor([int(c) for c in v], dtype=torch.bool) for v in tc_enc.values()]).to(device)

    # Set the permutation with the seed
    seed = 42
    g = torch.Generator()
    g.manual_seed(seed)

    # Add the first sample to the dataset
    dataset = (values[torch.randperm(values.size(0), generator=g)]).flatten().unsqueeze(0).to(device)

    # Initialise the dataset sample counter
    count = 1

    # Time tracking
    time_take = time.time()
    samples_between_logging = 1000

    # Add one sample at a time
    while count < n_samples:
        # Generate candidates for to add to dataset
        permutations = torch.stack(
            [values[torch.randperm(values.size(0), generator=g)].flatten() for _ in range(candidates_per_iter)]
        )

        # Obtain the distances between the candidates and the dataset samples.
        dist_list = []
        # Loop through the dataset in batch.
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            # Get the distance between the batch and the candidates.
            xor = batch[:, None, :] ^ permutations[None, :, :]
            # Get the minimum distance between each candidate and any sample in the batch
            min_dist_batch = xor.sum(dim=2).min(dim=0)[0]
            # Append distance for the batch.
            dist_list.append(min_dist_batch)
        # Derive min distance for each candidate and any sample in the dataset.
        min_dist_all = torch.stack(dist_list).min(dim=0)[0]
        # Retrieve the max distance in candidates (for logging purpose).
        max_dist = int(min_dist_all.max())
        # Add the candidate with the maximum minimum distance.
        dataset = torch.cat([dataset, permutations[min_dist_all.argmax().unsqueeze(0)]], dim=0)
        # Update dataset sample counter.
        count += 1

        if count % samples_between_logging == 0:
            logger.info(f"Current count: {count}")
            logger.info(f"Max distance: {max_dist}")
            logger.info(f"Time taken for {samples_between_logging} samples: {time.time() - time_take}")
            time_take = time.time()

    # Create a list of the input encodings.
    encodings_input_list = []
    for i in range(dataset.shape[0]):
        temp_dict = {}
        for j, k in enumerate(keys):
            temp_dict[k] = "".join([str(int(x)) for x in dataset[i, 4 * j : 4 * j + 4].detach().cpu()])
        encodings_input_list.append(temp_dict)

    # Use a template for the output encoding.
    # TODO: use the following:
    # encodings_output = classic_encoding["out_enc_dict"]
    encodings_output = {
        -56: "11001000",
        -49: "11001111",
        -48: "11010000",
        -42: "11010110",
        -40: "11011000",
        -36: "11011100",
        -35: "11011101",
        -32: "11100000",
        -30: "11100010",
        -28: "11100100",
        -25: "11100111",
        -24: "11101000",
        -21: "11101011",
        -20: "11101100",
        -18: "11101110",
        -16: "11110000",
        -15: "11110001",
        -14: "11110010",
        -12: "11110100",
        -10: "11110110",
        -9: "11110111",
        -8: "11111000",
        -7: "11111001",
        -6: "11111010",
        -5: "11111011",
        -4: "11111100",
        -3: "11111101",
        -2: "11111110",
        -1: "11111111",
        0: "00000000",
        1: "00000001",
        2: "00000010",
        3: "00000011",
        4: "00000100",
        5: "00000101",
        6: "00000110",
        7: "00000111",
        8: "00001000",
        9: "00001001",
        10: "00001010",
        12: "00001100",
        14: "00001110",
        15: "00001111",
        16: "00010000",
        18: "00010010",
        20: "00010100",
        21: "00010101",
        24: "00011000",
        25: "00011001",
        28: "00011100",
        30: "00011110",
        32: "00100000",
        35: "00100011",
        36: "00100100",
        40: "00101000",
        42: "00101010",
        48: "00110000",
        49: "00110001",
        56: "00111000",
        64: "01000000",
    }

    # Store the input and output encodings in a list.
    encodings_list = [{"in_enc_dict": d, "out_enc_dict": encodings_output} for d in encodings_input_list]

    # Return the list of encoding dictionaries.
    return encodings_list


def main():
    args_dict = parse_args()
    dir_config = ConfigDir(is_analysis=False, **args_dict)

    gener_args_dict = dir_config.args_dict

    if dir_config.args_dict.get("dst_output_dir_name") is not None:
        # Update args dict for generation. Prototype data and generated design will be stored there.
        gener_args_dict.update({"output_dir_name": dir_config.args_dict.get("dst_output_dir_name")})

    # Setup Generation
    gener_dir_config = ConfigDir(is_analysis=False, **gener_args_dict)

    # Set the device
    device = args_dict.get("device", "cuda")

    # Derive the initial dataset in a list of dictionary format
    encoding_dicts_list = derive_diverse_initial_dataset(
        dir_config=dir_config, n_samples=gener_args_dict.get("n_samples"), device=device
    )

    # Remove duplicates
    encodings_df = pd.DataFrame([{"proto_str": str(p)} for p in encoding_dicts_list])
    index_to_keep = set(encodings_df.drop_duplicates().index.tolist())
    encoding_dicts_list = [p for i, p in enumerate(encoding_dicts_list) if i in index_to_keep]

    # Find which design numbers to use
    nb_protos = len(encoding_dicts_list)
    design_numbers = file_parsers.get_list_of_gener_designs_number(gener_dir_config)
    design_numbers_int = [int(design_number) for design_number in design_numbers]

    if len(design_numbers_int) == 0:
        max_known_design_number = -1
    else:
        max_known_design_number = max(design_numbers_int)
    new_design_numbers = [str(i) for i in range(max_known_design_number + 1, max_known_design_number + 1 + nb_protos)]

    # Actally generate the prototypes
    design_generator = DesignGenerator(gener_dir_config)
    logger.info(f"Generating new designs, starting with design_number {max_known_design_number + 1}")
    generated_design_paths_list, generated_design_config_dicts_list = design_generator.generate_all_designs(
        nb_to_generate=nb_protos,
        design_numbers=new_design_numbers,
        output_dir_path=gener_dir_config.generation_out_dir,
        existing_encodings_dicts=encoding_dicts_list,
    )


if __name__ == "__main__":
    main()
