# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

import torch
from loguru import logger
from copy import copy
from time import gmtime, strftime, time
import argparse
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from genial.config.config_dir import ConfigDir
from genial.config.arg_parsers import sampling_analyzer_parser
import genial.experiment.file_parsers as file_parsers
from genial.experiment.task_analyzer import Analyzer
from genial.utils.merge_output_dirs import copy_design
from genial.utils.utils import process_pool_helper

from genial.utils.utils import extract_cont_str, convert_cont_str_to_np


def create_design_and_bin_maps(encodings_df):
    if "bins" not in encodings_df.keys():
        raise ValueError("encodings_df must contain a bins column")
    if "design_number" not in encodings_df.keys():
        raise ValueError("encodings_df must contain a design_number column")

    # Get a design number to bin map
    design_to_bin_map = dict(zip(encodings_df["design_number"], encodings_df["bins"]))

    # Get a bin to design number list map
    bin_to_design_map = {}
    for k, v in design_to_bin_map.items():
        if v in bin_to_design_map:
            bin_to_design_map[v].append(k)
        else:
            bin_to_design_map[v] = [k]

    return design_to_bin_map, bin_to_design_map


def split_design_numbers_in_bins_on_tgt_metric(
    to_split_df: pd.DataFrame, metric_df: pd.DataFrame, tgt_metric_name: str = "nb_transistors", bin_size: int = 10
) -> pd.DataFrame:
    """
    Return the list of bins for each design_number based on target_metric
    Warning: values of target_metric are expected to be of type `int`
    Warning: this function can return some nan values in bins if the metric_df and to_splt_df design numbers are not aligned
    """

    # Create a map to get the number of transistors for each design
    dn_to_tgt_metric_map = dict(zip(metric_df["design_number"], metric_df[tgt_metric_name]))

    # Only keep designs that have synth data
    cond = to_split_df["design_number"].isin(dn_to_tgt_metric_map)
    logger.info(f"Removing {int((~cond).sum())} designs from to_split_df as they do not have synth data!")
    _to_split_df = to_split_df[cond].reset_index(drop=True)
    remaining_to_split_df = to_split_df[~cond].reset_index(drop=True)  # Designs that do not have associated metric data

    # Map the number of transistors to the encodings df
    _to_split_df[tgt_metric_name] = _to_split_df["design_number"].map(dn_to_tgt_metric_map)

    # Create bins of interval size 10
    min_trans_count = _to_split_df[tgt_metric_name].min()
    max_trans_count = _to_split_df[tgt_metric_name].max()

    # Define the bin edges
    bin_edges = np.arange(min_trans_count - 1, max_trans_count + bin_size, bin_size)

    # Get bins for the number of transistors
    _to_split_df["bins"] = pd.cut(_to_split_df[tgt_metric_name], bins=bin_edges).astype(str)
    remaining_to_split_df["bins"] = "nan"

    # Recombine the to_split_df and the remaining_to_split_df
    to_split_df = pd.concat([_to_split_df, remaining_to_split_df], ignore_index=True)

    return to_split_df


def get_number_of_samples_per_bin_needed(
    bin_to_design_map: dict[str, list[str]],
    tgt_nb_samples: int,
):
    """
    This functionary uses a binary search to determine the right number of samples per bin needed to reach the
    target number of samples.
    """
    # Get a bin id to bin count map
    bin_to_design_map_count = {k: len(v) for k, v in bin_to_design_map.items()}

    # Find the right number of samples to retrieve per bin
    low = 0
    high = max(bin_to_design_map_count.values())

    # Function to get the number of samples retrieved depending on the max element count per bin
    def get_total_w_cap(bin_to_design_map_count, el_count):
        return sum(min(el_count, bin_count) for bin_count in bin_to_design_map_count.values())

    # Apply binary search
    while low < high:
        mid = (low + high + 1) // 2
        if get_total_w_cap(bin_to_design_map_count, el_count=mid) < tgt_nb_samples:
            low = mid
        else:
            high = mid - 1

    return low


def extend_list_of_selected_designs(
    selected_design_number_list: list[str],
    to_split_df: pd.DataFrame,
    tgt_nb_samples: int,
    random_gen: np.random.Generator,
    bin_to_design_map: dict[str, list[str]] = None,
    remaining_bins: list[str] = None,
    uniform_sampling_method: str = "random",
    device: str = "cuda",
) -> list[str]:
    """
    Extend a possibly already existing list of design number from the list of bins to reach tgt_nb_samples.
    Returns the list of design number, and updated bin_to_design_map and remaining_bins info.
    It generates the latter two if they are not provided in input
    """

    if bin_to_design_map is None:
        # Get a design number to bin map and a bin to design number list map
        design_to_bin_map, bin_to_design_map = create_design_and_bin_maps(to_split_df)

        # Get the bins for the loop initialize
        remaining_bins = list(bin_to_design_map.keys())
    else:
        if remaining_bins is None:
            logger.warning(
                f"bin_to_design_map has been passed in input argument but remaining_bins was not, "
                f"so it has been generated out of bin_to_design_map."
            )
            remaining_bins = list(bin_to_design_map.keys())

    # Shuffle the design number lists in bin_to_design_map to ensure randomness
    for k in bin_to_design_map:
        random_gen.shuffle(bin_to_design_map[k])

    # Log operation and start timer.
    logger.info("Sampling the design numbers based on their bins, this can take a while ...")
    start_time = time()

    # Remove nan bins
    if "nan" in remaining_bins:
        remaining_bins.pop(remaining_bins.index("nan"))
        logger.info(f"Excluded nan bins from sampling.")
    nb_bins = len(remaining_bins)

    if uniform_sampling_method == "random":
        return random_uniform_sampling_method(
            tgt_nb_samples, remaining_bins, bin_to_design_map, selected_design_number_list, random_gen, start_time
        )
    elif uniform_sampling_method == "distance_maximization":
        return distance_maximization_uniform_sampling_method(
            selected_design_number_list,
            bin_to_design_map,
            tgt_nb_samples,
            to_split_df,
            design_to_bin_map,
            nb_bins,
            start_time,
            device,
        )
    else:
        raise ValueError(f"uniform_sampling_method {uniform_sampling_method} not recognized!")


def distance_maximization_uniform_sampling_method(
    selected_design_number_list: list[str],
    bin_to_design_map: dict[str, list[str]],
    tgt_nb_samples: int,
    to_split_df: pd.DataFrame,
    design_to_bin_map: dict[str, str],
    nb_bins: int,
    start_time: float,
    device: str = "cuda",
):
    # Determine the maximum number of samples per bin needed to reach the right target number of samples.
    bin_maximum = get_number_of_samples_per_bin_needed(bin_to_design_map, tgt_nb_samples)

    # Add all bins that are below the count
    added_designs = []
    # Create a dict that contains the bins that will be added later
    new_bins_to_assign = {}
    # Dict to track how many samples of each bins are added
    bin_addition_count = {}

    for k, v in bin_to_design_map.items():
        if len(v) <= bin_maximum:
            added_designs += v
        else:
            new_bins_to_assign[k] = v
            bin_addition_count[k] = 0

    # Obtain a numpy representation of the bit sequence
    to_split_df["encodings_input_cont_str"] = to_split_df["in_enc_dict"].map(lambda x: extract_cont_str(x))
    to_split_df["encodings_input_np"] = to_split_df["encodings_input_cont_str"].map(lambda x: convert_cont_str_to_np(x))

    # Obtain a map from design_number to numpy representation
    id_to_np_map = dict(zip(to_split_df["design_number"], to_split_df["encodings_input_np"]))

    # Create a dataframe containing only the designs that have not yet been assigned
    assigned_cond = to_split_df["design_number"].isin(added_designs)
    remaining_df = to_split_df[~assigned_cond].reset_index(drop=True)

    # We use cuda to obtain the L1 distances between all the unassigned samples and the assigned ones.

    # Convert and move the unassigned encodings sequences to the GPU
    remaining_torch = torch.stack(
        [torch.tensor(i, dtype=torch.bool, device=device) for i in remaining_df["encodings_input_np"]]
    )

    # To meet memory constaint, we use batches of 100.
    batch_size = 100
    dist_list = []

    for i in range(0, len(added_designs), batch_size):
        # Retrieve the batch
        batch = added_designs[i : i + batch_size]
        # Store the batch in a tensor
        assigned_torch = torch.stack(
            [torch.tensor(id_to_np_map[d_id], dtype=torch.bool, device=device) for d_id in batch]
        )
        # Derive the L1 distance between the assigned and unassigned samples.
        xor = assigned_torch[:, None, :] ^ remaining_torch[None, :, :]
        dist_batch = xor.sum(dim=2).cpu()
        # Append the distances for the batch
        dist_list.extend(dist_batch)
        # Print a log to track progress
        logger.info(f"{i}/{len(added_designs)} distances from assigned samples derived.")

    # Stack all distances and retrieve the minimum distance to any assigned sample for each unassigned sample.
    dist_min = torch.stack(dist_list).min(dim=0).values

    # Set a value to count the number of designs.
    d_counter = len(added_designs)

    # Loop until conditions are met
    while True:
        # Map the latest count per bin
        remaining_df["count_per_bin"] = remaining_df["bins"].map(bin_addition_count)
        # Retrieve all indexes with max distance
        max_indexes = np.where(dist_min == dist_min.max())[0]
        # Find the index where the max distance is and select one where the bins have been the least chosen
        new_index = int(remaining_df.loc[max_indexes, "count_per_bin"].idxmin())
        # Retrieve the max value for this iteration
        max_min_value = int(dist_min[new_index])
        # Retrieve the design id
        d_id = remaining_df["design_number"].iloc[new_index]
        # Retrieve the bin
        curr_bin = design_to_bin_map[d_id]
        # Add the design id
        added_designs.append(d_id)
        # Add the bin count
        bin_addition_count[curr_bin] += 1

        # Update all the distances with the new sample information
        chosen_torch = torch.from_numpy(id_to_np_map[d_id]).to(device=device, dtype=torch.bool)
        new_dist = (remaining_torch ^ chosen_torch).sum(dim=1).cpu()
        dist_min = torch.min(dist_min, new_dist)

        # If the count for the bin has been reached, exclude all remaining samples in the bin.
        # (Technically, we could say strictly or equal to the bin maximum, but we allow an extra sample here with the
        # hope that it permits more diverse samples, in terms of L1 distance, in the dataset).
        if bin_addition_count[curr_bin] > bin_maximum:
            to_exclude = remaining_df.index[remaining_df["bins"] == curr_bin].tolist()
            dist_min[to_exclude] = 0

        # Add 1 to the design number counter
        d_counter += 1

        # If enough design have been added, we can break
        if d_counter >= tgt_nb_samples:
            break

        # Print the remaining samples to find
        remaining_to_assign = int(tgt_nb_samples - d_counter)
        if remaining_to_assign % 100 == 0:
            logger.info(f"{remaining_to_assign} samples remaining to assign!")
            logger.info(f"Current max min value: {max_min_value}")

    selected_design_number_list = selected_design_number_list + added_designs

    logger.info(
        f"Uniformly sampled {len(selected_design_number_list)} design from {nb_bins} bins in {time() - start_time}s"
    )
    return selected_design_number_list


def random_uniform_sampling_method(
    tgt_nb_samples: int,
    remaining_bins: list[str],
    bin_to_design_map: dict[str, list[str]],
    selected_design_number_list: list[str],
    random_gen: np.random.Generator,
    start_time: float,
):
    selected_design_number_set = set()
    _tgt_nb_samples = tgt_nb_samples
    # Remove nan bins
    if "nan" in remaining_bins:
        remaining_bins.pop(remaining_bins.index("nan"))
        logger.info(f"Excluded nan bins from sampling.")
    nb_bins = len(remaining_bins)
    while _tgt_nb_samples > 0:
        # Select the bin to sample from
        selected_bin = random_gen.choice(remaining_bins)

        # Store the element ID
        selected_design_number = bin_to_design_map[selected_bin].pop()
        if selected_design_number not in selected_design_number_set:  # Check in case of duplicates
            selected_design_number_list.append(selected_design_number)
            selected_design_number_set.add(selected_design_number)
            _tgt_nb_samples -= 1

        # If the bin no longer contains samples, remove it
        if not bin_to_design_map[selected_bin]:
            remaining_bins.remove(selected_bin)

        # Check to exit the while loop if all bins are empty
        if not remaining_bins:
            break

    logger.info(
        f"Uniformly sampled {len(selected_design_number_list)} design from {nb_bins} bins in {time() - start_time}s"
    )
    return selected_design_number_list


def assign_design_to_sets(test_ids, valid_ids, ids_to_sample, sample_frac, random_gen: np.random.Generator):
    """
    Assigns a fraction sample_frac of the ids_to_sample to both test_ids and valid_ids.
    """
    # Derive the number to sample from the sample frac and length of ids_to_sample.
    to_select = int(np.ceil(len(ids_to_sample) * sample_frac))

    # Sample randomly
    design_id_list_temp = random_gen.choice(ids_to_sample, to_select, replace=False).tolist()

    # If even, assign evenly
    if to_select % 2 == 0:
        half_index = int(to_select / 2)
        test_ids += design_id_list_temp[:half_index]
        valid_ids += design_id_list_temp[half_index:]

    # Elif valid is larger, assign one more to test
    elif len(test_ids) < len(valid_ids):
        half_index = int(np.ceil(to_select / 2))
        test_ids += design_id_list_temp[:half_index]
        valid_ids += design_id_list_temp[half_index:]

    # Else assign one more to valid
    else:
        half_index = int(np.ceil(to_select / 2))
        valid_ids += design_id_list_temp[:half_index]
        test_ids += design_id_list_temp[half_index:]

    # Return the updated test_ids and valid_ids lists
    return test_ids, valid_ids


def derive_fixed_sets_helper_df(all_encodings_df: pd.DataFrame, selected_encodings_df: pd.DataFrame):
    """
    Create a dataframe assisting in the creation of fixed valid/test sets.

    The df is used to create a balanced valid/test set.
    """
    # Get the initial and final count of transistor bins.
    fixed_sets_helper_df = all_encodings_df["bins"].value_counts().reset_index()
    fixed_sets_helper_df = fixed_sets_helper_df.rename(columns={"count": "initial_count"})
    fixed_sets_helper_df["final_count"] = 0

    final_bin_count_map = dict(selected_encodings_df["bins"].value_counts())

    cond = fixed_sets_helper_df["bins"].isin(final_bin_count_map)
    fixed_sets_helper_df.loc[cond, "final_count"] = fixed_sets_helper_df.loc[cond, "bins"].map(final_bin_count_map)

    # Sort using the final count
    fixed_sets_helper_df = fixed_sets_helper_df.sort_values("final_count").reset_index(drop=True)

    # Default assignment is 0 (for bin count of less than 10).
    fixed_sets_helper_df["set_assignment"] = 0.0

    # Only a single encoding gets assigned to either valid or test when bin count is between 10 and 17.
    cond = fixed_sets_helper_df["final_count"] >= 10
    fixed_sets_helper_df.loc[cond, "set_assignment"] = 1.0

    # 10 percent of samples get assigned equally to valid and test set when bin count is between 18 and 29.
    cond = fixed_sets_helper_df["final_count"] >= 18
    fixed_sets_helper_df.loc[cond, "set_assignment"] = 0.1

    # 5 percent of samples get assigned equally to valid and test set when bin count is between 30 and 59.
    cond = fixed_sets_helper_df["final_count"] >= 30
    fixed_sets_helper_df.loc[cond, "set_assignment"] = 0.05

    # 3 percent of samples get assigned equally to valid and test set when bin count is between 60 and 99.
    cond = fixed_sets_helper_df["final_count"] >= 60
    fixed_sets_helper_df.loc[cond, "set_assignment"] = 0.03

    # 2 percent of samples get assigned equally to valid and test set when bin count is equal or greater than 100.
    cond = fixed_sets_helper_df["final_count"] >= 100
    fixed_sets_helper_df.loc[cond, "set_assignment"] = 0.02

    return fixed_sets_helper_df


def main():
    """
    The goal of this script is to enable merging two separated output directories from the same experiment into a single directory.
    """

    # Get argument parser
    args_dict = sampling_analyzer_parser()

    dir_config = ConfigDir(is_analysis=True, **args_dict)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--dst_output_dir_name",
        default=None,
        type=str,
        help="Name of the output dir where to put the results. By default, all results are stored in `$WORK_DIR/output/<output_dir_name>`",
    )
    arg_parser.add_argument(
        "--frac", default=1.0, type=float, help="Percentage of the designs in output_dir_name_0 to merge"
    )
    arg_parser.add_argument(
        "--tgt_nb_samples", default=None, type=int, help="Specify how many samples to keep in the resulting dataset."
    )
    help_message = (
        "Device to use. Only applies to distance_maximization_uniform_sampling_method function. Remaining "
        "functionality runs on CPU."
    )
    arg_parser.add_argument("--device", default="cuda", type=str, help=help_message)
    arg_parser.add_argument("--uniform_sampling_method", default="random", type=str, help="Sampling method type.")
    arg_parser.add_argument(
        "--force",
        action="store_true",
        help="Perform the merge even if the destination `output_dir_name` folder already exists. Warning: if true, the destination output_dir_name gets overwritten",
    )
    arg_parser.add_argument(
        "--dry_run", action="store_true", help="Whether or not this is a dry run. If set, designs will not be copied"
    )
    arg_parser.add_argument(
        "--fixed_valid_test_ids",
        action="store_true",
        help="This will create a file to allow the same valid and test designs to be used independent of design subsets",
    )

    args = arg_parser.parse_known_args()
    args_dict_new = vars(args[0])
    args_dict.update(args_dict_new)

    do_copy = not args_dict.get("dry_run", False)
    # Setup output_directory
    if args_dict.get("dst_output_dir_name") is None:
        new_output_dir_name = "merge_" + strftime("%Y-%m-%d_%H-%M", gmtime())
    else:
        new_output_dir_name = args_dict.get("dst_output_dir_name")
    out_dir_args_dict = copy(args_dict)
    out_dir_args_dict.update(
        {"output_dir_name": new_output_dir_name, "rebuild_db": True, "force": args_dict.get("force")}
    )
    logger.info(f"Setting up destination output directory.")
    out_dir_config = ConfigDir(is_merging=True, **out_dir_args_dict)

    """
    --- Preliminary Step ---
    Setup execution.
    """

    # Fixed valid test set setup
    create_fixed_valid_test_flag = False
    fixed_dataset_dict = None
    if args_dict_new.get("fixed_valid_test_ids", False):
        fixed_dataset_file_path = dir_config.root_output_dir / "fixed_dataset_split.json"
        if fixed_dataset_file_path.exists():
            logger.info(f"Using the existing 'fixed_dataset_split.json' file.")
            fixed_dataset_dict = json.load(open(fixed_dataset_file_path, "r"))
        else:
            logger.info("'fixed_dataset_split.json' file does not yet exist. Creating it...")
            if args_dict.get("tgt_nb_samples") is not None:
                raise ValueError(
                    "Functionality to use create_fixed_valid_test_ids with tgt_nb_samples when the "
                    "'fixed_dataset_split.json' file has not yet been created is not implemented! Aborting!"
                )
            elif args_dict["frac"] != 0.1:
                raise ValueError(
                    "The logic of create_fixed_valid_test_ids relies on choosing a frac of 0.1. Create the "
                    "'fixed_dataset_split.json' file with a frac of 0.1 before creating using another fraction. "
                    "Aborting!"
                )
            create_fixed_valid_test_flag = True

    # Set the numpy seed for randomness
    random_gen = np.random.default_rng(args_dict.get("seed", 0))

    """
    --- First Step ---
    Get the designs encodings.
    """
    logger.info(f"Reading out all encoding from source directory.")
    # Get all existing encodings
    all_encodings_df = file_parsers.read_all_existing_encodings_v2(root_output_path=dir_config.root_output_dir)

    logger.info(f"Len: {len(all_encodings_df)}")

    # Get the synthesis db
    synth_df_path = dir_config.analysis_out_dir / "synth_analysis.db.pqt"
    if synth_df_path.exists():
        synth_df = Analyzer._load_database(synth_df_path)
    else:
        raise FileNotFoundError(f"Did not find synth analysis database at {synth_df_path}. Run analyzer first!")

    logger.info(f"Synthesized designs analysis database loaded from pre-existing file:")
    logger.info(synth_df_path)

    # Get the number of samples to keep
    if args_dict.get("tgt_nb_samples") is not None:
        n_samples = int(args_dict.get("tgt_nb_samples"))
        logger.info(
            f"Target nb of samples: {n_samples} | Defined from 'tgt_nb_samples'=={args_dict.get('tgt_nb_samples')} argument "
        )
    else:
        n_samples = int(all_encodings_df.shape[0] * args_dict["frac"])
        if not n_samples > 0:
            raise ValueError(f"Resulting `n_samples` is 0. Please adapt frac or check your source `output_dir_name`.")
        logger.info(f"Target nb of samples: {n_samples} | Defined from 'frac'=={args_dict['frac']} argument ")

    # Create a list to store the selected design numbers
    selected_design_number_list = []

    # If fixed_dataset_dict is not None, we have to add the design ids from the fixed valid/test sets.
    # To make sure they are present in any subset.
    if fixed_dataset_dict is not None:
        selected_design_number_list += (
            fixed_dataset_dict["test_design_numbers"] + fixed_dataset_dict["valid_design_numbers"]
        )
        # Subtract the number of samples already added
        n_samples -= len(selected_design_number_list)

        if n_samples < 10_000:
            logger.warning(
                "Design number count is too low to use fixed_valid_test_ids! Either remove the flag or "
                "increase the number of designs to add to the dataset! Aborting!"
            )

    # If special designs exist, add them
    special_design_numbers = ConfigDir.read_special_designs(dir_config)["design_numbers"]
    for design_number in special_design_numbers:
        if design_number not in selected_design_number_list:
            selected_design_number_list.append(design_number)
            n_samples -= 1

    # Split the designs into bins based on the target metric
    logger.info(f"Splitting designs into bins based on target metric.")

    # Get the bins for each design number
    all_encodings_df = split_design_numbers_in_bins_on_tgt_metric(
        to_split_df=all_encodings_df,
        metric_df=synth_df,
        tgt_metric_name="nb_transistors",
    )

    # Get the list of design numbers to keep based on their bins and on the target number of designs
    selected_design_number_list = extend_list_of_selected_designs(
        selected_design_number_list=selected_design_number_list,
        to_split_df=all_encodings_df,
        tgt_nb_samples=n_samples,
        random_gen=random_gen,
        uniform_sampling_method=args_dict.get("uniform_sampling_method"),
        device=args_dict.get("device"),
    )

    # Reduce all_encodings_df to the list of selected ones
    cond = all_encodings_df["design_number"].isin(selected_design_number_list)
    selected_encodings_df = all_encodings_df[cond].reset_index(drop=True)

    if create_fixed_valid_test_flag:
        # Get a design number to bin map and a bin to design number list map
        design_to_bin_map, bin_to_design_map = create_design_and_bin_maps(selected_encodings_df)

        fixed_sets_helper_df = derive_fixed_sets_helper_df(all_encodings_df, selected_encodings_df)

        valid_ids = []
        test_ids = []

        for i, row in fixed_sets_helper_df.iterrows():
            if row.set_assignment == 0:
                # Skip the brackets that have a low design id count
                pass
            elif row.set_assignment == 1:
                # Select a single design id
                new_design_list = random_gen.choice(bin_to_design_map[row.bins], 1).tolist()
                # Add the design id to the smallest list
                if len(test_ids) < len(valid_ids):
                    test_ids += new_design_list
                else:
                    valid_ids += new_design_list
            else:
                # Assign ids using assign_design_to_sets function
                test_ids, valid_ids = assign_design_to_sets(
                    test_ids,
                    valid_ids,
                    ids_to_sample=bin_to_design_map[row.bins],
                    sample_frac=row.set_assignment,
                    random_gen=random_gen,
                )

        fixed_dataset_dict = {
            "test_design_numbers": test_ids,
            "valid_design_numbers": valid_ids,
        }
        json.dump(fixed_dataset_dict, fp=open(fixed_dataset_file_path, "w"), indent=4)

    if "fixed_valid_test_ids" in args_dict_new and args_dict_new["fixed_valid_test_ids"]:
        # Let's validate all the design numbers from fixed_dataset_dict are in the selected_encodings_df
        cond1 = selected_encodings_df["design_number"].isin(fixed_dataset_dict["test_design_numbers"]).sum() == len(
            fixed_dataset_dict["test_design_numbers"]
        )
        cond2 = selected_encodings_df["design_number"].isin(fixed_dataset_dict["valid_design_numbers"]).sum() == len(
            fixed_dataset_dict["valid_design_numbers"]
        )
        if not (cond1 and cond2):
            raise ValueError(
                "This is unexpected. selected_encodings_df should contain all the designs in fixed_dataset_dict"
            )

    # # Check to inspect distribution
    # check_df = all_encodings_df['bins'].value_counts().copy().reset_index()
    # check_df = check_df.rename(columns={'count': 'initial_count'})
    # check_df['final_count'] = 0
    #
    # final_bin_count_map = dict(selected_encodings_df['bins'].value_counts())
    #
    # cond = check_df['bins'].isin(final_bin_count_map)
    # check_df.loc[cond, 'final_count'] = check_df.loc[cond, 'bins'].map(final_bin_count_map)
    # check_df = check_df.sort_values('bins').reset_index(drop=True)

    """ 
    --- Second Step ---
    Perform the designs copy into the new directory.
    """
    # Need to copy the generation files, the synthesis files and the test files
    if args_dict.get("synth_only", False):
        steps = ["gener", "synth"]
    else:
        steps = ["gener", "synth", "swact", "analysis"]
    logger.info(f"Copying steps: {steps}")

    to_copy_dn = selected_encodings_df["design_number"].tolist()

    new_design_number_vals_0 = list(range(len(to_copy_dn)))
    zfill_len = max([len(design_number) for design_number in to_copy_dn])

    logger.info(f"Doing the merge ...")
    all_copied_dicts_info = []
    to_enumerate = [(to_copy_dn, dir_config, new_design_number_vals_0)]
    for idx, (to_copy_list, src_dir_config, new_design_number_vals) in enumerate(to_enumerate):
        logger.info(f"Copying {len(to_copy_list)} designs from output_dir_name_{idx} ...")
        desc = f"x{args_dict.get('nb_workers', 12)} | Copying from output_dir_name_{idx}"
        with tqdm(total=len(to_copy_list), desc=desc) as pbar:  # Progress bar
            all_copied_dicts_info = process_pool_helper(
                func=copy_design,
                func_args_gen=(
                    (idx, dn, new_dn, zfill_len, src_dir_config, out_dir_config, steps, do_copy)
                    for dn, new_dn in zip(to_copy_list, new_design_number_vals)
                ),
                max_workers=args_dict.get("nb_workers", 12),
                pbar=pbar,
            )

    write_merge_db = pd.DataFrame(all_copied_dicts_info)
    write_merge_db.to_parquet(out_dir_config.root_output_dir / "merge_info.pqt")

    if "fixed_valid_test_ids" in args_dict_new and args_dict_new["fixed_valid_test_ids"]:
        # Create map
        old_to_new_map = dict(zip(write_merge_db["old_dn"], write_merge_db["new_dn"]))

        fixed_dataset_dict["test_design_numbers"] = [
            old_to_new_map[i] for i in fixed_dataset_dict["test_design_numbers"]
        ]
        fixed_dataset_dict["valid_design_numbers"] = [
            old_to_new_map[i] for i in fixed_dataset_dict["valid_design_numbers"]
        ]
        # Also store the fixed_dataset_dict in the new dataset dir
        fixed_dataset_file_path = out_dir_config.root_output_dir / "fixed_dataset_split.json"
        json.dump(fixed_dataset_dict, fp=open(fixed_dataset_file_path, "w"), indent=4)

    logger.info(f"All files copied in {out_dir_config.root_output_dir}")
    logger.info(f"Total number of files: {len(all_copied_dicts_info)}")
    logger.info(f"Old to new design numbers map written in {out_dir_config.root_output_dir / 'merge_info.pqt'}")

    """
    --- Third Step ---
    Deal with special designs.
    """
    special_designs = dir_config.read_special_designs(dir_config)
    out_special_designs_dict = {"legend": [], "design_numbers": [], "src_dir_idx": []}
    for idx, src_special_designs in enumerate(
        [
            special_designs,
        ]
    ):
        for leg, dn in zip(src_special_designs["legend"], src_special_designs["design_numbers"]):
            if dn in to_copy_dn:
                out_special_designs_dict["legend"].append(leg)
                out_special_designs_dict["design_numbers"].append(dn)
                out_special_designs_dict["src_dir_idx"].append(idx)

    # Get old to new design numbers mapping
    new_dns = []
    for idx, old_dn in enumerate(out_special_designs_dict["design_numbers"]):
        sub_merge_df = write_merge_db[write_merge_db["source_idx"] == out_special_designs_dict["src_dir_idx"][idx]]
        new_dn = sub_merge_df[sub_merge_df["old_dn"] == old_dn]["new_dn"].item()
        new_dns.append(new_dn)

    # Update the special designs dictionary
    out_special_designs_dict["design_numbers"] = new_dns
    out_special_designs_dict.pop("src_dir_idx")

    json.dump(out_special_designs_dict, open(out_dir_config.special_designs_filepath, "a"))
    logger.info(f"Special design information has been setup in {out_dir_config.special_designs_filepath}")
    logger.info(out_special_designs_dict)


if __name__ == "__main__":
    # Args used during testing: --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only
    # --output_dir_name loop_v2 --frac 0.1 --dst_output_dir_name ten_percent_uniform_loop_v2
    main()
