# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

from loguru import logger
from copy import copy
import argparse
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from genial.config.config_dir import ConfigDir
from genial.config.arg_parsers import sampling_analyzer_parser
import genial.experiment.file_parsers as file_parsers
from genial.utils.merge_output_dirs import copy_design
from genial.utils.utils import process_pool_helper


# def main():
if __name__ == "__main__":
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
        "--tgt_nb_samples", default=None, type=int, help="Specify how many samples to keep in the resulting dataset."
    )
    arg_parser.add_argument(
        "--force",
        action="store_true",
        help="Perform the merge even if the destination `output_dir_name` folder already exists. Warning: if true, the destination output_dir_name gets overwritten",
    )
    arg_parser.add_argument(
        "--dry_run", action="store_true", help="Whether or not this is a dry run. If set, designs will not be copied"
    )
    arg_parser.add_argument(
        "--which_valid_step",
        type=str,
        default="synth",
        help="Which step (gener, synth, swact) will be used to select design numbers to be copied.",
    )

    args = arg_parser.parse_known_args()
    args_dict_new = vars(args[0])
    args_dict.update(args_dict_new)

    do_copy = not args_dict.get("dry_run", False)
    # Setup output_directory
    if args_dict.get("dst_output_dir_name") is None:
        new_output_dir_name = dir_config.output_dir_name + "_copy"
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

    # Set the numpy seed for randomness
    random_gen = np.random.default_rng(args_dict.get("seed", 0))

    """
    --- First Step ---
    Get the designs encodings.
    """
    logger.info(f"Reading out all encoding from source directory.")
    # Get all existing encodings
    all_encodings_df = file_parsers.read_all_existing_encodings_v2(root_output_path=dir_config.root_output_dir)
    logger.info(f"Valid generated: {len(all_encodings_df)}")

    if args_dict.get("which_valid_step", "synth") != "synth":
        raise NotImplementedError(f"Only synth is a valid step to source design numbers to be copied.")

    # Get all valid designs
    valid_db = pd.read_parquet(dir_config.root_output_dir / "valid_designs.db.pqt")
    valid_design_numbers = valid_db[valid_db[f"valid_{args_dict.get('which_valid_step')}"]]["design_number"]

    logger.info(f"{len(valid_design_numbers)} valid {args_dict.get('which_valid_step')}ed designs will be copied.")

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

    to_copy_dn = valid_design_numbers.tolist()

    new_design_number_vals_0 = valid_design_numbers

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
                    (idx, dn, new_dn, 0, src_dir_config, out_dir_config, steps, do_copy)
                    for dn, new_dn in zip(to_copy_list, new_design_number_vals)
                ),
                max_workers=args_dict.get("nb_workers", 12),
                pbar=pbar,
            )

    write_merge_db = pd.DataFrame(all_copied_dicts_info)
    write_merge_db.to_parquet(out_dir_config.root_output_dir / "merge_info.pqt")

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


# if __name__ == "__main__":
#     # Args used during testing: --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only
#     # --output_dir_name loop_v2 --frac 0.1 --dst_output_dir_name ten_percent_uniform_loop_v2
#     main()
