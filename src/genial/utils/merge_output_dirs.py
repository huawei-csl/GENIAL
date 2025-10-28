from genial.utils.utils import process_pool_helper
from loguru import logger
from copy import copy

from time import gmtime, strftime

import argparse

import pandas as pd

import shutil
import json

from tqdm import tqdm


from genial.config.config_dir import ConfigDir
from genial.config.arg_parsers import double_analyzer_parser

import genial.experiment.file_parsers as file_parsers


def _get_dir_path(dir_config: ConfigDir, step: str, design_number: str):
    if step == "gener":
        return dir_config.generation_out_dir / f"res_{design_number}"
    if step == "synth":
        return dir_config.synth_out_dir / f"res_{design_number}"
    if step == "swact":
        return dir_config.swact_out_dir / f"res_{design_number}"
    if step == "power":
        return dir_config.power_out_dir / f"res_{design_number}"
    if step == "analysis":
        return dir_config.analysis_out_dir / "swact_analysis" / f"res_{design_number}"


def copy_design(idx, design_number, new_design_number_val, zfill_len, src_dir_config, out_dir_config, steps, do_copy):
    new_design_number = str(new_design_number_val).zfill(zfill_len)

    matching_number_dict = {
        "source_idx": idx,
        "old_dn": design_number,
        "new_dn": new_design_number,
    }

    for step in steps:
        new_path = _get_dir_path(out_dir_config, step, new_design_number)
        src_path = _get_dir_path(src_dir_config, step, design_number)
        copy_ok = False

        # logger.info(f"Copy of {src_path} to {new_path}")
        if src_path.exists():
            if do_copy:
                shutil.copytree(src_path, new_path)
                copy_ok = True
            else:
                pass
        else:
            # logger.warning(f"Path {src_path} does not exists")
            pass

        matching_number_dict[f"{step}_ok"] = copy_ok

    return matching_number_dict


def main():
    """
    The goal of this script is to enable merging two separated output directories from the same experiment into a single directory.
    """

    # Get argument parser
    args_0, args_1, args_dict = double_analyzer_parser()

    dir_config_0 = ConfigDir(is_analysis=True, **args_0)
    dir_config_1 = ConfigDir(is_analysis=True, **args_1)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--dst_output_dir_name",
        default=None,
        type=str,
        help="Name of the output dir where to put the results. By default, all results are stored in `$WORK_DIR/output/<output_dir_name>`",
    )
    arg_parser.add_argument(
        "--frac_0", default=1.0, type=float, help="Percentage of the designs in output_dir_name_0 to merge"
    )
    arg_parser.add_argument(
        "--frac_1", default=1.0, type=float, help="Precentage of the designs in output_dir_name_1 to merge"
    )
    arg_parser.add_argument(
        "--force",
        action="store_true",
        help="Perform the merge even if the destination `output_dir_name` folder already exists. Warning: if true, the destination output_dir_name gets overwritten",
    )
    arg_parser.add_argument(
        "--dry_run", action="store_true", help="Whether or not this is a dry run. If set, designs will not be copied"
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
    out_dir_args_dict = copy(args_1)
    out_dir_args_dict.update(
        {"output_dir_name": new_output_dir_name, "rebuild_db": True, "force": args_dict.get("force")}
    )
    logger.info(f"Setting up destination output directory.")
    out_dir_config = ConfigDir(is_merging=True, **out_dir_args_dict)

    """
    --- First Step ---
    Get all designs encodings and associated design numbers for both output directories.
    We need to get a single copy of each design with the same data encoding.
    """
    logger.info(f"Reading out all encoding of both source directories.")
    # Get all existing encodings
    all_encodings_df_0 = file_parsers.read_all_existing_encodings_v2(root_output_path=dir_config_0.root_output_dir)
    all_encodings_df_1 = file_parsers.read_all_existing_encodings_v2(root_output_path=dir_config_1.root_output_dir)

    logger.info(f"Len_0: {len(all_encodings_df_0)}")
    logger.info(f"Len_1: {len(all_encodings_df_1)}")

    # Reduce the design number list by a fraction
    if args_dict.get("frac_0", 1.0) != 1.0:
        all_encodings_df_0 = all_encodings_df_0.sample(frac=args_dict.get("frac_0", 1.0), replace=False)
        logger.warning(f"frac_0 was given as argument. Len_0 has been reduced to {len(all_encodings_df_0)}")

    if args_dict.get("frac_1", 1.0) != 1.0:
        all_encodings_df_1 = all_encodings_df_1.sample(frac=args_dict.get("frac_1", 1.0), replace=False)
        logger.warning(f"frac_1 was given as argument. Len_1 has been reduced to {len(all_encodings_df_1)}")

    # Add a column to know df origin after concatenation
    all_encodings_df_0["df_version"] = 0
    all_encodings_df_1["df_version"] = 1

    # Concatenate both dfs
    all_encodings_df_conc = pd.concat([all_encodings_df_0, all_encodings_df_1]).reset_index(drop=True)

    # Drop duplicates
    all_encodings_df_conc_no_dup = all_encodings_df_conc.drop_duplicates(
        ["enc_dict_str", "in_enc_dict", "out_enc_dict"]
    ).reset_index(drop=True)

    # Split conc df
    cond = all_encodings_df_conc_no_dup["df_version"] == 0
    to_copy_dn_0 = all_encodings_df_conc_no_dup[cond]["design_number"].tolist()
    cond = all_encodings_df_conc_no_dup["df_version"] == 1
    to_copy_dn_1 = all_encodings_df_conc_no_dup[cond]["design_number"].tolist()

    # Old code kept for reference
    # # Add numbered suffixes to all keys in each dataframes
    # all_encodings_df_0 = all_encodings_df_0.add_suffix("_0")
    # all_encodings_df_1 = all_encodings_df_1.add_suffix("_1")
    #
    # # Get the list of designs that must be taken only once
    # logger.info(f"Searching for duplicates design numbers ...")
    # duplicate_df = pd.merge(left=all_encodings_df_0, right=all_encodings_df_1, how='inner', left_on="enc_dict_str_0", right_on="enc_dict_str_1")
    #
    # logger.info(f"Number of duplictates: {len(duplicate_df)} | src_0 length: {len(all_encodings_df_0)} | src_1 length: {len(all_encodings_df_1)}")
    #
    # # Extract designs that must be copied
    # logger.info(f"Extracting unique design numbers ...")
    # df_0_unique = all_encodings_df_0[~all_encodings_df_0["design_number_0"].isin(duplicate_df["design_number_0"])]
    # df_1_unique = all_encodings_df_1[~all_encodings_df_1["design_number_1"].isin(duplicate_df["design_number_1"])]
    #
    # # Prepare the list of design numbers that my be copied from 0 and from 1
    # # We copy the duplicates only once
    # logger.info(f"Setting up the list of designs to copy from both source directories")
    # to_copy_dn_0 = df_0_unique["design_number_0"].to_list() + duplicate_df["design_number_0"].to_list()
    # to_copy_dn_1 = df_1_unique["design_number_1"].to_list() # No duplicates

    logger.info(
        f"Idx:{0} | {len(to_copy_dn_0)} designs will be copied from output_dir_name_0 {args_0.get('output_dir_name')}"
    )
    logger.info(
        f"Idx:{1} | {len(to_copy_dn_1)} designs will be copied from output_dir_name_1 {args_1.get('output_dir_name')}"
    )

    """ 
    --- Second Step ---
    Perform the designs copy into the new directory.
    """
    # Need to copy the generation files, the synthesis files and the test files
    if args_dict.get("synth_only", False):
        steps = ["gener", "synth"]
    elif args_dict.get("bulk_flow_dirname") is not None:
        if args_dict.get("bulk_flow_dirname") == "power_out":
            steps = ["gener", "power"]
        if args_dict.get("bulk_flow_dirname") == "synth_out":
            steps = ["gener", "synth"]
    else:
        steps = ["gener", "synth", "swact", "analysis"]
    logger.info(f"Copying steps: {steps}")

    new_design_number_vals_0 = list(range(len(to_copy_dn_0)))
    new_design_number_vals_1 = list(range(len(to_copy_dn_0), len(to_copy_dn_0) + len(to_copy_dn_1)))
    zfill_len = max([len(design_number) for design_number in to_copy_dn_0])

    logger.info(f"Doing the merge ...")
    all_copied_dicts_info = []
    for idx, (to_copy_list, src_dir_config, new_design_number_vals) in enumerate(
        [(to_copy_dn_0, dir_config_0, new_design_number_vals_0), (to_copy_dn_1, dir_config_1, new_design_number_vals_1)]
    ):
        logger.info(f"Copying {len(to_copy_list)} from output_dir_name_{idx} ...")
        desc = f"x{args_dict.get('nb_workers', 12)} | Copying from output_dir_name_{idx}"
        with tqdm(total=len(to_copy_list), desc=desc) as pbar:
            all_copied_dicts_info += process_pool_helper(
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

    logger.info(f"All files copied in {out_dir_config.root_output_dir}")
    logger.info(f"Total number of files: {len(all_copied_dicts_info)}")
    logger.info(f"Old to new design numbers map written in {out_dir_config.root_output_dir / 'merge_info.pqt'}")

    """
    --- Third Step ---
    Deal with special designs.
    """
    special_designs_0 = dir_config_0.read_special_designs(dir_config_0)
    special_designs_1 = dir_config_1.read_special_designs(dir_config_1)
    out_special_designs_dict = {"legend": [], "design_numbers": [], "src_dir_idx": []}
    for idx, src_special_designs in enumerate([special_designs_0, special_designs_1]):
        for leg, dn in zip(src_special_designs["legend"], src_special_designs["design_numbers"]):
            if dn in [to_copy_dn_0, to_copy_dn_1][0]:
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

    """
    --- Fourth Step ---
    Merge the databases.
    We need to copy only the design numbers that have been copied and remap their design numbers to the new ones.
    """
    # TODO:
    # Also merge synth and swact dbs to accelerate usage of the merged df
    # That would allow to avoid re-performing the full analysis after every merge
    # for row in write_merge_db.iterrows():
    #     idx = row["soure_idx"]


if __name__ == "__main__":
    main()
