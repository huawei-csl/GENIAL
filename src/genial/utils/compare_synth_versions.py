import argparse
from loguru import logger

import json
import pandas as pd
import matplotlib.pyplot as plt


from genial.config.config_dir import ConfigDir
from genial.config.arg_parsers import double_analyzer_parser

import genial.experiment.file_parsers as file_parsers
import genial.utils.utils as utils

import numpy as np
import shutil

plt.rcParams["font.size"] = 20

if __name__ == "__main__":
    # def main():
    """
    The goal of this script is to enable comparing two different synthesis version.
    Warning: The script excpets the original datasets to be the same.
    """

    # Get argument parser
    args_0, args_1, args_dict = double_analyzer_parser()

    dir_config_0 = ConfigDir(is_analysis=True, **args_0)
    dir_config_1 = ConfigDir(is_analysis=True, **args_1)

    output_dir_name_0 = dir_config_0.args_dict.get("output_dir_name")
    output_dir_name_1 = dir_config_1.args_dict.get("output_dir_name")

    arg_parser = argparse.ArgumentParser()

    args = arg_parser.parse_known_args()
    args_dict_new = vars(args[0])
    args_dict.update(args_dict_new)

    # Check if there are generated designs that are not present in both output dir
    all_encodings_df_0 = file_parsers.read_all_existing_encodings_v2(root_output_path=dir_config_0.root_output_dir)
    all_encodings_df_1 = file_parsers.read_all_existing_encodings_v2(root_output_path=dir_config_1.root_output_dir)

    all_encodings_df_0 = all_encodings_df_0.add_suffix("_0")
    all_encodings_df_1 = all_encodings_df_1.add_suffix("_1")

    # Search for uniques, just to inform the user
    logger.info(f"Searching for unique design numbers ...")
    duplicate_df = pd.merge(
        left=all_encodings_df_0,
        right=all_encodings_df_1,
        how="inner",
        left_on="enc_dict_str_0",
        right_on="enc_dict_str_1",
    )

    df_0_unique = all_encodings_df_0[~all_encodings_df_0["design_number_0"].isin(duplicate_df["design_number_0"])]
    df_1_unique = all_encodings_df_1[~all_encodings_df_1["design_number_1"].isin(duplicate_df["design_number_1"])]

    if len(df_0_unique) > 0 or len(df_1_unique) > 0:
        logger.warning(f"Some generated designs are present in either of the datasets, but not in both.")
        # TODO: log more useful information

    # Check design numbers in both datasets
    if (duplicate_df["design_number_0"] != duplicate_df["design_number_1"]).sum() > 0:
        # Clean special designs issues
        # There might be some special design present in one dataset but not in the other
        diff_duplicate_df = duplicate_df[duplicate_df["design_number_0"] != duplicate_df["design_number_1"]]
        try:
            special_designs_0 = set(json.load(dir_config_0.special_designs_filepath.open("r"))["design_numbers"])
        except Exception:
            special_designs_0 = None
        try:
            special_designs_1 = set(json.load(dir_config_1.special_designs_filepath.open("r"))["design_numbers"])
        except Exception:
            special_designs_1 = None

        if not (special_designs_1 is None or special_designs_0 is None):
            clean_idxs = []
            for idx, row in diff_duplicate_df.iterrows():
                print(row)
                if row["design_number_0"] in special_designs_0 and row["design_number_1"] in special_designs_1:
                    clean_idxs.append(idx)

            duplicate_df = duplicate_df.drop(clean_idxs)

        # If there are still issues, raise a warning
        diff_duplicate_df = duplicate_df[duplicate_df["design_number_0"] != duplicate_df["design_number_1"]]
        if len(diff_duplicate_df) > 0:
            logger.warning("Some encodings have different design numbers in the two datasets:")
            utils.log_dataframe(duplicate_df[["design_number_0", "design_number_1"]])
    # Note: from now on, the design numbers should be the same in both _0 and _1 datasets (if filtered from duplicate_df) if the datasets compared come from the same generation (which should be the case because we want to compare the synthesis results)

    # Plot the difference of metrics in both datasets
    synth_db_path_0 = dir_config_0.analysis_out_dir / "synth_analysis.db.pqt"
    synth_db_path_1 = dir_config_1.analysis_out_dir / "synth_analysis.db.pqt"

    synth_db_0 = pd.read_parquet(synth_db_path_0)
    synth_db_1 = pd.read_parquet(synth_db_path_1)

    # Filter the synth_db to only keep the designs that are in the duplicate_df
    synth_db_0 = synth_db_0[synth_db_0["design_number"].isin(duplicate_df["design_number_0"])]
    synth_db_1 = synth_db_1[synth_db_1["design_number"].isin(duplicate_df["design_number_1"])]

    # Keep only all design numbers that are all in both synth_db_0 and synth_db_1
    synth_db_0 = synth_db_0[synth_db_0["design_number"].isin(synth_db_1["design_number"])]
    synth_db_1 = synth_db_1[synth_db_1["design_number"].isin(synth_db_0["design_number"])]

    # Reorder the databases so that all the design numbers have the same index on both db
    synth_db_0 = synth_db_0.set_index("design_number")
    synth_db_1 = synth_db_1.set_index("design_number")
    synth_db_0 = synth_db_0.sort_index()
    synth_db_1 = synth_db_1.sort_index()

    # Restore design_number as a column
    synth_db_0 = synth_db_0.reset_index()
    synth_db_1 = synth_db_1.reset_index()

    # Check that the design numbers are the same in both datasets
    if not synth_db_0["design_number"].equals(synth_db_1["design_number"]):
        raise ValueError("The design numbers are not the same in both datasets")

    # Plot nb_transistors in _0 versus nb_transistors in _1
    relative_distance = synth_db_0["nb_transistors"] / synth_db_1["nb_transistors"]
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    main_scatter_handles = ax.scatter(
        y=synth_db_0["nb_transistors"], x=synth_db_1["nb_transistors"], c=relative_distance
    )
    ax.set_ylabel(f"nb_transistors in {output_dir_name_0}")
    ax.set_xlabel(f"nb_transistors in {output_dir_name_1}")
    ax.set_title(f"Comparison of nb_transistors between\n{output_dir_name_0}\nand\n{output_dir_name_1}")
    cbar = plt.colorbar(main_scatter_handles)
    cbar.set_label(f"ratio: y/x")

    # Plot the line y=x in red
    ax.axline((0, 0), slope=0.5, color="steelblue", linestyle="--", label="y=0.5x")
    ax.axline((0, 0), slope=1, color="green", linestyle="--", label="y=x")
    ax.axline((0, 0), slope=1.5, color="goldenrod", linestyle="--", label="y=1.5x")
    ax.axline((0, 0), slope=2, color="firebrick", linestyle="--", label="y=2x")
    # ax.axline((0, 0), slope=3, color="orange", linestyle="--", label="y=2x")

    # Make sure both the x and y axis start at 0
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])

    plt.legend()
    plt.tight_layout()
    plt.minorticks_on()
    plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
    plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

    figpath = (
        dir_config_0.analysis_out_dir / f"nb_transistors_{output_dir_name_0}_vs_nb_transistors_{output_dir_name_1}.png"
    )
    plt.savefig(figpath)
    logger.info(f"Saved figure to {figpath}")

    # Check for very closenb_transistors values
    # Make sure no wrong synthesis versions are presents
    # If there are, delete their synth folder
    cond = np.isclose(synth_db_0["nb_transistors"], synth_db_1["nb_transistors"])
    design_numbers = synth_db_0[cond]["design_number"]

    for dir_config in [dir_config_0, dir_config_1]:
        to_delete_dirpaths = []
        synth_versions = set()
        for dn in design_numbers:
            synth_dirpath = dir_config.synth_out_dir / f"res_{dn}"
            if synth_dirpath.exists():
                for filepath in synth_dirpath.iterdir():
                    if "synth_version" in filepath.name:
                        synth_version = filepath.name.split("_")[-1]
                        if synth_version == "0":
                            synth_versions.add(synth_version)
                            to_delete_dirpaths.append(synth_dirpath)
                            break

        if len(to_delete_dirpaths) > 0:
            do_delete = input(
                f"Warning: {len(to_delete_dirpaths)} design with synth versions {synth_versions} have been found in `output_dir_name` {dir_config.output_dir_name}, do you want to delete their synth folders? y/n"
            )
            if do_delete[0].lower() == "y":
                while len(to_delete_dirpaths) > 0:
                    dirpath = to_delete_dirpaths.pop()
                    shutil.rmtree(dirpath)
            logger.info(f"All folders successfully deleted.")

# if __name__ == "__main__":
# test = main()
