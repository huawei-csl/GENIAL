from typing import Any
import argparse
from time import time
from loguru import logger

import pandas as pd
import matplotlib.pyplot as plt

from genial.config.config_dir import ConfigDir
from genial.config.arg_parsers import analyzer_parser

import genial.experiment.file_parsers as file_parsers
from genial.experiment.encoding_distance import EncodingsDistanceHelper
from genial.training.elements.score_tools import ScoreComputeHelper
from genial.utils.utils import viz_encoding_as_tensor

import seaborn as sns

import ast
from tqdm import tqdm

plt.rcParams["font.size"] = 15


def from_enc_dict_str_to_enc_dict_str(enc_dict_str: str) -> dict[str, str]:
    """
    Converts a string representation of a dictionary of encoding to a dictionary of encoding.
    """
    return ast.literal_eval(enc_dict_str)


def from_encoding_dict_to_binary_repr_list(encoding_dict: dict[str, Any], invert_polarity: bool = False) -> list[str]:
    """
    Convert a dictionary of encoding to a list of binary representations by keeping the order.
    """
    binary_repr_list = []
    for key, value in encoding_dict.items():
        if not invert_polarity:
            binary_repr_list.append(value)
        else:
            binary_repr_list.append(value.replace("1", "2").replace("0", "1").replace("2", "0"))
    return binary_repr_list


if __name__ == "__main__":
    """
    The goal of this script is to evaluate the correlation between encoding distance and transistor number.
    Warning: The script excpets the original datasets to be the same.
    """

    # Setup some script parameters
    column_name = "in_enc_dict"
    target_metric = "nb_transistors"

    # Get argument parser
    args_dict = analyzer_parser()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--force", action="store_true", help="When set, it enforces re-computing all distances.")

    args = arg_parser.parse_known_args()
    args_dict_new = vars(args[0])
    args_dict.update(args_dict_new)

    # Setup Config Dir
    dir_config = ConfigDir(is_analysis=True, **args_dict)

    # Check if there are generated designs that are not present in both output dir
    all_encodings_df = file_parsers.read_all_existing_encodings_v2(root_output_path=dir_config.root_output_dir)

    # Get list of synthesized_design_numbers and reduce list of encodings
    synthed_design_numbers = file_parsers.get_list_of_synth_designs_number(dir_config)

    synth_db_path = dir_config.analysis_out_dir / "synth_analysis.db.pqt"
    synth_df = pd.read_parquet(synth_db_path)
    synthed_analyzed_design_numbers = set(synth_df["design_number"])
    if len(synthed_analyzed_design_numbers) != len(synthed_design_numbers):
        logger.warning(
            f"There are new designs synthesized that have not been analyzed yet. "
            f"You can launch the analyzer if you want to add them to the plots of this script."
        )

    reduced_encodings_df = all_encodings_df[all_encodings_df["design_number"].isin(synthed_analyzed_design_numbers)]

    # Check if the distance analysis df exists. If it does, open it
    output_dirpath = dir_config.analysis_out_dir / "encodings_distance_analysis"
    does_exist = False
    if not output_dirpath.exists():
        output_dirpath.mkdir()
    synth_db_with_distances_path = output_dirpath / "synth_db_with_distances.db.pqt"
    if synth_db_with_distances_path.exists() and not args_dict.get("force", False):
        does_exist = True
    # elif synth_db_with_distances_path.exists() and not args_dict.get("force", False):
    # logger.warning(f"Removing the synthesized db with distances file ...")
    # shutil.rmtree(synth_db_with_distances_path)

    if does_exist and not args_dict.get("force", False):
        synth_df_with_distances = pd.read_parquet(synth_db_with_distances_path)
        remaining_design_numbers = set(reduced_encodings_df["design_number"]) - set(
            synth_df_with_distances["design_number"]
        )
    else:
        synth_df_with_distances = None
        remaining_design_numbers = set(reduced_encodings_df["design_number"])

    if len(remaining_design_numbers) > 0:
        # Plot the difference of metrics in both datasets

        reduced_synth_df = synth_df[synth_df["design_number"].isin(remaining_design_numbers)]

        # Order the reduced_encodings_df on the synth_df database based on the design number
        reduced_encodings_df = (
            reduced_encodings_df.set_index("design_number").loc[reduced_synth_df["design_number"]].reset_index()
        )
        assert reduced_encodings_df["design_number"].equals(reduced_synth_df["design_number"]), (
            "The design numbers are not the same, is the reduced_synth_df up to date? (i.e., did you run the task_analyzer before running this script?)."
        )

        def map_dict_str_to_list_str(row):
            return from_encoding_dict_to_binary_repr_list(from_enc_dict_str_to_enc_dict_str(row[column_name]))

        def map_dict_str_to_list_str_inveted_polarity(row):
            return from_encoding_dict_to_binary_repr_list(
                from_enc_dict_str_to_enc_dict_str(row[column_name]), invert_polarity=True
            )

        reduced_encodings_df["repr_list"] = reduced_encodings_df.apply(map_dict_str_to_list_str, axis=1).to_list()
        reduced_encodings_df["negative_repr_list"] = reduced_encodings_df.apply(
            map_dict_str_to_list_str_inveted_polarity, axis=1
        ).to_list()

        # Get the best encoding (i.e. the encoding which has the minimum `nb_transistors`)
        best_design = synth_df.iloc[synth_df["nb_transistors"].argmin()]
        best_design_number = best_design["design_number"]
        best_encodings = all_encodings_df[all_encodings_df["design_number"] == best_design_number]
        best_repr_list = best_encodings.apply(map_dict_str_to_list_str, axis=1).to_list()[0]

        # Compute the distances
        logger.info(f"Measuring all distances between the best encoding and other encodings. This can take a while ...")
        start_time = time()
        distances_dicts = []
        with tqdm(total=len(reduced_encodings_df), desc=f"x1| Measuring all distances") as pbar:  # Progress bar
            for idx, encoding in reduced_encodings_df.iterrows():
                hamming_distance_inverted = EncodingsDistanceHelper.list_smart_hamming_distance(
                    best_repr_list, encoding["negative_repr_list"]
                )
                hamming_distance_default = EncodingsDistanceHelper.list_smart_hamming_distance(
                    best_repr_list, encoding["repr_list"]
                )
                if hamming_distance_default <= hamming_distance_inverted:
                    is_inverted = False
                    repr_list = encoding["repr_list"]
                else:
                    is_inverted = True
                    repr_list = encoding["negative_repr_list"]

                # TODO: change losses so that they can be column rotation invariant
                # Or simply use the column permutation loss as color
                # It's probably better to use a columns wise-kendall-tau distance
                distances_dict = {
                    "design_number": encoding["design_number"],
                    "in_enc_dict": encoding["in_enc_dict"],
                    "edit_distance": EncodingsDistanceHelper.list_edit_distance(best_repr_list, repr_list),
                    "kendall_tau_distance": EncodingsDistanceHelper.list_kendall_tau_distance(
                        best_repr_list, repr_list
                    ),
                    "column_kendall_tau_distance": EncodingsDistanceHelper.list_column_kendall_tau_distance(
                        best_repr_list, repr_list, weight=True
                    ),
                    "smart_kendall_tau_distance": EncodingsDistanceHelper.list_smart_kendall_tau_distance(
                        best_repr_list, repr_list
                    ),
                    "permutation_distance": EncodingsDistanceHelper.list_permutation_distance(
                        best_repr_list, repr_list
                    ),
                    "column_permutation_distance": EncodingsDistanceHelper.list_column_permutation_distance(
                        best_repr_list, repr_list, weight=True
                    ),
                    "flattened_edit_distance": EncodingsDistanceHelper.flattened_list_edit_distance(
                        best_repr_list, repr_list
                    ),
                    "hamming_distance": EncodingsDistanceHelper.list_hamming_distance(best_repr_list, repr_list),
                    "smart_hamming_distance": EncodingsDistanceHelper.list_smart_hamming_distance(
                        best_repr_list, repr_list
                    ),
                    "dice_loss": EncodingsDistanceHelper.list_dice_loss(best_repr_list, repr_list),
                    "dist_on_inverted": is_inverted,
                    "final_distance": EncodingsDistanceHelper.list_smart_composed_distance(
                        best_repr_list, encoding["repr_list"], nb_cycles=0
                    ),
                    "smarter_final_distance": EncodingsDistanceHelper.list_smarter_composed_distance(
                        best_repr_list, encoding["repr_list"], nb_cycles=2
                    ),
                    "smarter_kendall_tau_distance": EncodingsDistanceHelper.list_smarter_kendall_tau_distance(
                        best_repr_list, encoding["repr_list"], nb_cycles=2
                    ),
                }
                distances_dict.update(
                    {
                        # "total_kendall_tau_distance": distances_dict["column_kendall_tau_distance"] + distances_dict["kendall_tau_distance"],
                        "total_permutation_distance": distances_dict["column_permutation_distance"]
                        + distances_dict["permutation_distance"],
                    }
                )
                distances_dicts.append(distances_dict)
                pbar.update(1)

        distances_df = pd.DataFrame(distances_dicts)
        new_synth_df_with_distances = pd.merge(reduced_synth_df, distances_df, on="design_number", how="left")
        new_synth_df_with_distances["is_best"] = (
            new_synth_df_with_distances["design_number"] == best_encodings["design_number"].item()
        )
        logger.info(
            f"Measured all distances for {len(new_synth_df_with_distances)} encodings in {time() - start_time:.2f} seconds."
        )

        if synth_df_with_distances is not None:
            synth_df_with_distances = pd.concat(
                [synth_df_with_distances, new_synth_df_with_distances], ignore_index=True
            ).reset_index()
        else:
            synth_df_with_distances = new_synth_df_with_distances

        synth_df_with_distances.to_parquet(synth_db_with_distances_path, index=False)
        logger.info(
            f"Saved {len(synth_df_with_distances)} encodings synth_db with distances to {synth_db_with_distances_path}"
        )
    else:
        logger.warning(f"No new distances were computed.")

    """ PLOTS """

    # Filter some data out
    # _synth_df_with_distances = synth_df_with_distances.drop(synth_df_with_distances[synth_df_with_distances["is_best"]].index, axis=0)
    _synth_df_with_distances = synth_df_with_distances

    # Plot some of the outstanding encodings
    plot_cond = _synth_df_with_distances["smarter_final_distance"] > 0.2
    to_plot_df = _synth_df_with_distances[plot_cond]
    plot_cond = to_plot_df["nb_transistors"] < 600
    to_plot_df = to_plot_df[plot_cond]

    for idx, row in to_plot_df.iterrows():
        enc_dict = ast.literal_eval(row["encodings_input"])
        inverted_suffix = "\nINVERTED" if row["dist_on_inverted"] else ""
        viz_encoding_as_tensor(
            enc_dict,
            fig_title=f"Encoding {row['design_number']} - {row['nb_transistors']} transistors\n dice_loss={row['dice_loss']} smarter_final_distance={row['smarter_final_distance']}\nfinal_distance={row['final_distance']:0.2f} smart_kendall_tau_distance={row['smart_kendall_tau_distance']:0.2f}{inverted_suffix}",
            filepath=output_dirpath
            / f"outstanding_encoding_hamming{row['dice_loss']}_trans{row['nb_transistors']}.png",
            log=True,
        )

    # Plot distance versus target_metric
    for top_idx, db in enumerate([_synth_df_with_distances, to_plot_df]):
        db_suffix = {0: "", 1: "_outstanding"}
        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        for idx, distance_name in enumerate(
            ["final_distance", "smart_kendall_tau_distance", "smarter_final_distance", "smarter_kendall_tau_distance"]
        ):
            ax = axes[idx % 2, idx // 2]
            # Standardize datapoints
            db[distance_name + "_scaled"] = ScoreComputeHelper.scaler_map["raw"](db[distance_name])
            marker_map = {0: "o", 1: "s"}
            color_map = {0: "darkcyan", 1: "peru"}
            label_map = {0: "non inverted", 1: "inverted"}
            for idx, sub_db in enumerate([db[~db["dist_on_inverted"]], db[db["dist_on_inverted"]]]):
                sns.scatterplot(
                    y=distance_name + "_scaled",
                    x=target_metric,
                    data=sub_db,
                    ax=ax,
                    alpha=0.6,
                    marker=marker_map[idx],
                    c=color_map[idx],
                    label=label_map[idx],
                )

            ax.legend()
            ax.set_xlabel(type)
            ax.set_ylabel(distance_name)
            ax.set_xlabel(f"{target_metric}")
            ax.minorticks_on()
            ax.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
            ax.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

        plt.suptitle(
            f"Min(inverted/non-inverted) {db_suffix[top_idx]} Encodings Distance to Best vs {target_metric}\n{dir_config.experiment_name}\n{dir_config.output_dir_name}"
        )
        plt.tight_layout()

        save_filepath = output_dirpath / f"distances_vs_{target_metric}{db_suffix[top_idx]}.png"
        plt.savefig(save_filepath, dpi=300)
        plt.close()
        logger.opt(colors=True).info(f"<yellow>Plot</yellow> Encodings Distances versus {target_metric} plot saved at:")
        logger.info(save_filepath)

    print("DONE.")
