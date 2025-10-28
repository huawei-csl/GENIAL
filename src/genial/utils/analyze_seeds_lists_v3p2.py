# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

from loguru import logger
import argparse
from typing import Any
import json
import seaborn as sns

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from genial.config.config_dir import ConfigDir
from genial.config.arg_parsers import get_default_parser
import genial.experiment.file_parsers as file_parsers

plt.rcParams["font.size"] = 18


def get_is_min_column(
    df: pd.DataFrame, column_name: str = "design_number", metric_name: str = "nb_transistors"
) -> pd.Series:
    min_trans_dict = df.groupby(column_name)[metric_name].min().to_dict()
    return df[metric_name] == df[column_name].map(min_trans_dict)


def parse_args() -> dict[str, Any]:
    # Add arguments for this script
    arg_parser = argparse.ArgumentParser()

    args_dict = vars(arg_parser.parse_known_args()[0])

    # Add standard analyzer arguments
    default_args_dict = get_default_parser()
    args_dict.update(default_args_dict)

    return args_dict


if __name__ == "__main__":
    args_dict = parse_args()
    dir_config = ConfigDir(is_analysis=True, **args_dict)

    # Initialize variables
    total_nb_designs = 0

    # Get all number of transistors and all seeds
    flattened_data_pre_synth = []
    flattened_data_post_synth = []
    run_execution_info = []
    for synth_design_dirpath in dir_config.synth_out_dir.iterdir():
        seeds_filepath = synth_design_dirpath / "nb_trans_and_seeds.json"

        if seeds_filepath.exists():
            design_number = file_parsers.extract_design_number_from_path(seeds_filepath)
            total_nb_designs += 1
            seeds_dict_list = json.load(seeds_filepath.open("r"))
            for d in seeds_dict_list:
                row = {"design_number": design_number, "nb_transistors": d["nb_transistors"], "seeds": d["seeds"]}
                flattened_data_post_synth.append(row)

            origin_seeds_filepath = synth_design_dirpath / "launcher_seeds.json"

            if origin_seeds_filepath.exists():
                design_number = file_parsers.extract_design_number_from_path(origin_seeds_filepath)
                origin_seeds_dict_list = json.load(origin_seeds_filepath.open("r"))
                for d in origin_seeds_dict_list:
                    row = {"design_number": design_number, "nb_transistors": d["nb_transistors"], "seeds": d["seeds"]}
                    flattened_data_pre_synth.append(row)

            run_execution_info_filepath = synth_design_dirpath / "run_execution_info.db.pqt"

            if run_execution_info_filepath.exists():
                run_exec_dict = pd.read_parquet(run_execution_info_filepath)
                run_exec_dict["design_number"] = design_number
                run_execution_info.extend(run_exec_dict.to_dict(orient="records"))

    # Build Dataframes
    full_seeds_post_synth = pd.DataFrame(flattened_data_post_synth)
    full_seeds_pre_synth = pd.DataFrame(flattened_data_pre_synth)
    run_execution_info = pd.DataFrame(run_execution_info)

    # Associate an id with a set of seeds (create the seed_list to id map)
    full_seeds_pre_synth["seeds_list_str"] = full_seeds_pre_synth["seeds"].map(str)
    full_seeds_post_synth["seeds_list_str"] = full_seeds_post_synth["seeds"].map(str)

    # Note: uniques are returned in order of appearance
    seed_list_id_map = {k: v for v, k in enumerate(full_seeds_pre_synth["seeds_list_str"].unique())}

    full_seeds_post_synth["seeds_list_id"] = full_seeds_post_synth["seeds_list_str"].map(seed_list_id_map)

    """ ANALYSIS """
    # Get number of times a recipe is the best one
    # Get best recipes for each design number
    min_score_per_design_number = full_seeds_post_synth.groupby("design_number")["nb_transistors"].min()

    # Find which recipes are the best one for each design
    full_seeds_post_synth["is_best"] = get_is_min_column(
        full_seeds_post_synth, column_name="design_number", metric_name="nb_transistors"
    )
    best_recipe_count = full_seeds_post_synth.groupby("seeds_list_id")["is_best"].sum().reset_index()
    only_best_post_synth = full_seeds_post_synth[full_seeds_post_synth["is_best"]]

    # Get median nb trans when recipe is the best
    best_recipe_median_trans = (
        only_best_post_synth.groupby("seeds_list_id")["nb_transistors"]
        .median()
        .reset_index()
        .rename(columns={"nb_transistors": "nb_trans_median_if_best"})
    )

    # Get min nb trans when recipe is the best
    best_recipe_min_trans = (
        only_best_post_synth.groupby("seeds_list_id")["nb_transistors"]
        .min()
        .reset_index()
        .rename(columns={"nb_transistors": "min_if_best"})
    )

    # Get global median nb trans
    all_recipe_median_trans = (
        full_seeds_post_synth.groupby("seeds_list_id")["nb_transistors"]
        .median()
        .reset_index()
        .rename(columns={"nb_transistors": "nb_trans_median_all"})
    )

    combined_data = pd.merge(best_recipe_min_trans, best_recipe_median_trans, on="seeds_list_id")
    combined_data = pd.merge(combined_data, all_recipe_median_trans, on="seeds_list_id")

    melted_data = combined_data.melt(
        id_vars="seeds_list_id",
        value_vars=["min_if_best", "nb_trans_median_if_best", "nb_trans_median_all"],
        var_name="nb_trans_metrics",
        value_name="nb_transistors",
    )

    # Get failure rate of a recipe
    success_counts = run_execution_info.groupby("seed_id")["success"].sum()
    failure_counts = (total_nb_designs - success_counts).reset_index().rename(columns={"success": "failure_count"})

    """ DO PLOTS """
    plot_dirpath = dir_config.analysis_out_dir / "flowy_analysis"
    if not plot_dirpath.exists():
        plot_dirpath.mkdir()

    ###### Box Plot: nb_trans vs recipe list index
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
    sns.boxplot(x="seeds_list_id", y="nb_transistors", data=full_seeds_post_synth, ax=ax0)
    ax0.set_xlabel(None)
    ax0.set_ylabel("NB Transistors")

    sns.boxplot(x="seed_id", y="run_time_s", data=run_execution_info, ax=ax1)
    ax1.set_xlabel(None)
    ax1.set_ylabel("Run Time")

    ax2 = ax1.twinx()
    ax2.scatter(
        failure_counts["seed_id"], failure_counts["failure_count"], color="violet", marker="x", label="Failure Count"
    )
    # sns.barplot(x='seed_id', y='failure_count', data=failure_counts, ax=ax2)
    ax2.set_ylabel("Failure Rate", color="violet")

    title = (
        f"Nb Transistors Distributions, Run Time and Failure Rate of Seed Lists\n{args_dict.get('experiment_name')} -"
        f" {args_dict.get('output_dir_name')} - n={total_nb_designs}"
    )
    plt.suptitle(title)

    for ax in (ax0, ax1):
        ax.set_xlabel("Seed List ID")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
        ax.minorticks_on()
        ax.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
        ax.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)
    ax2.grid(visible=True, which="major", color="violet", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_filepath = plot_dirpath / "boxplot_nb_transitors_vs_seed_list_indexes.png"
    plt.savefig(save_filepath, dpi=350)
    logger.info(f"Plot saved at {save_filepath}")
    plt.close()

    ###### Transistor Distribution for All Seeds (individual)
    target_metric = "nb_transistors"
    for seed_list_id in seed_list_id_map.values():
        # Plot distribution of cell counts

        reduced_df = full_seeds_post_synth[full_seeds_post_synth["seeds_list_id"] == seed_list_id]
        vals = reduced_df[target_metric].to_numpy(dtype=np.int64)
        fig, ax0 = plt.subplots(1, 1, figsize=(20, 10))
        (n, bins, patches) = ax0.hist(vals, bins=(np.unique(vals).shape[0] + 10))

        ax0.legend()
        ax0.set_xlabel(type)
        ax0.set_yscale("log")
        ax0.set_ylabel("Nb occurences")
        ax0.set_title(
            f"Distribution of {target_metric} for seed_list {seed_list_id}\nn={len(vals)}\n"
            f"{args_dict.get('experiment_name')} - {args_dict.get('output_dir_name')}"
        )

        save_filepath = plot_dirpath / f"nb_transistors_distribution_seed_list_id{seed_list_id}.png"

        plt.tight_layout()
        plt.minorticks_on()
        plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
        plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

        plt.savefig(save_filepath, dpi=300)
        plt.close()
        logger.opt(colors=True).info(
            f"<yellow>Plot</yellow> {target_metric} distribution for iteration {seed_list_id} saved at:"
        )
        logger.info(save_filepath)

    ###### Bar plot best recipe count vs recipe list index
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
    sns.barplot(x="seeds_list_id", y="is_best", data=best_recipe_count, ax=ax0)
    ax0.set_xlabel(None)
    ax0.set_ylabel("Frequency: Best Seed List")

    sns.barplot(x="seeds_list_id", y="nb_transistors", hue="nb_trans_metrics", data=melted_data, ax=ax1)
    ax1.set_xlabel("Seed List ID")
    ax1.set_ylabel("NB Transistors")

    for ax in (ax0, ax1):
        ax.set_xlabel("Seed List ID")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
        ax.minorticks_on()
        ax.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
        ax.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

    title = (
        f"Best Seed List Frequency and Associated Nb Transistors Metrics\n{args_dict.get('experiment_name')} -"
        f" {args_dict.get('output_dir_name')} - n={total_nb_designs}"
    )
    plt.suptitle(title)

    plt.tight_layout()
    save_filepath = plot_dirpath / "barplots_best_seeds_frequency.png"
    plt.savefig(save_filepath, dpi=350)
    logger.info(f"Plot saved at {save_filepath}")
    plt.close()

    ###### Transistor Distribution for All Seeds (individual)
    target_metric = "nb_transistors"
    for seed_list_id in seed_list_id_map.values():
        # Plot distribution of cell counts
        reduced_df = full_seeds_post_synth[full_seeds_post_synth["seeds_list_id"] == seed_list_id]
        vals = reduced_df[target_metric].to_numpy(dtype=np.int64)
        fig, ax0 = plt.subplots(1, 1, figsize=(20, 10))
        (n, bins, patches) = ax0.hist(vals, bins=(np.unique(vals).shape[0] + 10))

        ax0.legend()
        ax0.set_xlabel(type)
        ax0.set_yscale("log")
        ax0.set_ylabel("Nb occurences")
        ax0.set_title(
            f"Distribution of {target_metric} for seed_list {seed_list_id}\nn={len(vals)}\n{args_dict.get('experiment_name')} - {args_dict.get('output_dir_name')}"
        )

        save_filepath = plot_dirpath / f"nb_transistors_distribution_seed_list_id{seed_list_id}.png"

        plt.tight_layout()
        plt.minorticks_on()
        plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
        plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

        plt.savefig(save_filepath, dpi=300)
        plt.close()
        logger.opt(colors=True).info(
            f"<yellow>Plot</yellow> {target_metric} distribution for iteration {seed_list_id} saved at:"
        )
        logger.info(save_filepath)


# if __name__ == "__main__":
#     """
#     Script for analysing the results of synth_v4 (seed search)
#     Simply use with :
#     python src/genial/utils/analyze_seeds_results.py --experiment_name <exp_name> --output_dir_name <output_dir_name>
#     """

#     df = main()
