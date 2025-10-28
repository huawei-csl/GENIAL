import sys
from loguru import logger

import pandas as pd
import numpy as np

from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt


from genial.config.config_dir import ConfigDir
from genial.config.arg_parsers import double_analyzer_parser
from genial.experiment.task_analyzer import Analyzer

from genial.utils.utils import load_database


def dictstr_to_hash(d: str):
    # dictionary = eval(d)
    # json_str = json.dumps(dictionary, sort_keys=True)
    return hash(str)


def main():
    # Get argument parser
    args_0, args_1, args_dict = double_analyzer_parser()

    dir_config_0 = ConfigDir(is_analysis=True, **args_0)
    dir_config_1 = ConfigDir(is_analysis=True, **args_1)

    # Setup output_directory
    work_dir_path = dir_config_0.work_dir
    meta_output_dir_top = work_dir_path / "meta_output"
    output_dir_root = (
        meta_output_dir_top
        / f"{args_dict['experiment_name_0']}-{args_dict['output_dir_name_0']}---{args_dict['experiment_name_1']}-{args_dict['output_dir_name_1']}"
    )
    if not output_dir_root.exists():
        output_dir_root.mkdir(parents=True)
        logger.info(f"Output directory was made at:")
        logger.info(f"{output_dir_root}")

    # Save the command used to run the script
    with open(output_dir_root / "exec.cmd", "w") as f:
        command = f"python {' '.join(sys.argv)}"
        f.write(command)
        logger.info(f"Run command saved in {output_dir_root / 'exec.cmd'}")

    if args_dict.get("rebuild_meta_db", False):
        analyzer_0 = Analyzer(dir_config=dir_config_0, reset_logs=False)
        analyzer_1 = Analyzer(dir_config=dir_config_1, reset_logs=False)

        logger.info(f"All analyzers have successfully been setup.")

        # Get valuable information
        test_types_0 = set(analyzer_0.test_type_names)
        test_types_1 = set(analyzer_1.test_type_names)
        test_types_shared = test_types_0.intersection(test_types_1)

        all_design_hash_0 = set(analyzer_0.swact_df["encodings_input"])
        all_design_hash_1 = set(analyzer_1.swact_df["encodings_input"])
        all_design_hash_shared = all_design_hash_0.intersection(all_design_hash_1)

        # Reduce all databases
        for analyzer in [analyzer_0, analyzer_1]:
            analyzer.swact_df = analyzer.swact_df[analyzer.swact_df["encodings_input"].isin(all_design_hash_shared)]
            analyzer.synth_df = analyzer.synth_df[
                analyzer.synth_df["design_number"].isin(analyzer.swact_df["design_number"])
            ]

        for test_type_name in test_types_shared:
            # Check existence of database
            db_filename = f"merged_score_{test_type_name}.db.csv"
            db_filepath = output_dir_root / db_filename

            logger.info(f"Evaluating score database for test type {test_type_name} ...")
            dfs = []
            for idx, analyzer in enumerate([analyzer_0, analyzer_1]):
                analyzer.format_databases()
                analyzer.align_databases()

                #  Reduce current swact_df to
                sub_swact_df: pd.DataFrame
                sub_swact_df = analyzer.swact_df[analyzer.swact_df["test_type"] == test_type_name]
                total_df = pd.merge(
                    sub_swact_df, analyzer.synth_df, how="left", on="design_number", suffixes=["", "_synth"]
                )
                total_df = total_df.loc[:, ~total_df.columns.duplicated()]

                # Compute score
                score_df = pd.DataFrame()
                # score_df["scores"] = total_df["swact_weighted_total"] * total_df["nb_transistors"]
                score_df["scores"] = total_df["nb_transistors"] / total_df["max_depth"]

                # score_df["design_hash"] = total_df["encodings_input"].map(dictstr_to_hash)
                score_df["design_hash"] = total_df["encodings_input"]

                # Sort the array
                score_df = score_df.sort_values("scores")

                # Add ranking
                score_df["ranking"] = np.arange(len(score_df))

                # Remember final df
                dfs.append(score_df)

            # Align dataframes
            score_df_0, score_df_1 = dfs
            merged_score_df = pd.merge(score_df_0, score_df_1, how="left", on="design_hash", suffixes=["_0", "_1"])

            # Add test_type_name
            merged_score_df["test_type"] = test_type_name

            merged_score_df.to_parquet(db_filepath, index=False)
            logger.info(f"Database has been saved at:")
            logger.info(db_filepath)

    else:
        logger.warning(f"Argument `rebuild_meta_db` was not given. Analyzer loading has been skipped.")

    # Extract all available databases
    db_filepaths = []
    for filepath in output_dir_root.iterdir():
        if filepath.is_file():
            if filepath.name.endswith(".db.csv") or filepath.name.endswith(".db.pqt"):
                db_filepaths.append(filepath)

    for db_dilepath in db_filepaths:
        merged_score_df = load_database(db_dilepath)
        merged_score_df.astype(
            {
                "ranking_0": int,
                "ranking_1": int,
            }
        )
        test_type_name = merged_score_df["test_type"].unique()[0]
        # Reduce df length (keep only best elements)

        kts_coeffs = []
        sp_coeffs = []
        lengths = range(1000, len(merged_score_df), 1000)
        for length in list(lengths):
            sub_length_df = merged_score_df.head(length)

            # Compute kendal tau's coefficient
            kendall_tau_results = kendalltau(sub_length_df["ranking_0"], sub_length_df["ranking_1"])
            kendall_tau_coeff = kendall_tau_results[0]

            # Compute the p-value (unused here)
            # p_value = kendall_tau_results[1]

            # Compute spearman rank correlation
            spearman_results = spearmanr(sub_length_df["ranking_0"], sub_length_df["ranking_1"])
            spearman_coeff = spearman_results[0]

            logger.info(
                f"Test type: {test_type_name} | DF length {len(sub_length_df)} | Kendall Tau's Coeff {kendall_tau_coeff:.2f} | Spearman coeff {spearman_coeff:.2f}"
            )

            # Save for plot
            kts_coeffs.append(kendall_tau_coeff)
            sp_coeffs.append(spearman_coeff)

        # Make the final plot
        logger.info(f"Realizing the plots ...")
        ax0: plt.axes
        ax1: plt.axes
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
        ax0.hist2d(
            merged_score_df["ranking_0"].to_numpy(),
            merged_score_df["ranking_1"].to_numpy(),
            bins=(20, 20),
            cmap="viridis",
        )
        ax0.hist2d(
            merged_score_df["ranking_0"].to_numpy(),
            merged_score_df["ranking_1"].to_numpy(),
            bins=(20, 20),
            cmap="viridis",
        )
        ax0.scatter(
            merged_score_df["ranking_0"].to_numpy(),
            merged_score_df["ranking_1"].to_numpy(),
            color="white",
            alpha=0.05,
            s=3,
        )
        ax0.set_xlabel(f"ranking_0 - {args_dict.get('output_dir_name_0')}")
        ax0.set_ylabel(f"ranking_1 - {args_dict.get('output_dir_name_1')}")
        ax0.set_title(
            f"Design ranking for different run configuration (see axis labels)\ntest_type:{test_type_name} | number_designs:{len(merged_score_df)}"
        )

        ax1.scatter(lengths, kts_coeffs, label="Kendall Tau's Coefficient")
        ax1.scatter(lengths, sp_coeffs, label="Spearman Coefficient")
        ax1.set_ylabel(f"Correlation coefficients values")
        ax1.set_xlabel(f"Number of designs taken (from higher to lower ranking_0)")
        ax1.legend()

        plt.minorticks_on()
        plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
        plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)
        plt.tight_layout()

        filepath = output_dir_root / f"plot_ranking_correlation-{test_type_name}.png"
        plt.savefig(filepath, dpi=200)
        logger.info(f"Figure saved at {filepath}")


if __name__ == "__main__":
    main()
