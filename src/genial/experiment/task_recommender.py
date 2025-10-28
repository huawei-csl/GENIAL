# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

from typing import Any
from pathlib import Path
from time import time
import traceback
from loguru import logger
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
from torch.utils.data import DataLoader

import genial.experiment.file_parsers as file_parsers
from genial.experiment.task_generator import DesignGenerator
from genial.experiment.task_analyzer import analyzer_parser, Analyzer
from genial.experiment.binary_operators import min_value_tc
from genial.config.logging import Logging as logging
import genial.utils.utils as utils


from genial.training.mains.trainer_enc_to_score_value import EncToScoreTrainer
from genial.training.elements.configs import ModelConfig, DatasetConfig
from genial.training.elements.models import FullTransformer
from genial.training.elements.lit.models import LitTransformer
from genial.training.elements.datasets import RawEncodingDataset, SwactedDesignDataset
from genial.training.elements.score_tools import ScoreComputeHelper
from genial.training.elements.utils import setup_analyzer

from genial.experiment.loop_module import LoopModule


from genial.utils.utils import extract_int_string_from_string


class EncodingRecommender(LoopModule):
    __valid_recommender_modes__ = ["score_predictor", "random"]

    def __init__(
        self,
        analyzer: Analyzer | None = None,
        init_logging: bool = False,
        mode: str = "score_predictor",
    ) -> None:
        super().__init__()
        logger.info(f"Setting up EncodingRecommender ...")

        # Check the configuration
        assert mode in self.__valid_recommender_modes__
        self.run_mode = mode

        # Initialize the experiment configuration
        if analyzer is None:
            self.args_dict = EncodingRecommender.parse_args()
            _analyzer = setup_analyzer(**self.args_dict)
            do_setup_model = True
        else:
            _analyzer = analyzer
            do_setup_model = False

        self.analyzer = _analyzer
        self.args_dict = self.analyzer.dir_config.args_dict
        self.dir_config = self.analyzer.dir_config

        self.nb_workers = self.args_dict.get("nb_workers", 64)
        self.batch_size = self.args_dict.get("batch_size", 256)
        self.is_debug = self.args_dict.get("debug", False)
        self.dry_run = self.args_dict.get("dry_run", False)

        # Setup number of designs to generate
        self.keep_percentage = self.args_dict.get("keep_percentage", 10)
        self.nb_new_designs = self.args_dict.get("nb_new_designs", 100)
        self.nb_designs_to_generate = int(100 * self.nb_new_designs / self.keep_percentage)
        assert self.nb_designs_to_generate >= 1.0, (
            f"Number of designs to generate (nb_new_designs * keep_percentage) should be > 1"
        )
        logger.info(f"{self.nb_new_designs} new design will be kept (out of {self.nb_designs_to_generate}).")
        # Note: number of designs kept is self.nb_design_to_generate*self.keep_percentage/100

        # Open the training database
        training_database_filepath = self.analyzer.dir_config.root_output_dir / "training_results.pqt"
        if training_database_filepath.exists():
            self.train_df = utils.load_database(training_database_filepath)
        else:
            self.train_df = None
            logger.info(f"Training database has not been found.")

        # Setup the model
        if do_setup_model:
            self.setup_model(self.args_dict.get("trainer_version_number", None))
        else:
            logger.warning(
                f"The model has not been initialized. Please do so by calling manually setup_model on this instance."
            )
            self.model_ckpt_path = None
            self.lit_model = None

        # Setup the recommendation database
        self.setup_recom_df()

        if init_logging:
            # Setup the logger
            log_dirpath = self.dir_config.recommender_out_dir_ver
            if log_dirpath is not None:
                logging().init_logging(log_dirpath=log_dirpath, mode="trainer", reset=False)

        logger.info(f"EncodingRecommender initialized.\n\n")

        self.seed = self.args_dict.get("seed", 626)

    def get_recom_db_path(self) -> Path:
        """Helper function to access the path of the recommendation database."""
        return self.dir_config.root_output_dir / "recommendation.db.pqt"

    def setup_recom_df(self) -> None:
        self.recom_df = None
        self.recom_db_file_path = None

        check_load_db = (
            self.run_mode == "score_predictor" and self.model_ckpt_path is not None
        ) or self.run_mode == "random"
        if check_load_db:
            self.recom_db_file_path = self.get_recom_db_path()
            if self.recom_db_file_path.exists():
                self.recom_df = utils.load_database(self.recom_db_file_path)
                logger.info(f"Recommendation DB has been loaded from file:")
                logger.info(self.recom_db_file_path)

    def get_trainer_version_number(self, trainer_version_number: int | None = None) -> int | None:
        """Helper function to always get the right trainer version number."""
        if hasattr(self, "tb_logger"):
            _trainer_version_number = self.tb_logger.version
            _loaded_from = "tb logger"
        elif trainer_version_number is not None:
            _trainer_version_number = trainer_version_number
            _loaded_from = "function argument"
        elif self.args_dict.get("trainer_version_number", None) is not None:
            _trainer_version_number = self.args_dict.get("trainer_version_number", None)
            _loaded_from = "cli argument"
        else:
            _trainer_version_number = None
            _loaded_from = "None"

        logger.info(f"Trainer version number has been obtained from {_loaded_from}")

        return _trainer_version_number

    def setup_model(self, trainer_version_number: int | None = None, strategy: str | None = None) -> None:
        if self.run_mode == "score_predictor":
            _trainer_version_number = self.get_trainer_version_number(trainer_version_number)

            self.args_dict["trainer_version_number"] = _trainer_version_number
            self.model_config = ModelConfig(
                device_nb=self.args_dict.get("device_nb"),
                args_dict=self.args_dict,
                dir_config=self.dir_config,
                trainer_version_number=_trainer_version_number,
            )
            self.dir_config.setup_trainer_version_output_dirs(trainer_version_number=_trainer_version_number)

            # Get best checkpoint path
            self.model_ckpt_path, _ = EncToScoreTrainer._find_best_checkpoint(
                dir_config=self.dir_config,
                trainer_version_number=self.args_dict["trainer_version_number"],
                strategy=strategy,
            )

            # Init the lightning meta model based on a transformer
            model = FullTransformer(model_config=self.model_config)

            # Load model
            self.lit_model = LitTransformer.load_from_checkpoint(
                self.model_ckpt_path, meta_model=model, model_config=self.model_config
            )

            logger.warning(f"Encoder model has been updated with model version {_trainer_version_number}")

        elif self.run_mode == "random":
            if trainer_version_number is not None:
                _trainer_version_number = trainer_version_number
            elif self.args_dict.get("trainer_version_number", None) is not None:
                _trainer_version_number = self.args_dict.get("trainer_version_number", None)
            else:
                _trainer_version_number = None
            self.args_dict["trainer_version_number"] = _trainer_version_number
            self.model_config = None
            self.model_ckpt_path = None
            self.lit_model = None

    def main_suggest(self, design_generator: DesignGenerator | None = None) -> list[Path]:
        """This function generate a few encodings and test them, it returns a few of them which should be the best."""

        start_time = time()

        logger.info(f"Starting EncodingRecommender.main_suggest ...")

        # Generate new potential designs
        logger.info(f"Generating new designs ...")

        # Initialize the design generator
        design_generator: DesignGenerator
        if design_generator is None:
            _design_generator = DesignGenerator(dir_config=self.dir_config)
        else:
            _design_generator = design_generator

        # Generate N new designs
        gener_dirpath_list, new_design_config_dicts_list = _design_generator.generate_more_designs(
            nb_designs=self.nb_designs_to_generate, do_encodings_only=True, ignore_already_existing=self.dry_run
        )
        # src_dirpaths, tgt_dirpaths = list(zip(*tocopy_design_dirpath_lists))

        # Get the minimum design number
        design_numbers = map(lambda x: int(extract_int_string_from_string(Path(x).name)), gener_dirpath_list)
        min_design_number = min(design_numbers)
        zfill_len = len(extract_int_string_from_string(Path(gener_dirpath_list[0]).name))

        # Run the recommendation on the generated designs
        # No matter which run mode is set, this loop must populate 4 final lists:
        # predictions, encodings, values, paths
        final_predictions = []
        final_encoding_dicts = []
        logger.info(f"Recommender mode is {self.run_mode}.")
        nb_generated_paths = len(gener_dirpath_list)
        if self.run_mode == "score_predictor":
            # Pass encodings of newly generated designs through the model
            logger.info(
                f"Selecting a target number of {self.nb_new_designs} out of {nb_generated_paths} generated new designs based on their predicted scores."
            )
            dataset_eval = RawEncodingDataset(new_design_config_dicts_list, gener_dirpath_list)
            dataloader_eval = DataLoader(
                dataset_eval, shuffle=False, batch_size=self.batch_size, num_workers=self.nb_workers
            )

            # Predict on the newly generated designs
            self.lit_model.freeze()  # Pass the lightning model in eval mode
            min_value_twos_comp = min_value_tc(int(self.dir_config.exp_config["input_bitwidth"]))
            assert min_value_twos_comp < 0

            tgt_paths = []
            predictions = []
            encodings = []
            values = []
            for batch_idx, batch in enumerate(dataloader_eval):
                _batch = {
                    "encodings": batch["encodings"].to(self.lit_model.device),
                    "values": batch["values"].to(self.lit_model.device),
                    "scores": batch["scores"].to(self.lit_model.device),
                }
                y, y_expected = self.lit_model.predict_step(_batch)

                # Append tensors to lists
                predictions.append(y)
                encodings.append(batch["encodings"])
                values.append(
                    batch["values"] + min_value_twos_comp
                )  # Rescale values for generating the correct encodings
                # Extend lists with lists
                # src_paths.extend(batch["src_path"])
                tgt_paths.extend(batch["path"])
                # config_dicts.extend(batch["config_dict"])
                # print(batch["config_dict"])

            predictions = torch.concatenate(predictions)
            encodings = torch.concatenate(encodings)
            values = torch.concatenate(values)

            # Get the top N% of the predictions
            logger.info(f"Scores predicted, now selecting champions ...")
            top_p_idx = torch.argwhere(
                predictions.squeeze() < torch.quantile(predictions, self.keep_percentage / 100.0)
            ).squeeze()
            top_p_idx = torch.atleast_1d(top_p_idx)
            logger.info(f"Keeping {len(top_p_idx)} designs.")

            # Filter elements to keep
            final_representations = encodings[top_p_idx.cpu()]
            final_values = values[top_p_idx.cpu()]
            final_predictions = predictions[top_p_idx]
            # for _top_p_idx in list(top_p_idx):
            # selected_config_dicts = config_dicts[_top_p_idx.item()]
            # src_paths = np.atleast_1d(np.array(src_paths)[top_p_idx.cpu()])
            # print(tgt_paths)
            tgt_paths = np.atleast_1d(np.array(tgt_paths)[top_p_idx.cpu()])

            for idx, (repre, value) in enumerate(zip(final_representations, final_values)):
                # Process back representations and values into encoding dictionnaries
                _repr = utils.from_int_array_to_binstr_array(repre.numpy().astype(np.int32))
                encoding_dict = {int(v): str(r) for v, r in zip(value, _repr)}
                final_encoding_dicts.append(encoding_dict)

        elif self.run_mode == "random":
            logger.info(
                f"Randomly selecting a target number of {self.nb_new_designs} out of {nb_generated_paths} generated new designs."
            )

            # Random generation
            randon_gen = np.random.default_rng(self.seed)

            random_indices = np.arange(len(gener_dirpath_list))

            # Kept indices
            # Keeping the minimum between target number of design and number of actually generated designs
            keep_indices = randon_gen.choice(
                random_indices, size=min(self.nb_new_designs, nb_generated_paths), replace=False
            )

            # Preprare final lists
            # src_paths = np.array(gener_dirpath_list)[keep_indices].tolist()
            tgt_paths = np.array(gener_dirpath_list)[keep_indices].tolist()
            for idx in keep_indices:
                encoding_dict = new_design_config_dicts_list[idx]["in_enc_dict"]
                final_encoding_dicts.append(encoding_dict)
                final_predictions.append("nan")
            # selected_config_dicts = config_dicts[keep_indices]

        # Organize the output paths based on their design numbers
        origin_design_numbers = [file_parsers.extract_design_number_from_path(path) for path in gener_dirpath_list]
        selected_design_numbers = [
            file_parsers.extract_design_number_from_path(Path(path.item())) for path in tgt_paths
        ]
        selected_config_dicts = []
        for idx, origin_design_number in enumerate(origin_design_numbers):
            if origin_design_number in selected_design_numbers:
                selected_config_dicts.append(new_design_config_dicts_list[idx])

        # Perform the generation of the selected designs
        all_rows = []
        all_recom_paths = []

        # Print
        design_numbers = [
            str(dn).zfill(zfill_len) for dn in range(min_design_number, min_design_number + len(selected_config_dicts))
        ]
        # print(design_numbers)
        # print(selected_config_dicts)
        # print(self.dir_config.generation_out_dir)
        # exit()
        if self.dry_run:
            tmp_dir = utils.prepare_temporary_directory([])
            tmp_dir_path = Path(tmp_dir.name)
            _output_dir_path = tmp_dir_path
        else:
            _output_dir_path = self.dir_config.generation_out_dir
        gener_file_paths, used_design_config_list = _design_generator.perform_design_files_generation(
            selected_config_dicts, design_numbers, output_dir_path=_output_dir_path
        )

        logger.info(f"Generating the report")

        for idx, (prediction, encoding_dict, design_number, final_path) in enumerate(
            zip(final_predictions, final_encoding_dicts, design_numbers, gener_file_paths)
        ):
            # Format the data
            # path = Path(path)
            # src_path = Path(src_path)
            # design_number = str(min_design_number + idx).zfill(zfill_len) # Minimum design number

            # Copy the entire generated design in the design files
            # final_path = self.dir_config.generation_out_dir/f"res_{design_number}"
            # tocopy_paths.append((src_path, final_path))

            # Check some of the designs to make sure encodings and
            # if idx%1 == 100 and count < 10:
            #     input(src_path/"hdl"/"mydesign_comb.v")
            #     read_encoding = file_parsers.extract_encodings(src_path/"hdl"/"mydesign_comb.v")
            #     assert encoding_dict == read_encoding["input"]
            #     count += 1

            # Generate the report for the recommendation database
            report = {
                "trainer_version_number": self.args_dict.get("trainer_version_number"),
                "pred_score": float(prediction),
                "final_path": str(final_path),
                "encodings_input": str(encoding_dict),
                "design_number": design_number,
                "recom_iter_nb": self.current_iter_nb,
                "recom_run_mode": self.run_mode,
            }
            new_row = pd.DataFrame([report])
            all_rows.append(new_row)

            # Expend list of file paths for returning them
            all_recom_paths.append(final_path)
        logger.info(
            f"{len(all_rows)} new designs have been prepared and copied and will be added the recommendation database."
        )

        # Add to recommendation database
        if self.recom_df is None and len(all_rows) > 0:
            self.recom_df = pd.concat(all_rows)
        else:
            _all_rows = [self.recom_df]
            _all_rows.extend(all_rows)
            self.recom_df = pd.concat(_all_rows)

        if self.dry_run:
            # Plot the hypothetical results
            output_plot_dirpath = self.dir_config.recommender_out_dir_ver
            try:
                self.plot_simulated_distribution(output_dir_path=output_plot_dirpath)
            except ValueError:
                logger.error("Plotting smiulation failed. There was probably an error in trainer db.")

        else:
            #     utils.perform_parallel_copy(tocopy_paths, nb_workers=self.nb_workers)
            self.save_recom_db()

        end_time = time()

        logger.info(f"{len(all_recom_paths)} new designs have been added to {self.dir_config.generation_out_dir}.")
        logger.info(f"EncodingRecommender.main_suggest done in {(end_time - start_time) / 60:.2f}min.\n")

        return all_recom_paths

    def save_recom_db(self):
        logger.info(f"Saving recommendation database ...")
        if not any(self.recom_df["recom_iter_nb"] == 0):
            # Check whether there are existing generated designs that are not part of the db
            # This should mean that they are part of iteration 0 (initialization)
            genered_design_numbers = file_parsers.get_list_of_synth_designs_number(self.dir_config)
            genered_design_numbers = set(genered_design_numbers)
            encodings_dict = file_parsers.read_all_existing_encoding(
                root_existing_designs_path=self.dir_config.root_output_dir, curr_root_output_path=None, type="generator"
            )

            recom_df_design_numbers = set(self.recom_df["design_number"])

            # Get list of design numbers that are not in the recom_df
            diff_design_number = genered_design_numbers - recom_df_design_numbers

            if len(diff_design_number) > 0:
                logger.info(
                    f"Adding {len(diff_design_number)} initial designs, with `iter_count_nb=0`, to the recommendation db."
                )
                all_rows = [
                    self.recom_df,
                ]
                for design_number in diff_design_number:
                    # Get the input encoding dictionnary
                    encodings_input = str(encodings_dict[design_number]["input"])
                    report = {
                        "design_number": design_number,
                        "recom_iter_nb": 0,
                        "final_path": self.dir_config.generation_out_dir / f"res_{design_number}",
                        "encodings_input": encodings_input,
                        "trainer_version_number": self.args_dict.get("trainer_version_number"),
                        "recom_run_mode": "init",
                    }
                    row = pd.DataFrame([report])
                    all_rows.append(row)
                self.recom_df = pd.concat(all_rows, ignore_index=True)

        self.recom_df.to_parquet(self.recom_db_file_path, index=False)
        logger.info(f"Recommendation database saved at:")
        logger.info(self.recom_db_file_path)

    def plot_simulated_distribution(self, output_dir_path: Path):
        # Note: this will probably fail if the training did not go up to writing down the report
        _trainer_version_number = self.args_dict["trainer_version_number"]
        score_type = self.train_df[self.train_df["version_number"] == _trainer_version_number]["score_type"].item()
        score_rescale_mode = self.train_df[self.train_df["version_number"] == _trainer_version_number][
            "score_rescale_mode"
        ].item()
        seed = self.train_df[self.train_df["version_number"] == _trainer_version_number]["dataset_seed"].item()
        dataset_config = DatasetConfig(
            {
                "seed": seed,
                "score_rescale_mode": score_rescale_mode,
                "score_type": score_type,
                "batch_size": 1,
                "nb_workers": 1,
            }
        )
        dataset = SwactedDesignDataset(dataset_config=dataset_config, analyzer=self.analyzer)
        dataset_train, dataset_test = dataset.split_datasets(split_ratio=dataset_config.split_ratio)

        for test_type_name in self.analyzer.test_type_names:
            ax0: plt.axes
            ax1: plt.axes
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10), width_ratios=[15, 5])
            # Reduce swact to the current test type
            sub_swact_df = self.analyzer.swact_df[self.analyzer.swact_df["test_type"] == test_type_name]
            total_df = ScoreComputeHelper.merge_data_df(self.analyzer.synth_df, sub_swact_df)

            total_df["scores"] = ScoreComputeHelper.compute_scores(score_type, score_rescale_mode, total_df=total_df)

            # Correlate the existing data with the simulated data
            new_total_df = pd.merge(total_df, self.recom_df, how="right", on="encodings_input")
            # Remove train set samples
            train_design_numbers = dataset_train.total_df["design_number"].to_numpy()
            new_total_df_nottrain = new_total_df[~new_total_df["design_number_x"].isin(train_design_numbers)]
            new_total_df_train = new_total_df[new_total_df["design_number_x"].isin(train_design_numbers)]

            # Plot swact versus nb transistors with color meanings
            scatter_handles = ax0.scatter(
                total_df["nb_transistors"],
                total_df["swact_weighted_total"],
                c=total_df["scores"],
                label="initial",
                alpha=0.50,
            )
            ax0.scatter(
                new_total_df_train["nb_transistors"],
                new_total_df_train["swact_weighted_total"],
                c="darkred",
                label=f"next good | train | {self.iteration_nb}",
            )
            ax0.scatter(
                new_total_df_nottrain["nb_transistors"],
                new_total_df_nottrain["swact_weighted_total"],
                c="limegreen",
                label=f"next good | not train | {self.iteration_nb}",
            )

            cbar = plt.colorbar(scatter_handles)
            cbar_title = "Ground Truth Design Scores (lower is better)"
            cbar.set_label(cbar_title)

            ax0.set_xlabel("Number of Transistors")
            ax0.set_ylabel("Total Swact Weighted")

            ax0.legend()

            # Plot predicted score versus ground truth score
            ax1.scatter(new_total_df_train["scores"], new_total_df_train["pred_score"], label=f"train", c="darkred")
            ax1.scatter(
                new_total_df_nottrain["scores"], new_total_df_nottrain["pred_score"], label=f"not train", c="limegreen"
            )
            ax1.set_xlabel("Groundtruth Score")
            ax1.set_ylabel("Predicted Score")

            ax1.legend()

            plt.suptitle(
                f"Weighted Switching Activity versus Number of Transistors\n{test_type_name} | trainer_version_number {_trainer_version_number} | score_type `{score_type}` | nb_pred_total {len(new_total_df)} == keep_perc {self.keep_percentage}% \n{self.dir_config.experiment_name} - {self.dir_config.output_dir_name}"
            )
            plt.tight_layout()

            filepath = (
                output_dir_path
                / f"simulated_score_distribution_{_trainer_version_number}_{test_type_name}_iter{self.iteration_nb}.png"
            )
            plt.savefig(filepath)

            logger.info(f"Prediction simulation figure saved at:")
            logger.info(filepath)

            plt.close()

    def plot_expected_distribution(
        self,
        output_dir_path: Path,
    ):
        # TODO
        pass

    @staticmethod
    def parse_args() -> dict[str, Any]:
        analyzer_args_dict = analyzer_parser()
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument(
            "--trainer_version_number",
            type=int,
            help="Which version of the lightning model to load. It will automatically find the checkpoint to load.",
        )
        arg_parser.add_argument("--batch_size", type=int, default=256, help="Btach size to use for the prediction.")
        arg_parser.add_argument(
            "-d",
            "--device",
            type=int,
            default=4,
            help="Device number to be used for running the recommender model. Note: this script is not built for multi-gpu trainig.",
        )
        arg_parser.add_argument(
            "-n", "--nb_new_designs", type=int, required=True, help="Number of new designs that will be kept."
        )
        arg_parser.add_argument(
            "-p",
            "--keep_percentage",
            type=float,
            required=True,
            help="Top-k percentages of designs generated that will actually be kept. (Values should be in ]0;100].)",
        )
        arg_parser.add_argument(
            "--dry_run",
            action="store_true",
            help="When set, the recommender will not copy the newly generated designs in the original generation directory.",
        )

        args = arg_parser.parse_known_args()
        args_dict = vars(args[0])
        args_dict.update(analyzer_args_dict)

        return args_dict

    def switch_to_iter_mode(self, config_dict: dict[str, Any], trainer_version_number: int) -> None:
        logger.warning(f"TRAINER_VERSION_NUMBER: {trainer_version_number}")
        self.args_dict["trainer_version_number"] = trainer_version_number
        self.setup_model(trainer_version_number)
        self.setup_recom_df()

    def prepare_next_iter(self, **kwargs) -> bool:
        """
        Placeholder for prepare next iteration function.
        Returns:
            True if something went wrong.
            False otherwise.
        """
        self.setup_model(strategy=kwargs.get("strategy", None))

        return False

    @staticmethod
    def fix_recom_db_iter_counts(recom_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix problems in iteration numbers that might be due to previous method for loading iteration nb
        Because of restarts and model changes, there could have been some discontinuities in the values of iteration number
        But the order of designs added to the recommendation db can be trusted, so we can fix the iteration number nased on that
        """
        recom_iter_nb = 1
        real_recom_iter_nb = 1
        real_recom_iter_nb_serie = []
        memorized_real_recom_iter_nb = None
        memorized_recom_iter_nb = None
        entered = False
        for idx, row in recom_df.iterrows():
            if row["recom_iter_nb"] == 0:
                memorized_recom_iter_nb = recom_iter_nb
                memorized_real_recom_iter_nb = real_recom_iter_nb
                real_recom_iter_nb = 0
                real_recom_iter_nb_serie.append(real_recom_iter_nb)
                entered = True
                continue
            elif entered:
                recom_iter_nb = memorized_recom_iter_nb
                real_recom_iter_nb = memorized_real_recom_iter_nb
                entered = False

            if row["recom_iter_nb"] != recom_iter_nb:
                recom_iter_nb = row["recom_iter_nb"]
                real_recom_iter_nb += 1
            real_recom_iter_nb_serie.append(real_recom_iter_nb)

        recom_df["real_recom_iter_nb"] = real_recom_iter_nb_serie

        return recom_df


def main_cli() -> None:
    start_time = time()
    try:
        # Initialize the experiment configuration
        encoding_recommender = EncodingRecommender()
        encoding_recommender.main_suggest()

        status = "Success"
        error_msg = ""

    except Exception:
        status = "Failed"
        error_msg = traceback.format_exc()

    logger.info(error_msg)
    encoding_recommender.send_email(
        config_dict=encoding_recommender.args_dict,
        start_time=start_time,
        status=status,
        error_message=error_msg,
        calling_module="Recommender",
        root_output_dir=encoding_recommender.dir_config.root_output_dir,
    )

    logger.info("EncodingRecommender's `main_cli` exited properly.")


if __name__ == "__main__":
    main_cli()
