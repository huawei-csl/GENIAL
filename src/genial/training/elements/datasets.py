# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

from loguru import logger
from copy import deepcopy
from pathlib import Path
import json

from math import floor, ceil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import lightning as L

from genial.experiment.task_analyzer import Analyzer
from genial.utils.utils import from_binstr_list_to_int_array

from genial.config.config_dir import ConfigDir
from genial.training.elements.configs import DatasetConfig

from genial.utils.utils import enc_dict_to_tensor, enc_dict_to_values

from genial.training.elements.score_tools import (
    ScoreComputeHelper,
)

from genial.training.elements.score_tools import MetricAlignmentHelper


def half_round_up(value):
    return floor(value + 0.5)


def half_round_down(value):
    return ceil(value - 0.5)


class RawEncodingDataset(Dataset):
    def __init__(
        self,
        design_config_dict_list: list[dict[str, Any]],
        # src_design_paths_list:list[Path],
        design_paths_list: list[Path],
    ):
        self.design_config_dict_list = design_config_dict_list
        # self.src_design_paths_list = src_design_paths_list
        self.design_paths_list = design_paths_list

    def __len__(self):
        return len(self.design_config_dict_list)

    def __getitem__(self, idx):
        # Get path
        # src_design_path = self.src_design_paths_list[idx]
        design_path = self.design_paths_list[idx]

        # Get config dict
        config_dict = self.design_config_dict_list[idx]
        encodings_dict = config_dict["in_enc_dict"]

        # Convert enngs representing a dictionnary) to a tensor
        # encodings_dict = eval(encodings)
        representations = list(encodings_dict.values())
        representation_array = from_binstr_list_to_int_array(representations)
        encodings_tensor = torch.tensor(representation_array, dtype=torch.float32)

        values = torch.tensor(list(encodings_dict.keys()), dtype=torch.int32)
        values = values + torch.abs(torch.min(values))  # rescale for nn.Embedding layer

        sample = {
            "values": values,
            "encodings": encodings_tensor,
            "scores": 0,
            # "src_path": str(src_design_path),
            "path": str(design_path),
            # "config_dict": config_dict,
        }

        return sample


class SwactedDesignDatamodule(L.LightningDataModule):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        analyzer: Analyzer,
        task: list[str] | None = None,
        persitent_workers: bool = True,
    ):
        super().__init__()
        self.batch_size = dataset_config.batch_size
        self.shuffle = dataset_config.shuffle
        self.nb_workers = dataset_config.nb_workers
        logger.info(f"Datamodule will setup dataloaders num_workers == {self.nb_workers}")
        # If the task is None, we set the task to ["enc_to_score"] by default.
        if task is None:
            self.task = ["enc_to_score"]
        else:
            self.task = task

        if "synthv0_to_synthv3" in self.task:
            logger.warning(f"Dataset in use is MixedSynthDataset because task is `synthv0_to_synthv3`")
            self.swacted_dataset = MixedSynthDataset(dataset_config, analyzer)
        elif "custom_io_encodings" in self.task and "ssl" in self.task:
            self.swacted_dataset = CustomIOEncodingsSslDesignDataset()
        elif "custom_io_encodings" in self.task:
            logger.warning(f"Dataset in use is CustomIOEncodingsDataset because task is `custom_io_encodings`")
            self.swacted_dataset = CustomIOEncodingsDataset(dataset_config, analyzer)
        elif "ssl" in self.task:
            logger.warning(f"Dataset in use is SslDesignDataset because task is `ssl`")
            self.swacted_dataset = SslDesignDataset(dataset_config, analyzer)
        else:
            logger.warning(f"Dataset in use defaulted to SwactedDesignDataset")
            self.swacted_dataset = SwactedDesignDataset(dataset_config, analyzer)

        self.persitent_workers = persitent_workers

        self.dataset_train = None
        self.dataset_test = None
        self.dataset_valid = None
        self.dataloader_train = None
        self.dataloader_test = None
        self.dataloader_valid = None
        self.fixed_valid_test_sets = True

    def setup(self, stage: str):
        if self.dataset_train is None:
            self.dataset_train, self.dataset_test, self.dataset_valid = self.swacted_dataset.split_datasets(
                fixed_valid_test_sets=self.fixed_valid_test_sets
            )
            logger.info(
                f"Set sizes: Train {len(self.dataset_train)} | Test {len(self.dataset_test)} | Valid {len(self.dataset_valid)} |"
            )
        else:
            pass

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.nb_workers,
            persistent_workers=self.persitent_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.nb_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.nb_workers)

    def predict_dataloader(self):
        return self.val_dataloader()


class SwactedDesignDataset(Dataset):
    """Swact Dataset"""

    @staticmethod
    def check_nan(vals: list[float]):
        return np.any(np.isnan(vals))

    def __init__(
        self,
        dataset_config: DatasetConfig,
        analyzer: Analyzer,
        keep_encodings: bool = False,
        data_reduction: bool = True,
    ):
        logger.info("Initializing the SwactedDesignDataset dataset ...")

        # Store dir config
        self.dir_config = analyzer.dir_config

        # Store the 'augment_data' boolean
        self.augment_data = dataset_config.augment_data

        # Extract the right test type
        available_test_type_names = analyzer.test_type_names
        if dataset_config.test_type_name not in available_test_type_names:
            logger.warning(
                f"{dataset_config.test_type_name} is not a valid test type name. Avaliable test type for this design: {available_test_type_names}. Changing the required test_type to one of the existing ones."
            )
            self.test_type_name = available_test_type_names[0]
            dataset_config.test_type_name = available_test_type_names[0]
        else:
            self.test_type_name = dataset_config.test_type_name

        # Merge all databases into a single one
        self.total_df = None
        db_content_str = None
        for step in Analyzer.__existing_steps__:
            df = getattr(analyzer, f"{step}_df")
            if not df.empty:
                if "test_type" in df.columns:
                    # Filter the test type to a single one
                    df = df[df["test_type"] == dataset_config.test_type_name]

                if self.total_df is not None:
                    self.total_df = ScoreComputeHelper.merge_data_df(self.total_df, df, suffix=f"_{step}")
                else:
                    self.total_df = df

                if db_content_str is None:
                    db_content_str = step
                else:
                    db_content_str += f" + {step}"

        logger.warning(f"Database has been set with {db_content_str} data. | Contains {len(self.total_df)} designs")

        # If set, removed the special designs
        if dataset_config.exclude_special_designs:
            exclude_design_numbers = []
            for idx, name in enumerate(analyzer.special_designs_dict["legend"]):
                # if "classic_encoding" in name:
                #     continue
                # else:
                exclude_design_numbers.append(analyzer.special_designs_dict["design_numbers"][idx])

            self.total_df = self.total_df[~self.total_df["design_number"].isin(exclude_design_numbers)].reset_index(
                drop=True
            )
            logger.warning(f"Removed all special designs from all datasets.")

        if dataset_config.fast_init:
            # Sample a few elements of the dataset only
            logger.warning(f"Sampling 1000 elements from the dataset for fast initialization.")
            self.total_df = self.total_df.sample(n=1000, random_state=0).reset_index(drop=True)

        if self.augment_data:
            logger.info(f"Setting up data augmentation ...")
            self.total_df = MetricAlignmentHelper.merge_metric_according_to_invariance(
                df=self.total_df, score_type=dataset_config.score_type
            )

        # Define the score for all designs
        scores_ret = ScoreComputeHelper.compute_scores(
            score_type=dataset_config.score_type,
            score_rescale_mode=dataset_config.score_rescale_mode,
            total_df=self.total_df,
            return_args=True,
        )
        if isinstance(scores_ret, list):
            all_scaling_args = []
            for idx, (scores, scaling_args) in enumerate(scores_ret):
                self.total_df[f"scores_{idx}"] = scores
                all_scaling_args.append(scaling_args)

                # Sort the data based on the score and add ranking
                self.total_df = self.total_df.sort_values(f"scores_{idx}")
                self.total_df[f"ranking_{idx}"] = np.arange(len(self.total_df))

            self.scaling_args = scaling_args
            self.nb_scores = idx + 1
            assert self.nb_scores == dataset_config.nb_scores, (
                f"Number of scores obtained from ScoreComputeHelper ({self.nb_scores}) does not match with dataset_config.nb_scores ({dataset_config.nb_scores}), the model will probably be broken."
            )

            # Get score tensors
            # Each score_tensors element will be a 1d tensor composed of several elements
            self.total_df[f"score_tensors"] = torch.cat(
                [
                    torch.tensor(self.total_df[f"scores_{idx}"].to_numpy(), dtype=torch.float32).unsqueeze(1)
                    for idx in range(self.nb_scores)
                ],
                dim=1,
            ).split(1, dim=0)
            self.total_df[f"score_tensors"] = self.total_df["score_tensors"].map(lambda x: x.squeeze())
            # Each scores element is a list of values
            self.total_df[f"scores"] = self.total_df["score_tensors"].map(lambda x: x.tolist())
            # Each ranking will be a list of values
            self.total_df[f"ranking"] = torch.cat(
                [
                    torch.tensor(self.total_df[f"ranking_{idx}"].to_numpy(), dtype=torch.int32).unsqueeze(1)
                    for idx in range(self.nb_scores)
                ],
                dim=1,
            ).split(1, dim=0)
            self.total_df[f"ranking"] = self.total_df[f"ranking"].map(lambda x: x[0].tolist())

        else:
            self.total_df[f"scores"] = scores_ret[0]
            self.scaling_args = scores_ret[1]
            self.nb_scores = 1

            # Sort the data based on the score and add ranking
            self.total_df = self.total_df.sort_values("scores")
            self.total_df["ranking"] = np.arange(len(self.total_df))

            # Get score tensors
            self.total_df["score_tensors"] = self.total_df["scores"].map(lambda x: torch.tensor(x, dtype=torch.float32))

        # Deal with possible nan values in the score data
        size_before = len(self.total_df)
        nan_mask = ~self.total_df[
            "scores"
        ].map(
            SwactedDesignDataset.check_nan
        )  # Another option: nan_mask = torch.any(torch.isnan(torch.stack(self.total_df[f"score_tensors"].to_list())), axis=1)
        self.total_df = self.total_df[nan_mask]
        size_after = len(self.total_df)
        if size_after != size_before:
            logger.warning(
                f"!! Had to drop {size_before - size_after} rows due to nan values in the score data. Resulting dataset size is {size_after}."
            )

        # Create dictionary mapping values to encodings from the encodings_input string.
        self.total_df["encodings_dict"] = self.total_df["encodings_input"].map(lambda x: eval(x))

        # Derive encodings tensors from the dictionaries
        self.total_df["encodings_tensor"] = self.total_df["encodings_dict"].map(lambda x: enc_dict_to_tensor(x))

        # Derive values from the dictionaries
        self.total_df["values"] = self.total_df["encodings_dict"].map(lambda x: enc_dict_to_values(x))

        # Shuffle back to avoid any bias
        self.total_df = self.total_df.sample(frac=1, replace=False)

        if data_reduction:
            # Reduce data to the minimum amount of information
            key_list = [
                "encodings_tensor",
                "values",
                "scores",
                "score_tensors",
                "ranking",
                "design_number",
            ]
            if self.augment_data:
                key_list.append("encodings_input_group_id")
            if keep_encodings:
                key_list.append("encodings_input")

            if not analyzer.synth_df.empty:
                key_list += [
                    "nb_transistors",
                    "max_cell_depth",
                ]

            if not analyzer.swact_df.empty:
                if "swact_weighted_total" in analyzer.swact_df.columns:
                    swact_metric = "swact_weighted_total"
                else:
                    swact_metric = "swact_weighted_average"
                key_list += [
                    swact_metric,
                ]

            if not analyzer.cmplx_df.empty:
                key_list += [
                    "complexity_post_opt",
                ]

            old_keys = list(self.total_df.keys())
            self.total_df = self.total_df.get(key_list)

            if not analyzer.swact_df.empty:
                self.total_df.rename(columns={swact_metric: "swact_cost"}, inplace=True)

            if self.total_df is None:
                raise ValueError(
                    f"self.total_df is None. There has been an error reducing database keys to {key_list}. "
                    f"Database contained {old_keys}"
                )

        # Memorize seed for later split
        self.seed = dataset_config.seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Shuffle if wanted
        self.shuffle = dataset_config.shuffle
        # if dataset_config.shuffle:
        #     randon_gen = np.random.default_rng(self.seed)
        #     self.total_df = self.total_df.iloc[randon_gen.permutation(len(self.total_df))]

    def __len__(self):
        return len(self.total_df)

    def __getitem__(self, idx):
        encodings_raw = self.total_df["encodings_tensor"].iloc[idx]
        if self.augment_data:
            encodings = encodings_raw[:, torch.randperm(encodings_raw.shape[-1])]
            if torch.rand(1).item() >= 0.5:
                # Flip bits
                encodings = (encodings - 1).abs()
        else:
            encodings = encodings_raw
        return {
            "values": self.total_df["values"].iloc[idx],
            "encodings": encodings,
            "scores": self.total_df["score_tensors"].iloc[idx],
            "design_number": self.total_df["design_number"].iloc[idx],
            # "nb_transistors": torch.tensor(self.total_df["nb_transistors"]),
        }

    def split_datasets(
        self, fixed_valid_test_sets: bool = True, restore_split: bool = True, split_ratios=None
    ) -> tuple[Dataset]:
        """
        Splits the data in two sets and return the copies of the class instance with reduced sets
        Argmuents:
         - fixed_valid_test_sets: Use a fixed valid and test set across data subset.
         - restore_split: Old functionality (Superseded)
         - split_ratios: train/test/valid number of samples ratio (Superseded)
        """

        # Restore previous split
        reduced_df = self.total_df
        prev_split_exists = False
        if restore_split:
            fixed_split_ok = False
            if fixed_valid_test_sets:
                if self.get_fixed_dataset_split_filepath().exists():
                    logger.info(f"Restoring dataset split based on fixed valid/test sets.")
                    fixed_dataset_split_dict = json.load(open(self.get_fixed_dataset_split_filepath(), "r"))

                    prev_test_design_numbers = fixed_dataset_split_dict["test_design_numbers"]
                    prev_valid_design_numbers = fixed_dataset_split_dict["valid_design_numbers"]

                    prev_test_database = self.total_df[self.total_df["design_number"].isin(prev_test_design_numbers)]
                    prev_valid_database = self.total_df[self.total_df["design_number"].isin(prev_valid_design_numbers)]
                    prev_train_database = self.total_df[
                        ~self.total_df["design_number"].isin(prev_valid_design_numbers + prev_test_design_numbers)
                    ]

                    # To skip the reduce if statement
                    reduced_df = pd.DataFrame()

                    fixed_split_ok = True
                else:
                    logger.warning(f"Fixed valid/test does not exists. Should be in:")
                    logger.warning(self.get_fixed_dataset_split_filepath())
                    logger.warning(
                        f"Maybe check that the file name is correct. ({self.get_fixed_dataset_split_filepath().name}):"
                    )

            if not fixed_split_ok:
                logger.warning(
                    f"!Fixed valid/test set splits could not be restored from `fixed_split_filepath`."
                    "So, the validation sets are going to be different if ever you move your dataset to a different "
                    "folder without copying the dataset_split."
                )

            if not fixed_valid_test_sets or not fixed_split_ok:
                if self.get_dataset_split_filepath().exists():
                    logger.info(f"Restoring dataset split based on fixed pure random split (old functionality) ...")
                    previous_dataset_split_dict = json.load(open(self.get_dataset_split_filepath(), "r"))

                    prev_train_design_numbers = previous_dataset_split_dict["train_design_numbers"]
                    prev_test_design_numbers = previous_dataset_split_dict["test_design_numbers"]
                    prev_valid_design_numbers = previous_dataset_split_dict["valid_design_numbers"]

                    prev_train_database = self.total_df[self.total_df["design_number"].isin(prev_train_design_numbers)]
                    prev_test_database = self.total_df[self.total_df["design_number"].isin(prev_test_design_numbers)]
                    prev_valid_database = self.total_df[self.total_df["design_number"].isin(prev_valid_design_numbers)]

                    all_prev_design_numbers = []
                    all_prev_design_numbers.extend(prev_train_design_numbers)
                    all_prev_design_numbers.extend(prev_test_design_numbers)
                    all_prev_design_numbers.extend(prev_valid_design_numbers)

                    logger.info(f"Dataset splitting has been restored from previous split.")
                    reduced_df = self.total_df[~self.total_df["design_number"].isin(all_prev_design_numbers)]
                    prev_split_exists = True

        if len(reduced_df) > 0:
            # If the dataset must be shuffled, then indexes for split will be randomly ordered
            if self.shuffle:
                randon_gen = np.random.default_rng(self.seed)
                indexes = randon_gen.permutation(len(reduced_df))
            else:
                indexes = np.arange(len(reduced_df))

            if split_ratios is None:
                split_ratios = (0.94, 0.01, 0.05)

            assert sum(split_ratios) == 1.0

            # Extract split ratios
            train_ratio, test_ratio, valid_ratio = split_ratios

            # Get train indexes
            train_split_idx = min(half_round_up(len(reduced_df) * train_ratio), len(reduced_df) - 2)
            train_indexes = indexes[:train_split_idx]

            # Get test and valid indexes
            test_n_valid_indexes = indexes[train_split_idx:]
            remaining_size = len(test_n_valid_indexes)
            test_n_valid_tot_split_ratio = test_ratio + valid_ratio

            tnv_split_idx = max(half_round_down(remaining_size * (test_ratio / test_n_valid_tot_split_ratio)) - 1, 1)
            test_indexes = test_n_valid_indexes[:tnv_split_idx]
            valid_indexes = test_n_valid_indexes[tnv_split_idx:]

            # Reduce the databases
            train_database = reduced_df.iloc[train_indexes]
            test_database = reduced_df.iloc[test_indexes]
            valid_database = reduced_df.iloc[valid_indexes]

            # Combine all databases if split has been restored from previous split
            if restore_split and prev_split_exists:
                train_database = pd.concat([prev_train_database, train_database])
                test_database = pd.concat([prev_test_database, test_database])
                valid_database = pd.concat([prev_valid_database, valid_database])

            dataset_split_dict = {
                "train_design_numbers": train_database["design_number"].unique().astype(str).tolist(),
                "test_design_numbers": test_database["design_number"].unique().astype(str).tolist(),
                "valid_design_numbers": valid_database["design_number"].unique().astype(str).tolist(),
            }
            json.dump(dataset_split_dict, fp=open(self.get_dataset_split_filepath(), "w"), indent=4)
            logger.info(f"Dataset split file has been written to:")
            logger.info(f"{self.get_dataset_split_filepath()}")
        else:
            train_database = prev_train_database
            test_database = prev_test_database
            valid_database = prev_valid_database

        assert (len(train_database) + len(test_database) + len(valid_database)) == len(self.total_df), (
            f"Something went wrong while splitting the dataset."
        )

        if self.augment_data:
            # Remove train samples from the validation set.
            train_group_ids = set(train_database["encodings_input_group_id"])
            valid_database = self.remove_samples_from_database(
                group_ids=train_group_ids, database=valid_database, database_type="val"
            )
            # Remove train and validation samples from the test set.
            train_val_group_ids = train_group_ids | set(valid_database["encodings_input_group_id"])
            test_database = self.remove_samples_from_database(
                group_ids=train_val_group_ids, database=test_database, database_type="test"
            )

        dataset_train = deepcopy(self._update_database(train_database))
        dataset_test = deepcopy(self._update_database(test_database))
        dataset_valid = deepcopy(self._update_database(valid_database.sample(frac=1, replace=False)))

        logger.info(f"Dataset has been split.")
        logger.info(f"Train shape: {dataset_train.total_df.shape[0]}")
        logger.info(f"Valid shape: {dataset_valid.total_df.shape[0]}")
        logger.info(f"Test shape: {dataset_test.total_df.shape[0]}")

        return dataset_train, dataset_test, dataset_valid

    @staticmethod
    def remove_samples_from_database(group_ids: set, database: pd.DataFrame, database_type: str) -> pd.DataFrame:
        # Get length before removal
        old_len = database.shape[0]
        # Remove samples
        cond = ~database["encodings_input_group_id"].isin(group_ids)
        database = database[cond]
        # Get new length
        len_diff = old_len - database.shape[0]
        # Log the change in samples
        if len_diff > 0:
            logger.info(f"{len_diff} samples were removed from the {database_type} dataset to avoid leakage.")
        return database

    def get_dataset_split_filepath(self):
        return self.dir_config.trainer_out_root_dir / "dataset_split.json"

    def get_fixed_dataset_split_filepath(self):
        return self.dir_config.root_output_dir / "fixed_dataset_split.json"

    def _update_database(self, total_df: pd.DataFrame) -> None:
        """Update the database hold by the class instance"""

        self.total_df = total_df

        return self

    def plot_dataset_and_scores(
        self,
        output_dir_path: Path,
        scores: np.ndarray | None = None,
        iter_count: int = 0,
        title_comment: str = "",
        do_plot_distibution: bool = False,
    ):
        """TODO

        Args:
        """

        try:
            self.total_df["nb_transistors"]
        except KeyError:
            logger.info(
                "total_df does not contain the column 'nb_transistors' | Cannot plot the dataset and scores. Retuning."
            )
            return None

        # Plot dataset and scores
        nb_transistors = self.total_df["nb_transistors"]
        try:
            total_swact_count = self.total_df["swact_weighted_total"]
        except KeyError:
            total_swact_count = None
        if scores is None:
            scores = self.total_df["scores"]

        ax0: plt.axes
        if do_plot_distibution:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
        else:
            fig, ax0 = plt.subplots(1, 1, figsize=(15, 15))

        if total_swact_count is not None:
            scatter_handles = ax0.scatter(nb_transistors, total_swact_count, c=scores, label="total", alpha=0.50)
            ax0.set_xlabel("Number of Transistors")
            ax0.set_ylabel("Total Swact Weighted")

            title_str = "Weighted Switching Activity versus Number of Transistors"
            ax0.set_title(
                f"Weighted Switching Activity versus Number of Transistors\n{self.test_type_name} - n={len(self)}\n{self.dir_config.experiment_name}\n{self.dir_config.output_dir_name}"
            )

            cbar = plt.colorbar(scatter_handles)
            cbar_title = "Design Scores (lower is better)"
            cbar.set_label(cbar_title)
        else:
            scatter_handles = ax0.hist(x=nb_transistors.to_numpy(), bins=100)
            ax0.set_xlabel("Number of Transistors")
            # sns.x0.hist(nb_transistors)
            title_str = "Number of Transistors Distribution"
        title_str += f"\n{self.test_type_name} - n={len(self)}\n{self.dir_config.experiment_name}\n{self.dir_config.output_dir_name}"
        ax0.set_title(title_str)

        plt.minorticks_on()
        plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
        plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

        if do_plot_distibution:
            ax1.hist(np.atleast_1d(scores), bins=100)
            ax1.set_xlabel("Design Scores")
            ax1.set_ylabel("Occurence")
            ax1.set_title("Score Distribution")

        plt.tight_layout()
        if output_dir_path is not None:
            _iter_count = f"{iter_count}".zfill(4)
            fig_path = output_dir_path / f"it{_iter_count}_total_swact_vs_nb_transistors_{title_comment}.png"
            plt.savefig(fig_path, dpi=200)
            logger.info(f"Figure {title_comment} saved at:")
            logger.info(fig_path)

        plt.close()


class CustomIOEncodingsDataset(SwactedDesignDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, keep_encodings=True, data_reduction=False, **kwargs)

        logger.info(f"Initializing the CustomIOEncodingsDataset dataset, this can take a while ...")

        # Create dictionary mapping values to encodings from the encodings_input string.
        self.total_df["encodings_dict_input"] = self.total_df["encodings_input"].map(lambda x: eval(x))
        self.total_df["encodings_dict_output"] = self.total_df["encodings_output"].map(lambda x: eval(x))

        # Derive encodings tensors from the dictionaries
        self.total_df["encodings_tensor_input"] = self.total_df["encodings_dict_input"].map(
            lambda x: enc_dict_to_tensor(x)
        )
        self.total_df["encodings_tensor_output"] = self.total_df["encodings_dict_output"].map(
            lambda x: enc_dict_to_tensor(x)
        )

        # Extend input values
        nb_new_values = self.total_df["encodings_tensor_output"].iloc[0].shape[0]
        initial_nb_values = self.total_df["values"].iloc[0].max().item()
        new_values_tensor = torch.arange(start=0, end=nb_new_values + initial_nb_values + 1)
        self.total_df["values"] = [
            new_values_tensor,
        ] * len(self.total_df["values"])

    def __getitem__(self, idx):
        return {
            "values": self.total_df["values"].iloc[idx],
            "encodings": (
                self.total_df["encodings_tensor_input"].iloc[idx],
                self.total_df["encodings_tensor_output"].iloc[idx],
            ),
            "scores": self.total_df["score_tensors"].iloc[idx],
            "design_number": self.total_df["design_number"].iloc[idx],
            # "nb_transistors": torch.tensor(self.total_df["nb_transistors"]),
        }


class MixedSynthDataset(SwactedDesignDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, keep_encodings=True, **kwargs)

        # Get args from super init call
        dataset_config = args[0]
        analyzer = args[1]

        self.plots_dirpath = self.dir_config.trainer_out_root_dir / "synthv0_to_synthv3/dataset_plots"
        self.plots_dirpath.mkdir(parents=True, exist_ok=True)

        # Setup dir_config instance from other synth dataset
        self.other_synth_dir_config: ConfigDir
        self.other_synth_dir_config = ConfigDir.setup_other_dir_config(analyzer.args_dict)

        # Open the synth_df from other synth dataset
        other_synth_df = Analyzer._load_database(self.other_synth_dir_config.analysis_out_dir / "synth_analysis.db.pqt")

        # Computes scores of others in their own scale
        scores_other_synth, other_scaling_args = ScoreComputeHelper.compute_scores(
            score_type=dataset_config.score_type,
            score_rescale_mode=dataset_config.score_rescale_mode,
            total_df=other_synth_df,
            return_args=True,
        )
        other_synth_df["scores_other"] = scores_other_synth
        other_synth_df["score_tensors_other"] = other_synth_df["scores_other"].map(
            lambda x: torch.tensor(x, dtype=torch.float32)
        )

        # Computes scores of new/current in the scale of others
        scores_self_with_other_scale, _ = ScoreComputeHelper.compute_scores(
            score_type=dataset_config.score_type,
            score_rescale_mode=dataset_config.score_rescale_mode,
            scaling_args=other_scaling_args,
            total_df=self.total_df,
            return_args=True,
        )
        self.total_df["scores"] = scores_self_with_other_scale

        # TODO align other and self dataframes
        # Take the alignment logic from comparison script
        # Warning!! This is not correct for now
        # self.total_df["scores_other"] = scores_other_synth[:len(self.total_df)]
        # Plot all known scores
        MixedSynthDataset.plot_scores_count_distribution_comparison(
            vals_0=scores_other_synth,
            vals_1=self.total_df["scores"],
            experiment_name=self.dir_config.experiment_name,
            output_dir_name_0=self.other_synth_dir_config.output_dir_name,
            output_dir_name_1=self.dir_config.output_dir_name,
            plot_dirpath=self.plots_dirpath,
            target_metric="nb_transistors",
        )

        # Reduce the elements of other synth df
        other_synth_df = other_synth_df[
            [
                "encodings_input",
                "nb_transistors",
                "design_number",
                "scores_other",
                "score_tensors_other",
            ]
        ]

        # Merge new/current total_df and old other_synth_df based on their encodings
        total_df = pd.merge(
            self.total_df, other_synth_df, on="encodings_input", how="inner", suffixes=["_current", "_other"]
        )
        # Check merging results
        if any(~(total_df["design_number_current"] == total_df["design_number_other"])):
            msg = (
                f"Merging orignal and new dataset led to a major issue: "
                f"Design numbers are not aligned.\n"
                f"Are you sure the synthesized designs from output_dir_name {self.dir_config.output_dir_name} "
                f"are the same as the one from the `other_output_dirpath` output_dir_name {self.other_synth_dir_config.output_dir_name}?"
            )
            logger.error(msg)
            raise ValueError(msg)

        # Reduce new merged df to minimum elements and rename some columns
        key_list = [
            "encodings_tensor",
            "values",
            "scores",
            "score_tensors",
            "ranking",
            "design_number_current",
            "nb_transistors_current",
            "max_cell_depth",
            "scores_other",
            "score_tensors_other",
            "nb_transistors_other",
        ]
        total_df = total_df[key_list]
        total_df = total_df.rename(
            {
                "max_cell_depth": "max_cell_depth_current",
                "design_number_current": "design_number",
            },
            axis=1,
        )
        self.total_df = total_df
        logger.info(
            f"Mixed Synth Dataset has been successfully initialized with new target scores from {self.dir_config.output_dir_name} and input scores from {self.other_synth_dir_config.output_dir_name}."
        )
        logger.info(f"It contains (train+test+valid) a total of {len(self.total_df)} samples.")

    def __getitem__(self, idx):
        return {
            "values": self.total_df["values"].iloc[idx],
            "scores": self.total_df["score_tensors"].iloc[idx],
            "design_number": self.total_df["design_number"].iloc[idx],
            "input_scores": self.total_df["score_tensors_other"].iloc[idx],
            # Prepend score other to input tensor
            "encodings": (self.total_df["encodings_tensor"].iloc[idx], self.total_df["score_tensors_other"].iloc[idx]),
        }

    @staticmethod
    def plot_scores_count_distribution_comparison(
        vals_0: np.array,
        vals_1: np.array,
        experiment_name,
        output_dir_name_0,
        output_dir_name_1,
        plot_dirpath: Path,
        target_metric: str = "nb_transistors",
    ) -> None:
        # Plot distribution of cell counts
        fig, ax0 = plt.subplots(1, 1, figsize=(20, 10))
        n, bins, patches = ax0.hist(vals_0, bins=(np.unique(vals_0).shape[0] + 10), label=output_dir_name_0)
        n, bins, patches = ax0.hist(vals_1, bins=(np.unique(vals_1).shape[0] + 10), label=output_dir_name_1)

        ax0.legend()
        ax0.set_xlabel(type)
        ax0.set_yscale("log")
        ax0.set_ylabel("Nb occurences")
        ax0.set_title(f"Distribution of {target_metric} \nn_0={len(vals_0)} - n_1={len(vals_1)}\n{experiment_name}")

        save_filepath = plot_dirpath / f"comparison_score_distributions_for_mixed_synth_training.png"

        plt.tight_layout()
        plt.minorticks_on()
        plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
        plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

        plt.savefig(save_filepath, dpi=300)
        plt.close()
        logger.opt(colors=True).info(f"<yellow>Plot</yellow> {target_metric} distribution comparison plot saved at:")
        logger.info(save_filepath)


class SslDesignDataset(Dataset):
    def __init__(self, dataset_config: DatasetConfig, analyzer: Analyzer):
        self.encoding_bit = int(analyzer.exp_config["input_bitwidth"])
        self.values_cache = torch.tensor(list(range(2**self.encoding_bit)), dtype=torch.int32)
        self.encoding_cache = torch.tensor(
            [[int(b) for b in bin(i)[2:].zfill(self.encoding_bit)] for i in range(2**self.encoding_bit)],
            dtype=torch.float32,
        )
        self.encoding_shape_0 = self.encoding_cache.shape[0]
        self.encoding_shape_1 = self.encoding_cache.shape[1]
        pairs_one_way = torch.combinations(torch.arange(self.encoding_shape_0), r=2)
        pairs_reverse = torch.flip(pairs_one_way, [1])
        self.pairs = torch.cat([pairs_one_way, pairs_reverse], dim=0)

        self.len_dict = {
            "train": 100_000,
            "valid": 1_000,
            "test": 1_000,
        }
        self.set_type = ""  # Placeholder

    def __len__(self):
        return self.len_dict[self.set_type]

    def __getitem__(self, idx):
        if self.encoding_bit > 4:
            perm = torch.randperm(self.encoding_shape_0)  # Only 60 bit needed for output
            perm_0 = torch.floor_divide(perm, 2 ** (self.encoding_bit // 2))
            perm_1 = torch.remainder(perm, 2 ** (self.encoding_bit // 2))
            encodings_tensor = self.encoding_cache[perm]

            scores = (
                perm_0,
                perm_1,
            )
            encodings = encodings_tensor
        else:
            perm = torch.randperm(self.encoding_shape_0)
            encodings_tensor = self.encoding_cache[perm]

            scores = perm
            encodings = encodings_tensor
        # label = self.derive_ssl_label(encodings_tensor)

        return {
            "values": self.values_cache,
            # "perm": perm,
            "encodings": encodings,
            "scores": scores,
            "design_number": -1,
            # "nb_transistors": torch.tensor(self.total_df["nb_transistors"]),
        }

    def derive_ssl_label(self, encodings_tensor):
        return (encodings_tensor[self.pairs[:, 0]] + 2 * encodings_tensor[self.pairs[:, 1]]).flatten().to(torch.long)

    def split_datasets(
        self, fixed_valid_test_sets: bool = True, restore_split: bool = True, split_ratios=None
    ) -> tuple[Dataset]:
        dataset_train = deepcopy(self._update_database(set_type="train"))
        dataset_test = deepcopy(self._update_database(set_type="test"))
        dataset_valid = deepcopy(self._update_database(set_type="valid"))
        return dataset_train, dataset_test, dataset_valid

    def _update_database(self, set_type: str) -> None:
        self.set_type = set_type
        return self


class CustomIOEncodingsSslDesignDataset:
    def __init__(self, in_encoding_bit=4, out_encoding_bit=8):
        self.in_encoding_bit = in_encoding_bit
        self.out_encoding_bit = out_encoding_bit
        # Value cache
        self.in_values_cache = torch.tensor(list(range(2**self.in_encoding_bit)), dtype=torch.int32)
        self.out_values_cache = torch.tensor(list(range(60)), dtype=torch.int32)  # Only 60 bit needed for output
        # Encoding cache
        self.in_encoding_cache = torch.tensor(
            [[int(b) for b in bin(i)[2:].zfill(in_encoding_bit)] for i in range(2**in_encoding_bit)],
            dtype=torch.float32,
        )
        self.out_encoding_cache = torch.tensor(
            [[int(b) for b in bin(i)[2:].zfill(out_encoding_bit)] for i in range(2**out_encoding_bit)],
            dtype=torch.float32,
        )
        # Shape cache
        self.in_encoding_shape_0 = self.in_encoding_cache.shape[0]
        self.in_encoding_shape_1 = self.in_encoding_cache.shape[1]
        self.out_encoding_shape_0 = self.out_encoding_cache.shape[0]
        self.out_encoding_shape_1 = self.out_encoding_cache.shape[1]
        # Pairs
        in_pairs_one_way = torch.combinations(torch.arange(self.in_encoding_shape_0), r=2)
        in_pairs_reverse = torch.flip(in_pairs_one_way, [1])
        self.in_pairs = torch.cat([in_pairs_one_way, in_pairs_reverse], dim=0)
        out_pairs_one_way = torch.combinations(torch.arange(self.out_encoding_shape_0), r=2)
        out_pairs_reverse = torch.flip(out_pairs_one_way, [1])
        self.out_pairs = torch.cat([out_pairs_one_way, out_pairs_reverse], dim=0)

        self.len_dict = {
            "train": 100_000,
            "valid": 1_000,
            "test": 1_000,
        }
        self.set_type = ""  # Placeholder

    def __len__(self):
        return self.len_dict[self.set_type]

    def __getitem__(self, idx):
        in_perm = torch.randperm(self.in_encoding_shape_0)
        in_encodings_tensor = self.in_encoding_cache[in_perm]
        out_perm = torch.randperm(self.out_encoding_shape_0)[:60]  # Only 60 bit needed for output
        out_perm_0 = torch.floor_divide(out_perm, 16)
        out_perm_1 = torch.remainder(out_perm, 16)
        out_encodings_tensor = self.out_encoding_cache[out_perm]

        return {
            "values": (self.in_values_cache, self.out_values_cache),
            "encodings": (in_encodings_tensor, out_encodings_tensor),
            "scores": (in_perm, out_perm_0, out_perm_1),
            "design_number": -1,
            # "nb_transistors": torch.tensor(self.total_df["nb_transistors"]),
        }

    def split_datasets(
        self, fixed_valid_test_sets: bool = True, restore_split: bool = True, split_ratios=None
    ) -> tuple[Dataset]:
        dataset_train = deepcopy(self._update_database(set_type="train"))
        dataset_test = deepcopy(self._update_database(set_type="test"))
        dataset_valid = deepcopy(self._update_database(set_type="valid"))
        return dataset_train, dataset_test, dataset_valid

    def _update_database(self, set_type: str) -> None:
        self.set_type = set_type
        return self
