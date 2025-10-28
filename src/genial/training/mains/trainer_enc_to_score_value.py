# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

import traceback
from time import time
import datetime
from loguru import logger
import pandas as pd
from typing import Any
from pathlib import Path
import numpy as np
import argparse

from scipy.stats import linregress

import torch
from torch.utils.tensorboard.writer import SummaryWriter

import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from genial.experiment.task_analyzer import analyzer_parser, Analyzer
from genial.config.config_dir import ConfigDir
from genial.config.logging import Logging as logging

from genial.training.elements.utils import setup_analyzer
from genial.training.elements.datasets import SwactedDesignDatamodule
from genial.training.elements.score_tools import ScoreComputeHelper
from genial.training.elements.configs import DatasetConfig, ModelConfig
from genial.training.elements.models import FullTransformer
from genial.training.elements.lit.models import LitTransformer, AbstractLitModule
from genial.training.elements.modules.transformer_cores import TransformerEncoder

from genial.experiment.loop_module import LoopModule
from genial.utils.utils import load_database

from genial.utils.utils import extract_int_string_from_string

from genial.training.elements.models import AbstractModel

torch.set_float32_matmul_precision("high")


class HistogramsLoggerCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        get_histogram_of = [
            "transformer.embedding.conv1.weight",
            "transformer.embedding.positional_encodings.weight",
        ]

        tensorboard: SummaryWriter = pl_module.logger.experiment
        # Add histogram of the positional encoding
        for name, parameter in pl_module.named_parameters():
            if name in get_histogram_of:
                try:
                    tensorboard.add_histogram(f"weights/{name}", parameter.data, pl_module.global_step)
                except Exception:
                    pass


class MetricTrackerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.collection = {}

    def on_test_epoch_end(self, trainer, pl_module):
        self.collection["test_loss"] = trainer.logged_metrics["loss/test_loss"]


class EncToScoreTrainer(LoopModule):
    __valid_checkpoint_metrics__ = [
        "loss/val_loss",
        "acc/golden_acc",
    ]

    __valid_tasks__ = [
        "enc_to_score",
        "synthv0_to_synthv3",
        "custom_io_encodings",
        "ssl",
        "vae",
        "inference",
        "finetune",
        "finetune_from_ssl",  # Specific finetune case where checkpoint loading does not have trivial logic
    ]

    def __init__(
        self,
        analyzer: Analyzer | None = None,
        trainer_version_number: int | None = None,
        init_logging: bool | None = False,
    ):
        """
        Initializes the Trainer, the Datamodule, etc.
        Args:
            analyzer:
                The analyzer to use. If None, a new one will be created.
            trainer_version_number:
                The version number of the Trainer. If None, the latest version +1 will be used. If set, trainer state will be loaded from previous trainer step.
                #TODO
        """
        super().__init__()
        logger.info(f"Setting up Trainer ...")

        # Initialize the experiment configuration
        if analyzer is None:
            _args_dict = EncToScoreTrainer.parse_args()
            _analyzer = setup_analyzer(**_args_dict)
            do_setup_datasets = True
        else:
            _analyzer = analyzer
            do_setup_datasets = False
        self.analyzer = _analyzer
        self.args_dict = self.analyzer.dir_config.args_dict
        self.dir_config = self.analyzer.dir_config

        # Check the task
        self.task = self.args_dict.get("trainer_task")
        if "synthv0_to_synthv3" in self.task:
            assert self.analyzer.args_dict.get("other_output_dirpath", None) is not None, (
                "`other_output_dirpath` must be specified for mixed synth training."
            )
            logger.info(
                f"The task has been set to: {self.task} | Train a model to predict new synthesis results from the results of another synthesis run."
            )
        else:
            logger.info(f"The task has been set to: {self.task}")

        # Check Finetuning and Inference
        self.update_all_status()

        # Setup Dataset Config
        self.dataset_config = DatasetConfig(args_dict=self.args_dict)

        # Prepare lightning paths
        self.tb_logger = self.get_tb_logger()
        # trainer_version_number = self.tb_logger.version

        # Init the lightning meta model based on a transformer
        # self.setup_model(trainer_version_number=trainer_version_number)
        self.setup_model()

        # Training configuration
        self.is_fast_dev_run = self.args_dict["fast_dev_run"]
        if self.is_fast_dev_run:
            self.max_epochs = 10
            self.trainer_runs_args = {}
        elif self.is_inference:
            self.max_epochs = 1
            self.trainer_runs_args = {"ckpt_path": self.restored_checkpoint_path}
        else:
            self.max_epochs = self.args_dict.get("max_epochs", 3000)
            self.trainer_runs_args = {"ckpt_path": "best"}
        self.device_nb = self.args_dict.get("device", 5)

        # Prepare callbacks
        self.model_checkpoint_cb = None
        self.setup_checkpoint_callback()

        callbacks = [
            HistogramsLoggerCallback(),
            LearningRateMonitor(logging_interval="step"),
            # EarlyStopping(monitor="loss/mse_loss", mode="min", patience=200),
        ]
        if isinstance(self.model_checkpoint_cb, list):
            callbacks += self.model_checkpoint_cb
        else:
            callbacks.append(self.model_checkpoint_cb)

        # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
        self.trainer_args_dict = {
            "accelerator": "cuda",
            "devices": [self.device_nb],
            # "max_epochs":int(self.restored_ckpt_epoch + self.max_epochs),
            "max_epochs": self.max_epochs,
            "logger": self.tb_logger,
            "callbacks": callbacks,
            "check_val_every_n_epoch": self.args_dict.get("check_val_every_n_epoch", 5),
            "log_every_n_steps": 20,
        }
        if self.device_nb == -1:
            self.trainer_args_dict.update(
                {
                    "accelerator": "cpu",
                    "devices": 1,
                }
            )
        # if self.is_finetuning:
        #     trainer_args_dict.update({
        #         "resume_from_checkpoint":self.restored_checkpoint_path,
        #     })
        self.trainer = L.Trainer(
            **self.trainer_args_dict,
        )

        # Extract model version from trainer instances
        # TODO check this: it is not compatible with optuna sweeping, but it's very probably not needed.
        self.log_dirpath = self.dir_config.trainer_out_dir_ver

        if not self.is_fast_dev_run:
            # Initialize logging
            logging().init_logging(log_dirpath=self.log_dirpath, mode="trainer", reset=False)
            logger.info(f"Pytorch Lightning logs will be saved in:")
            logger.info(self.log_dirpath)

        # Prepare datasets
        if do_setup_datasets:
            self.setup_datasets(self.analyzer)
        else:
            self.datamodule = None

        logger.info(f"Trainer initialized.\n")

    def update_all_status(self):
        """Inference < Finetuning < Control Run"""
        self.update_finetuning_status()
        self.update_inference_status()
        status_dict = {True: "ON", False: "OFF"}
        logger.info(f"Finetuning is {status_dict[self.is_finetuning]}.")
        logger.info(f"Inference is {status_dict[self.is_inference]}.")

    def update_finetuning_status(self):
        """Finetunung < Control Run"""
        self.is_finetuning: bool = (not self.args_dict.get("is_control_run", False)) and (
            self.current_iter_nb != 0 or ("finetune" in self.task or "finetune_from_ssl" in self.task)
        )
        no_checkpoint_cond = (
            self.args_dict.get("trainer_version_number", None) is None
            and self.args_dict.get("model_checkpoint_path", None) is None
        )
        if self.is_finetuning and no_checkpoint_cond:
            raise ValueError(f"Model should be finetuned but trainer_version number to load has not been given.")

    def update_inference_status(self):
        """Inference < Finetuning"""
        self.is_inference: bool = (not self.is_finetuning) and "inference" in self.task

    def get_tb_logger(self, trainer_version_number: int | None = None) -> pl_loggers.TensorBoardLogger:
        """
        Get the TensorBoardLogger for the current trainer version number.
        """

        _trainer_version_number = self.get_trainer_version_number(trainer_version_number)

        logdir_path = self.dir_config.trainer_out_root_dir  # trainer_out
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=logdir_path, version=_trainer_version_number)
        self.dir_config.setup_trainer_version_output_dirs(trainer_version_number=tb_logger.version)

        return tb_logger

    def setup_checkpoint_callback(self):
        """Setups the model checkpoint or update its filename."""
        if self.args_dict.get("checkpoint_naming_style", "default") == "enforce_increase":
            if self.restored_checkpoint_path is None:
                # We did not restore a previous model, so it should be 0
                ckpt_iter_number = 0
            else:
                ckpt_iter_number = int(self.restored_checkpoint_path.name.split("_")[-1].split(".")[0]) + 1
        elif self.args_dict.get("checkpoint_naming_style", "default") == "default":
            ckpt_iter_number = self.current_iter_nb

        if "ssl" in self.task:
            ckpt_filename = "{epoch}_{loss/val_loss:.4f}" + f"_{ckpt_iter_number:03d}"
        else:
            ckpt_filename = (
                "{epoch}_{loss/val_loss:.4f}_{loss/r2_loss:.4f}_{acc/golden_acc:.4f}" + f"_{ckpt_iter_number:03d}"
            )

        logger.info(f"Checkpoint naming style set as: {ckpt_filename}")
        if self.model_checkpoint_cb is None:
            if self.checkpoint_metric == "loss/val_loss":
                monitor = "loss/val_loss"
            elif self.checkpoint_metric == "acc/golden_acc":
                monitor = "acc/golden_acc"

            self.model_checkpoint_cb = [
                ModelCheckpoint(
                    monitor=monitor,
                    mode="min",
                    save_top_k=3,
                    # every_n_epochs=20,
                    auto_insert_metric_name=False,
                    filename=ckpt_filename,
                ),
            ]
            if "ssl" in self.task:
                self.model_checkpoint_cb += [
                    ModelCheckpoint(
                        monitor=monitor,
                        mode="min",
                        save_top_k=10,
                        every_n_epochs=25,
                        auto_insert_metric_name=False,
                        filename=ckpt_filename,
                    ),
                ]

        else:
            # That should be enough to update the filename of the checkpoint inside the trainer
            self.model_checkpoint_cb.filename = ckpt_filename
            assert self.trainer.checkpoint_callback.filename == self.model_checkpoint_cb.filename

    def setup_model(
        self: object = None, trainer_version_number: int | None = None, strategy: str | None = None
    ) -> None:
        self.update_all_status()

        # _trainer_version_number = self.get_trainer_version_number(trainer_version_number)
        _trainer_version_number = self.get_trainer_version_number(trainer_version_number)

        yml_config_path = self.args_dict.get("yml_config_path", None)
        if yml_config_path is not None:
            if self.is_finetuning and "finetune_from_ssl" not in self.task:
                logger.warning(
                    "YAML configuration file has been provided in command line argument, but finetuning has been requested, "
                    "it will thus be ignored. The model configuration will not be loaded from the YAML file provided by the user. "
                    "Instead, it will be loaded by the ModelConfig class from the YAML file located at in the lightning logs of the "
                    "trainer_version_number provided for finetuning."
                )
            yml_config_path = Path(yml_config_path)

        # Condition: whether a model must be loaded from a checkpoint or initialized from scratch
        if _trainer_version_number is not None and (self.is_finetuning or self.is_inference):
            # Load model from checkpoint

            # Setup model configuration from a checkpoint
            model_config_kwargs = {
                "dir_config": self.dir_config,
                "trainer_version_number": _trainer_version_number,
                "task": self.task,
            }
            if "finetune_from_ssl" in self.task or "inference" in self.task or "finetune" in self.task:
                model_config_kwargs.update({"yml_config_path": yml_config_path})
            self.model_config = ModelConfig(**model_config_kwargs)
            EncToScoreTrainer.setup_checkpoint_metric(self, strategy)

            # Setup model
            model = FullTransformer(model_config=self.model_config)

            # Find checkpoint and restore lightning module
            arg_model_checkpoint_path = self.args_dict.get("model_checkpoint_path", None)
            if arg_model_checkpoint_path is None:
                checkpoint_path, ckpt_epoch = EncToScoreTrainer._find_best_checkpoint(
                    self.dir_config, trainer_version_number, strategy=self.load_ckpt_strategy
                )
                restored_steps = torch.load(checkpoint_path, weights_only=False).get("global_step")
            else:
                checkpoint_path = Path(arg_model_checkpoint_path)
                ckpt_dict = torch.load(checkpoint_path, weights_only=False, map_location=torch.device("cpu"))
                restored_steps = ckpt_dict.get("global_step")
                ckpt_epoch = ckpt_dict.get("epoch")

            if "finetune_from_ssl" in self.task:
                # Setup the lightning module
                self.lit_model = LitTransformer(
                    meta_model=model,
                    model_config=self.model_config,
                    restored_steps=restored_steps,
                    restored_epochs=ckpt_epoch,
                )
                checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
                weight_dict = {k[12:]: v for k, v in checkpoint["state_dict"].items() if "decoder." not in k}
                self.lit_model.transformer.load_state_dict(weight_dict, strict=False)
            else:
                self.lit_model = LitTransformer.load_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    meta_model=model,
                    model_config=self.model_config,
                    restored_steps=restored_steps,
                    restored_epochs=ckpt_epoch,
                )

            # NOTE: It may be possible to change learning rate here
            # Because learning rate is set during optimzer configuration not during Lightning Module instantiation
            # ? self.lit_model.config["max_scratch_lr"] = <value> ?
            self.restored_checkpoint_path = checkpoint_path
            self.restored_ckpt_epoch = ckpt_epoch

            logger.info(f"Model has been loaded from a previously existing checkpoint:")
            logger.info(f"{checkpoint_path}")
        else:
            # Initialize model from scratch

            # Setup model configuration
            self.model_config = ModelConfig(
                args_dict=self.args_dict,
                dir_config=self.dir_config,
                yml_config_path=yml_config_path,
                task=self.task,
            )
            if _trainer_version_number is not None:
                self.model_config.save_model_config(self.dir_config, _trainer_version_number)
            EncToScoreTrainer.setup_checkpoint_metric(self, strategy)

            # Setup the model
            model = FullTransformer(model_config=self.model_config)

            # Setup the lightning module
            self.lit_model = LitTransformer(meta_model=model, model_config=self.model_config)

            # Store some configuration parameters
            self.restored_checkpoint_path = None
            self.restored_ckpt_epoch = 0
            logger.info(f"Lit Model has been initialized from scratch.")

    @staticmethod
    def setup_checkpoint_metric(
        self: object | None = None, model_config: ModelConfig = None, strategy: str | None = None
    ) -> None | tuple[str, str]:
        if self is not None:
            checkpoint_metric = self.model_config.checkpoint_metric
        else:
            assert model_config is not None
            checkpoint_metric = model_config.checkpoint_metric

        assert checkpoint_metric in EncToScoreTrainer.__valid_checkpoint_metrics__

        # We expect the checkpoint metric name to be `<acc_or_loss>/<metric_name>``
        _metric_name = checkpoint_metric.split("/")[1]
        if strategy is None:
            load_ckpt_strategy = f"latest_iter_best_{_metric_name}"
        else:
            load_ckpt_strategy = strategy
        logger.info(f"Checkpoint loading strategy has been set to: {load_ckpt_strategy}")

        if self is not None:
            self.checkpoint_metric = checkpoint_metric
            self.load_ckpt_strategy = load_ckpt_strategy
        else:
            return checkpoint_metric, load_ckpt_strategy

    def setup_datasets(self, analyzer: Analyzer) -> None:
        # Setup the Datasets and Splits
        # The data is obtained from an Analyzer
        self.datamodule = SwactedDesignDatamodule(dataset_config=self.dataset_config, analyzer=analyzer, task=self.task)
        # dataset = SwactedDesignDataset(dataset_config=self.dataset_config, analyzer=analyzer)
        # self.dataset_train, self.dataset_test, self.dataset_valid = dataset.split_datasets(split_ratios=self.dataset_config.split_ratios, restore_split=True)
        # logger.info(f"Set sizes: Train {len(self.dataset_train)} | Test {len(self.dataset_test)} | Valid {len(self.dataset_valid)} |")

    def execute_full_test(self):
        """This function runs the test of the network and some characterization operations."""

        # Analyze the results
        test_dict = self.trainer.test(self.lit_model, datamodule=self.datamodule, **self.trainer_runs_args)
        test_loss = test_dict[0]["loss/test_loss"]
        logger.info(f"Testing successfully ended with test_lost={test_loss:.4f}.")

        # Plot predicted distribution
        predict_output = self.trainer.predict(
            model=self.lit_model, dataloaders=self.datamodule.test_dataloader(), **self.trainer_runs_args
        )
        prediction_scores, groundtruth_scores = list(zip(*predict_output))
        prediction_scores = torch.concatenate(prediction_scores).squeeze()
        groundtruth_scores = torch.concatenate(groundtruth_scores)
        self.datamodule.dataset_test.plot_dataset_and_scores(
            output_dir_path=self.log_dirpath,
            scores=prediction_scores,
            iter_count=self.current_iter_nb,
            title_comment="fast_dev_run_after_training",
            do_plot_distibution=True,
        )
        # Fit a linear regression model
        slope, _, r_value, p_value, _ = linregress(
            torch.atleast_1d(prediction_scores).cpu(), torch.atleast_1d(groundtruth_scores).cpu()
        )

        test_res_dict = {
            "best_test_loss": float(test_loss),
            "linreg_slope": slope,
            "linreg_rval": r_value,
            "linreg_pval": p_value,
        }
        return test_res_dict

    def get_tain_db_path(self):
        """Helper function to access the path of the training database."""
        return self.dir_config.root_output_dir / "training.db.pqt"

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

    def generate_n_save_report(self, test_res_dict: dict | None = None, iter_checkpoint_only: bool = False):
        report_filepath = self.get_tain_db_path()

        # Reporting
        if not self.is_fast_dev_run or self.is_inference:
            logger.info(f"Reporting ...")
            try:
                if iter_checkpoint_only:
                    report_dict = {
                        "date": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        "trainer_version_number": self.trainer.logger.version,
                        "train_iter_nb": self.current_iter_nb,
                    }

                else:
                    report_dict = {
                        "date": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        "trainer_version_number": self.trainer.logger.version,
                        "train_iter_nb": self.current_iter_nb,
                        "batch_size": self.datamodule.batch_size,
                        # "learning_rate":self.trainer.lr_scheduler_configs[0].scheduler._schedulers[1].max_lrs[0],
                        # "learning_rate":self.model_config.max_scratch_lr,
                        "global_step": self.lit_model.global_step,
                        "best_checkpoint_metric": float(self.model_checkpoint_cb.best_model_score),
                        "ckpt_strategy": self.load_ckpt_strategy,
                        # "best_mse_loss":float(self.trainer.logged_metrics["loss/mse_loss"]),
                        # "best_rvalue_loss":float(self.trainer.logged_metrics["loss/rvalue_loss"]),
                        # "best_golden_acc":float(self.trainer.logged_metrics["acc/golden_acc"]),
                        "loss_type": "mse_loss",
                        "score_type": self.dataset_config.score_type_name,
                        "score_rescale_mode": self.dataset_config.score_rescale_mode,
                        "model_type": "FullTransformerV0",
                        "experiment_name": self.dir_config.experiment_name,
                        "outptut_dir_name": self.dir_config.root_output_dir.name,
                        "best_model_checkpoint": self.model_checkpoint_cb.best_model_path,
                        "dataset_seed": self.dataset_config.seed,
                    }
                    if test_res_dict is not None:
                        report_dict.update(test_res_dict)

                # Build dataframe for save operation
                new_df = pd.DataFrame([report_dict])
                if report_filepath.exists():
                    df = load_database(report_filepath)
                    df = pd.concat([df, new_df], ignore_index=True)
                else:
                    df = new_df

                # Save the database
                df.to_parquet(report_filepath, index=False)

                # Log some stuff
                logger.info(report_dict)
                logger.info(f"Report Databased saved at {report_filepath}")

            except Exception:
                logger.error(f"Something went wrong while reporting for version number {self.trainer.logger.version}.")
                logger.error(traceback.format_exc())
                report_dict = None

            return report_dict

        else:
            return None

    @staticmethod
    def _find_best_checkpoint(
        dir_config: ConfigDir, trainer_version_number: int | None = None, strategy: str | None = None
    ) -> Path:
        """
        Find which lightning checkpoint to load.
        Args:
         - dir_config of the experiment to load and save data from
         - trainer_version_number: model version to load checkpoint from. This defines the lightning checkpoint folder into which to search for a checkpoint.
         - strategy: which strategy to use for loading a checkpoint. Currently, `alltime_best_<metric_name>` and `latest_iter_best_golden_acc` are supported.
        """

        if strategy is None:
            _strategy = "latest_iter_best_val_loss"
        else:
            _strategy = strategy

        if dir_config.trainer_out_dir_ver is None:
            dir_config.setup_trainer_version_output_dirs(trainer_version_number)

        pl_logs_dirpath = dir_config.trainer_out_dir_ver
        ckpts_path = pl_logs_dirpath / "checkpoints"

        # Get all existing checkpoint files in the provided lightning logs directory
        epochs = []
        val_losses = []
        rvalue_losses = []
        golden_acces = []
        versions = []
        iter_counts = []
        paths = []

        if not ckpts_path.exists():
            return None, None

        for ckpt_path in ckpts_path.iterdir():
            if ckpt_path.is_file():
                # TODO fix this ...
                ckpt_name_split = ckpt_path.name.split("_")
                if len(ckpt_name_split) == 5:
                    epoch, val_loss, rvalue_loss, golden_acc, iter_count = ckpt_name_split
                    version = 0
                elif len(ckpt_name_split) == 4:
                    epoch, val_loss, rvalue_loss, iter_count = ckpt_name_split
                    golden_acc = 0
                    version = 0
                elif len(ckpt_name_split) == 3:
                    epoch, val_loss, iter_count = ckpt_name_split
                    rvalue_loss = 0
                    golden_acc = 0
                    version = 0
                else:
                    logger.error(f"Found checkpoint split {ckpt_name_split}")
                    logger.error(
                        "Checkpoint Path name is too old. Please change naming to something following this pattern: '{epoch}_{loss/val_loss:.4f}_{loss/r2_loss:.4f}_{acc/golden_acc:.4f}_{current_iter_nb:02d}"
                    )
                    raise ValueError()

                # if int(extract_int_string_from_string(str(epoch))) < 100:
                #     continue

                epochs.append(int(extract_int_string_from_string(str(epoch))))
                val_losses.append(float(val_loss))
                rvalue_losses.append(float(rvalue_loss))
                golden_acces.append(float(golden_acc))
                versions.append(int(extract_int_string_from_string(str(version))))
                iter_counts.append(int(extract_int_string_from_string(str(iter_count))))
                paths.append(str(ckpt_path))

        epochs = np.array(epochs)
        val_losses = np.array(val_losses)
        rvalue_losses = np.array(rvalue_losses)
        golden_acces = np.array(golden_acces)
        versions = np.array(versions)
        iter_counts = np.array(iter_counts)
        paths = np.array(paths)

        if _strategy == "alltime_best_val_loss":
            best_loss_idxs = np.atleast_1d(np.argmin(val_losses))
        elif _strategy == "alltime_best_rvalue_loss":
            best_loss_idxs = np.atleast_1d(np.argmin(rvalue_losses))
        elif _strategy == "alltime_best_golden_acc":
            best_loss_idxs = np.atleast_1d(np.argmin(golden_acces))
        elif _strategy == "latest_iter_best_golden_acc":
            latest_iter_counts = np.where(iter_counts == np.max(iter_counts))[0]
            reduced_best_losses_idxs = np.argmin(golden_acces[latest_iter_counts])
            best_loss_idxs = np.atleast_1d(latest_iter_counts[reduced_best_losses_idxs])
        elif _strategy == "latest_iter_best_val_loss":
            latest_iter_counts = np.where(iter_counts == np.max(iter_counts))[0]
            reduced_best_losses_idxs = np.argmin(val_losses[latest_iter_counts])
            best_loss_idxs = np.atleast_1d(latest_iter_counts[reduced_best_losses_idxs])
        elif _strategy == "best_mse_no_ssl":
            non_zero_indices = np.nonzero(golden_acces != 0)[0]
            best_loss_idxs = np.atleast_1d(non_zero_indices[np.argmin(golden_acces[non_zero_indices])])
        else:
            raise NotImplementedError(f"_strategy {_strategy} not implemented")

        # Get only iter counts and versions that correspond to minimum loss
        iter_counts = iter_counts[best_loss_idxs]
        versions = versions[best_loss_idxs]
        epochs = epochs[best_loss_idxs]
        paths = paths[best_loss_idxs]
        val_losses = val_losses[best_loss_idxs]
        rvalue_losses = rvalue_losses[best_loss_idxs]
        golden_acces = golden_acces[best_loss_idxs]

        # Get the best path from latest epoch
        max_iter_counts_val = np.max(iter_counts)
        max_version_val = np.max(versions)

        max_version_idx = np.atleast_1d(
            np.where(np.logical_and(versions == max_version_val, iter_counts == max_iter_counts_val))[0]
        )

        latest_version_val_losses = val_losses[max_version_idx]
        latest_version_rvalue_losses = rvalue_losses[max_version_idx]
        latest_version_golden_acces = golden_acces[max_version_idx]
        latest_versions_paths = paths[max_version_idx]
        latest_versions_epochs = epochs[max_version_idx]

        if _strategy == "alltime_best_val_loss":
            filtered_idx = np.argmin(latest_version_val_losses)
        elif _strategy == "alltime_best_rvalue_loss":
            filtered_idx = np.argmin(latest_version_rvalue_losses)
        elif _strategy == "alltime_best_golden_acc":
            filtered_idx = np.argmin(latest_version_golden_acces)
        elif _strategy == "latest_iter_best_golden_acc":
            filtered_idx = np.argmin(latest_version_golden_acces)
        elif _strategy == "latest_iter_best_val_loss":
            filtered_idx = np.argmin(latest_version_golden_acces)
        elif _strategy == "best_mse_no_ssl":
            filtered_idx = np.argmin(latest_version_golden_acces)

        logger.info(f"Best Checkpoint obtained based on load strategy:{_strategy}")

        best_ckpt_path = Path(latest_versions_paths[filtered_idx])
        ckpt_epoch = latest_versions_epochs[filtered_idx]
        logger.info(best_ckpt_path)

        return best_ckpt_path, ckpt_epoch

    def main(self):
        """
        Runs the training.
        """
        # Start Tick
        start_time = time()
        logger.info(f"Starting Trainer.main with batch size {self.lit_model.batch_size}....")

        if not self.is_fast_dev_run:
            # Do a first validation check before training
            self.trainer.validate(self.lit_model, datamodule=self.datamodule)
            # Try to plot dataset distribution
            try:
                # Plot dataset and scores
                if self.datamodule.dataset_test is None:
                    self.datamodule.setup(stage="test")
                self.datamodule.dataset_test.plot_dataset_and_scores(
                    output_dir_path=self.log_dirpath,
                    iter_count=self.current_iter_nb,
                    title_comment="before_training",
                    do_plot_distibution=False,
                )
            except AttributeError:
                logger.info(f"self.datamodule.dataset_test has not plot_dataset_and_scores, skipping.")
            except ValueError:
                logger.info(f"self.datamodule.plot_dataset_and_scores does not work for multi targets.")

            except Exception as e:
                error_msg = traceback.format_exc()
                logger.error(error_msg)
                raise e

        # Train the model
        # model = torch.compile(self.lit_model)
        model = self.lit_model
        fit_args_dict = {
            "model": model,
            "datamodule": self.datamodule,
        }

        if self.is_inference:
            logger.info("Training will be skipped because we are running inference only.")
        else:
            logger.info(f"Training will be done on device(s):{self.trainer.device_ids}")
            self.trainer.fit(**fit_args_dict)
            logger.info("Training successfully ended.")
            self.lit_model = model

        # if not "ssl" in self.task:
        #     # Test the model
        #     test_res_dict = self.execute_full_test()
        #
        #     # Make and save report
        #     self.generate_n_save_report(test_res_dict=test_res_dict)

        end_time = time()

        # DONE
        time_diff = end_time - start_time
        logger.info(f"Trainer.main done in {time_diff / 60 / 60:.2f}h or {int(time_diff)}s. \n")

        return self.trainer.logger.version

    @staticmethod
    def parse_args():
        analyzer_args_dict = analyzer_parser()

        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument(
            "-d",
            "--device",
            type=int,
            default=0,
            help="Device number to be used for running this training script. Note: this script is not built for multi-gpu trainig.",
        )
        arg_parser.add_argument("--seed", type=int, default=626, help="Random seed used for dataset splitting.")
        arg_parser.add_argument("--batch_size", type=int, default=128, help="Size of a bacth to use for training.")
        arg_parser.add_argument(
            "--fast_dev_run",
            action="store_true",
            help="Whether to do a single train, val and test epoch to check if everythin is working. Reporting and plotting will not occur.",
        )
        arg_parser.add_argument("--max_epochs", type=int, default=3000, help="Max number of epochs to train for.")
        arg_parser.add_argument(
            "--check_val_every_n_epoch",
            type=int,
            default=2,
            help="Number of train epoch after whih a validation run will be done (andcheckpoint will be saved).",
        )
        arg_parser.add_argument(
            "--trainer_version_number",
            type=int,
            default=None,
            help="If set, the training will restart from the best checkpoint obtained for the specified version.",
        )  # TODO
        # arg_parser.add_argument("--finetune", action="store_true", help="Whether to do a single train, val and test epoch to check if everything is working. Reporting and plotting will not occur.")
        arg_parser.add_argument(
            "--yml_config_path", type=str, default=None, help="Specify which configuration to load."
        )
        arg_parser.add_argument(
            "--model_checkpoint_path", type=str, default=None, help="Specify the model checkpoint to load."
        )
        arg_parser.add_argument(
            "--checkpoint_naming_style",
            type=str,
            default="default",
            help="Specify which checkpoint naming style to adopt avaliable: `default` or `enforce_increase`.",
        )
        arg_parser.add_argument(
            "--test_type_name",
            type=str,
            default="fullsweep",
            help="Specify which test to use for building the database.",
        )
        # analyzer_arg_parser.add_argument("--learning_rate", type=float, default=None, help="Specify which max learning rate to use when starting a training.")

        arg_parser.add_argument(
            "--score_type",
            type=str,
            default="trans",
            choices=list(ScoreComputeHelper.transformer_map.keys()),
            help="Which design score to use for this experiment.",
        )
        arg_parser.add_argument(
            "--score_rescale_mode",
            type=str,
            default="standardize",
            choices=list(ScoreComputeHelper.scaler_map.keys()),
            help="Which rescaling method to use for this experiment.",
        )
        arg_parser.add_argument(
            "--nn_embedding_type",
            type=str,
            default="default",
            choices=list(AbstractModel.embedding_type_mapping.keys()),
            help="Which embedding model to take for this experiment. Is used by ModelConfig.",
        )
        arg_parser.add_argument(
            "--nn_core_type",
            type=str,
            default="default",
            choices=list(AbstractModel.transformer_type_mapping.keys()),
            help="Which core (transformer) model to take for this experiment. Is used by ModelConfig.",
        )
        arg_parser.add_argument(
            "--nn_decoder_type",
            type=str,
            default="default",
            choices=list(AbstractModel.decoder_type_mapping.keys()),
            help="Which decoder model to take for this experiment. Is used by ModelConfig.",
        )

        arg_parser.add_argument(
            "--criterion_type",
            type=str,
            default="mse_loss",
            choices=list(AbstractLitModule.__available_criterion_types__),
            help="Which criterion use for training. Is used by ModelConfig. Is overwriten by the same field in yaml config files.",
        )
        arg_parser.add_argument(
            "--lr_scheduler_type",
            type=str,
            default="cyclic_lr",
            choices=list(AbstractLitModule.__available_scheduler_types__),
            help="Which learning rate scheduler to use. Is used by ModelConfig. Is overwriten by the same field in yaml config files.",
        )
        arg_parser.add_argument(
            "--optimizer_type",
            type=str,
            default="lamb",
            choices=list(AbstractLitModule.__available_optimizer_types__),
            help="Which decoder model to take for this experiment. Is used by ModelConfig. Is overwriten by the same field in yaml config files.",
        )

        arg_parser.add_argument(
            "--trainer_task",
            nargs="*",
            type=str,
            default=["enc_to_score"],
            choices=list(EncToScoreTrainer.__valid_tasks__),
            help="Which task will be performed by the trainer.",
        )
        arg_parser.add_argument(
            "--mask_type",
            choices=list(TransformerEncoder.__valid_mask_types__),
            help="Which task will be performed by the trainer.",
        )

        arg_parser.add_argument(
            "--no_data_augmentation",
            action="store_true",
            help="This skips the data augmentation step.",
        )

        args = arg_parser.parse_known_args()
        args_dict = vars(args[0])
        args_dict.update(analyzer_args_dict)

        return args_dict

    def prepare_next_iter(self, **kwargs) -> bool:
        """
        Placeholder for prepare next iteration function.
        Returns:
            True if something went wrong.
            False otherwise.
        """
        self.setup_checkpoint_callback()
        self.setup_model(strategy=kwargs.get("strategy", None))
        self.setup_datasets(kwargs.get("analyzer"))

        # Reset the trainer
        # logger.info(f"Trainer max epoch was {self.trainer.fit_loop.max_epochs}")
        self.trainer_args_dict.update({"max_epochs": int(kwargs.get("max_epochs_p_iter", 0))})
        self.trainer = L.Trainer(
            **self.trainer_args_dict,
        )
        logger.info(
            f"Trainer prepared for next iter. Max epoch for this iteration has been set to {self.trainer.fit_loop.max_epochs}"
        )
        # if self.trainer.fit_loop.max_epochs != int(kwargs.get("max_epochs_p_iter",0)):
        #     self.trainer.fit_loop.max_epochs = int(kwargs.get("max_epochs_p_iter",0))
        # else:
        #     self.trainer.fit_loop.max_epochs += int(kwargs.get("max_epochs_p_iter",0))

        return False

    def switch_to_iter_mode(self, config_dict: dict[str, Any], **kwargs) -> None:
        """
        Placeholder for switching to iteration mode function.
        """
        # TODO enable switching to loading mode instead of starting from scratch the training
        pass


def main_cli():
    start_time = time()
    try:
        # Initizalize the trainer
        enc_to_score_trainer = EncToScoreTrainer()

        # Run the trainer
        enc_to_score_trainer.main()

        status = "Success"
        error_msg = ""
    except Exception:
        status = "Failed"
        error_msg = traceback.format_exc()

    logger.info(error_msg)
    enc_to_score_trainer.send_email(
        config_dict=enc_to_score_trainer.args_dict,
        start_time=start_time,
        status=status,
        error_message=error_msg,
        calling_module="Trainer",
        root_output_dir=enc_to_score_trainer.dir_config.root_output_dir,
    )

    logger.info("EncToScoreTrainer's `main_cli` exited properly.")


if __name__ == "__main__":
    # main_cli()

    # Initialize the trainer
    enc_to_score_trainer = EncToScoreTrainer()

    # Run the trainer
    enc_to_score_trainer.main()
