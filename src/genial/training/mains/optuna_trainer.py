# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

import time
from loguru import logger
from typing import Any

import math

import lightning as L

from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.tuner import Tuner

from genial.training.elements.models import FullTransformer
from genial.training.elements.lit.models import LitTransformer

from genial.training.mains.trainer_enc_to_score_value import EncToScoreTrainer

import optuna
from lightning.pytorch.callbacks import EarlyStopping

from genial.config.logging import Logging as logging

import argparse
from pathlib import Path
import os


class OptunaTrainer(EncToScoreTrainer):
    """Wrapper class around EncToScoreTrainer to optimize hyperparamters of the network."""

    __nas_config__ = {
        "percent_valid_dataloader": 1.0,
        "max_epochs": 300,
        "check_val_every_n_epoch": 1,
        "nb_optuna_trials": 250,
        "pruning_metric": "acc/golden_acc",
        "early_stop_patience_epochs": 100,
        "log_tensorboard": False,
    }

    def __init__(self, **kwargs):
        """
        Initializes the EncToScoreTrainer which can initialize itself.
        """
        super().__init__()
        logging().init_logging(self.dir_config.root_output_dir, mode="optuna_sweep")
        self.study_id = kwargs.get("study_id", "default")
        self.tb_log_dirpath: Path = self.dir_config.trainer_out_root_dir / "optuna" / self.study_id
        if not self.tb_log_dirpath.exists():
            self.tb_log_dirpath.mkdir(parents=True, exist_ok=True)

        # Remove useless elements from the EncToScoreTrainer
        # self.tb_logger = None

        logger.info(f"Optuna Trainer Successfully Setup.")

    def setup_optuna_iter(self, param_tuner_config_dict: dict[str, Any], trial: optuna.trial.Trial):
        model_config = self.model_config.update_config(param_tuner_config_dict, return_cfg_object=True, pow_two=True)

        model = FullTransformer(model_config=model_config)
        lit_model = LitTransformer(meta_model=model, model_config=model_config, do_validation_plots=True)

        # tb_log_path =  self.tb_log_dirpath / f"trial_{trial.number:5d}"
        if self.__nas_config__["log_tensorboard"]:
            train_logger = pl_loggers.TensorBoardLogger(save_dir=self.tb_log_dirpath, version=trial.number)
        else:
            logger.warning(f"Tensorboard logging is deactivated.")
            train_logger = True

        trainer_args_dict = {
            "logger": train_logger,
            "limit_val_batches": self.__nas_config__["percent_valid_dataloader"],
            "enable_checkpointing": False,
            "max_epochs": self.__nas_config__["max_epochs"],
            "check_val_every_n_epoch": self.__nas_config__["check_val_every_n_epoch"],
            "accelerator": "cuda",
            "devices": [
                self.device_nb,
            ],
            "callbacks": [
                EarlyStopping(
                    monitor="loss/val_loss",
                    mode="min",
                    patience=int(
                        self.__nas_config__["early_stop_patience_epochs"]
                        / self.__nas_config__["check_val_every_n_epoch"]
                    ),
                ),
                # PyTorchLightningPruningCallback(trial, monitor=self.__nas_config__["pruning_metric"]),
            ],
            "enable_progress_bar": True,
            "log_every_n_steps": 10,
            # "tb_logger":tb_logger,
        }
        trainer = L.Trainer(
            **trainer_args_dict,
        )

        tuner = Tuner(trainer)
        logger.info(f"Finding the best batch size ...")
        tuner.scale_batch_size(lit_model, datamodule=self.datamodule, init_val=128, max_trials=4, steps_per_trial=2)
        logger.info(f"Batch Size after tuning: {lit_model.batch_size}")

        return trainer, lit_model


def objective(optuna_trainer: OptunaTrainer, trial: optuna.trial.Trial) -> float:
    """ """
    start_time = time.time()

    param_tuner_config_dict = {
        "d_model": trial.suggest_int("d_model", int(math.log2(64)), int(math.log2(1024))),
        "nhead": trial.suggest_int("nhead", int(math.log2(1)), int(math.log2(16))),
        "dim_feedforward": trial.suggest_int("dim_feedforward", int(math.log2(64)), int(math.log2(2048))),
        "num_layers": trial.suggest_int("num_layers", 2, 10),
        "dropout": trial.suggest_float("dropout", 0.1, 0.11, step=0.05),
        "max_scratch_lr": trial.suggest_float("learning_rate", 2.5e-5, 2.5e-4, step=2.5e-5),
        "mask_type": trial.suggest_categorical("mask_type", choices=["none", "skewed_subsequent"]),
        # "mask_type":trial.suggest_categorical("mask_type", choices=["skewed_subsequent","skewed_subsequent"])
    }

    param_tuner_config_dict.update(
        {
            "score_rescale_mode": optuna_trainer.args_dict.get("score_rescale_mode"),
            "score_type": optuna_trainer.args_dict.get("score_type"),
        }
    )

    trainer: L.Trainer
    lit_model: L.LightningModule

    trainer, lit_model = optuna_trainer.setup_optuna_iter(param_tuner_config_dict, trial)
    # TODO create a datamodule instead to enable batch_size tuning
    # Also, add the batch_size to the model configuration

    param_tuner_config_dict.update({"batch_size": lit_model.batch_size})
    trainer.logger.log_hyperparams(param_tuner_config_dict)

    logger.info(f"Starting new training.")
    logger.info(f"Trial parameters:{param_tuner_config_dict}")
    logger.info(f"Dataloaders used are the ones of object {optuna_trainer} ..")
    trainer.fit(lit_model, datamodule=optuna_trainer.datamodule)

    end_time = time.time()

    val_loss = trainer.callback_metrics["loss/val_loss"].item()
    rvalue_loss = trainer.callback_metrics["loss/rvalue_loss"].item()
    slope_acc = trainer.callback_metrics["acc/slope_acc"].item()
    golden_acc = trainer.callback_metrics["acc/golden_acc"].item()
    logger.info(
        f"Current trial resulted in: validation_loss={val_loss} | rvalue_loss={rvalue_loss} | slope_acc={slope_acc}"
    )
    logger.info(f"Trial parameters:{trial.params}")
    logger.info(f"Trial time: {(end_time - start_time) / 60.0:.2f}min")

    # if optuna_trainer.__nas_config__["target_metric"] == "loss/val_loss":
    #     return val_loss
    # elif optuna_trainer.__nas_config__["target_metric"] == "loss/rvalue_loss":
    #     return rvalue_loss

    # return val_loss, rvalue_loss, slope_acc
    return golden_acc


def log_best_trial(study: optuna.study.Study) -> None:
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: {}".format(trial.value))

    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))


def main_cli():
    arg_parser = argparse.ArgumentParser(add_help=True)
    arg_parser.add_argument(
        "--study_id",
        type=str,
        default="default",
        help="Study identifier which will be suffixed in the study name for loading the correct study from the studay database.",
    )

    args = arg_parser.parse_known_args()[0]
    start_time = time.time()

    # Initizalize the trainer
    optuna_trainer = OptunaTrainer(**vars(args))

    # Setup the study
    study_name = f"optuna_{optuna_trainer.dir_config.experiment_name}_{optuna_trainer.dir_config.output_dir_name}_{vars(args).get('study_id')}"
    db_uri = f"mysql://{os.environ.get('USER')}@localhost/optuna_studies?unix_socket=/var/run/mysqld/mysqld.sock"
    logger.info(f"Study name is: {study_name}")
    study_kwargs = dict(
        study_name=study_name,
        # directions=["minimize", "minimize", "maximize"],
        direction="minimize",
        # pruner=optuna.pruners.HyperbandPruner(),
        sampler=optuna.samplers.TPESampler(),
        storage=db_uri,
        load_if_exists=True,
    )
    study = optuna.create_study(**study_kwargs)

    logger.info(f"Optuna Sampler is {study.sampler.__class__.__name__}")

    def _objective(trial):
        return objective(optuna_trainer, trial)

    # Launch optimization
    logger.info(f"Starting optimization ...")
    study.optimize(_objective, n_trials=optuna_trainer.__nas_config__["nb_optuna_trials"])

    end_time = time.time()

    log_best_trial(study)
    logger.info(f"Full optuna search study done .")
    logger.info(f"It took: {(end_time - start_time) / 60 / 60:.2f}h.")
    logger.info(f"Exited properly.")


if __name__ == "__main__":
    main_cli()
