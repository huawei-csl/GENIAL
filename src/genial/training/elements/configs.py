# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

from loguru import logger
import yaml
from pathlib import Path
import math

from typing import Any
from genial.config.config_dir import ConfigDir

__default_score_type__ = "trans"
__default_score_rescale_mode__ = "standardize"


def get_score_numbers(score_type: str):
    if "composed" in score_type:
        if score_type == "composed_all_power":
            nb_scores = 3
        elif score_type == "composed_power_area":
            nb_scores = 2
        elif score_type == "composed_power":
            nb_scores = 1
        else:
            raise NotImplementedError(f"Composed score_type {score_type} not implemented")
    else:
        nb_scores = 1

    return nb_scores


class ModelConfig:
    # All parameters specified here will be saved into the YAML config file
    # (And thus restored from config file when a yaml config filepath is provided)
    __model_params__ = [
        "d_model",
        "nhead",
        "dim_feedforward",
        "num_layers",
        "dropout",
        "activation",
        "encoding_width",
        "encoding_width_output",
        "max_sequence_length",
        "final_layer_type",
        "batch_size",
        "max_scratch_lr",
        "max_restart_lr",
        "checkpoint_metric",
        "mask_type",
        "embedding_type",
        "core_type",
        "decoder_type",
        "score_rescale_mode",
        "score_type",
        "nb_scores",
        "n_cls_token",
        "lr_scheduler_type",
        "optimizer_type",
        "criterion_type",
        "task",
        "vae_constraint",
        "num_decoder_layers",
        "freeze_embedding",
        "do_slope_regularization",
    ]

    # Parameters that should be set from a power of two
    __pow_two_params__ = [
        "d_model",
        "nhead",
        "dim_feedforward",
    ]

    def __init__(
        self,
        dir_config: ConfigDir,
        args_dict: dict | None = None,
        trainer_version_number: int | None = None,
        yml_config_path: Path | None = None,
        device_nb: int | None = None,
        task: list[str] = ["enc_to_score"],
    ):
        # Remember device
        self.device_nb = device_nb

        if args_dict is None:
            args_dict = dir_config.args_dict

        # Training configuration
        self.task = task  # Task must be set by the trainer, not the user.
        self.batch_size = args_dict.get("batch_size", 512)
        self.max_scratch_lr = args_dict.get("max_scratch_lr", 1.0e-4)
        self.max_restart_lr = args_dict.get("max_restart_lr", 1.0e-4)
        self.checkpoint_metric = args_dict.get("checkpoint_metric", "loss/val_loss")

        # Encoder layer configuration
        self.d_model = 512
        self.nhead = 8
        self.dim_feedforward = 512
        self.dropout = 0.1
        self.activation = "gelu"
        self.mask_type = args_dict.get("mask_type", "none")

        # Encoder transformer configuration
        self.num_layers = 5
        self.num_decoder_layers = 2

        # Essential analyzer configuration
        self.encoding_width = int(dir_config.exp_config["input_bitwidth"])
        self.encoding_width_output = int(dir_config.exp_config["output_bitwidth"])

        # Setup Modules
        # ==== Embedding ====
        if "custom_io_encodings" in self.task:
            self.embedding_type = "pointnetv2"
            self.max_sequence_length = 2**self.encoding_width + 2**self.encoding_width_output
            logger.warning(
                f"Because `task` has been set to {self.task}, the model's `embedding_type` has been set to {self.embedding_type}"
            )
        else:
            self.embedding_type = args_dict.get("nn_embedding_type", "default")
            self.max_sequence_length = 2**self.encoding_width
        logger.info(
            f"Model parameter `max_seq_length` will be {self.max_sequence_length}, according to bitwidth configured."
        )

        # ==== Core ====
        self.core_type = args_dict.get("nn_core_type", "default")

        # ==== Decoder ====
        self.decoder_type = args_dict.get("nn_decoder_type", "default")

        # Score Rescale Mode
        self.score_rescale_mode = args_dict.get("score_rescale_mode", __default_score_rescale_mode__)

        # Setup final layer based on score_rescale_mode
        self.score_type = args_dict.get("score_type", __default_score_type__)
        self.nb_scores = get_score_numbers(self.score_type)
        self.n_cls_token = args_dict.get("n_cls_token", 1)

        self.vae_constraint = False
        self.freeze_embedding = False
        self.do_slope_regularization = False

        # Training Configuration
        self.criterion_type = args_dict.get("criterion_type", "mse_loss")
        self.lr_scheduler_type = args_dict.get("lr_scheduler_type", "cyclic_lr")
        self.optimizer_type = args_dict.get("optimizer_type", "lamb")

        if yml_config_path is not None:
            # Restore configuration
            logger.info(f"Loading model config from {yml_config_path}")
            self.load_model_config(dir_config, yml_config_path=yml_config_path)
            logger.warning(f"Model config initialized based on yaml configuration file {yml_config_path}")
        elif trainer_version_number is not None:
            # Restore configuration
            self.load_model_config(dir_config, trainer_version_number=trainer_version_number)
            logger.warning(f"Model config initialized based on pre-existing model configuration")
        else:
            logger.warning(f"Model configuration initialized without providing yaml configuration file")

        if "ssl" in self.task:
            if "custom_io_encodings" in self.task:
                self.decoder_type = "ssl_double_cls"
            else:
                self.decoder_type = "ssl"
            self.score_rescale_mode = "raw"
            self.criterion_type = "cross_entropy_loss"
            logger.warning(
                f"Configuration of [`decoder_type`, `criterion_type`, `score_rescale_mode`] have been overridden for task `ssl`"
            )

        # Set final layer type base on score rescale mode
        self.final_layer_type = self.setup_final_layer(self.score_rescale_mode)

    def update_config(
        self, tuner_config_dict: dict[str, Any], return_cfg_object: bool = False, pow_two: bool = False
    ) -> None | object:
        """Updater configuraiton function used for automated hyperparameter search."""
        for param in self.__model_params__:
            if pow_two:
                if param in self.__pow_two_params__:
                    value = tuner_config_dict.get(param, int(math.log2(self.__dict__[param])))
                    self.__dict__[param] = int(math.pow(2, value))
                else:
                    self.__dict__[param] = tuner_config_dict.get(param, self.__dict__[param])

            if param == "score_rescale_mode":
                self.final_layer_type = self.setup_final_layer(tuner_config_dict.get(param, self.__dict__[param]))

        if return_cfg_object:
            return self
        else:
            return None

    def setup_final_layer(self, score_rescale_mode: int):
        if score_rescale_mode == "minmax":
            final_layer_type = "tanh"
        elif score_rescale_mode == "standardize":
            final_layer_type = "identity"
        elif score_rescale_mode == "raw":
            final_layer_type = "identity"
        else:
            raise NotImplementedError(
                f"score_rescale_mode {score_rescale_mode} is not supported by model and model config."
            )
        return final_layer_type

    def load_model_config(
        self,
        dir_config: ConfigDir | None = None,
        trainer_version_number: int | None = None,
        yml_config_path: Path | None = None,
    ) -> None:
        assert not (trainer_version_number is None and yml_config_path is None)

        if yml_config_path is not None:
            if yml_config_path.exists():
                model_config_path = yml_config_path
            else:
                raise ValueError(
                    f"The yaml file provided for the model configuraiton does not exists. Received path: {yml_config_path}"
                )
        elif trainer_version_number is not None:
            model_config_path = self._get_model_config_path(dir_config, trainer_version_number)

        with open(model_config_path, "r") as stream:
            logger.info(f"Loading model config from {model_config_path}")
            state_dict = yaml.safe_load(stream)

        for param in self.__model_params__:
            if param in state_dict.keys():
                self.__dict__[param] = state_dict[param]
        logger.warning(
            f"LOADED MODEL CONFIG: {model_config_path}. Overwritten {len(set(state_dict.keys()))}/{len(set(self.__model_params__))} parameters: {state_dict.keys()}"
        )

    def save_model_config(self, dir_config: ConfigDir, trainer_version_number: int) -> None:
        model_config_path = self._get_model_config_path(dir_config, trainer_version_number)

        state_dict = {}
        for param in self.__model_params__:
            state_dict[param] = self.__dict__[param]
        state_dict["batch_size"] = self.__dict__["batch_size"]

        with open(model_config_path, "w") as file:
            yaml.dump(state_dict, file, default_flow_style=False)

        logger.info(f"Model configuration yaml file has been saved to:")
        logger.info(model_config_path)

    def _get_model_config_path(self, dir_config: ConfigDir, trainer_version_number: int):
        assert dir_config is not None
        lightning_version_dir = (
            dir_config.root_output_dir / "trainer_out" / "lightning_logs" / f"version_{trainer_version_number}"
        )
        if not lightning_version_dir.exists():
            lightning_version_dir.mkdir(exist_ok=True, parents=True)
        return lightning_version_dir / "model_config.yml"


class DatasetConfig:
    def __init__(self, args_dict: dict, fast_init: bool = False):
        self.args_dict: dict = args_dict

        self.test_type_name: str = args_dict.get("test_type_name", "fullsweep")

        self.shuffle: bool = args_dict.get("shuffle", True)
        self.seed: int | None = args_dict.get("seed", 626)

        self.batch_size = args_dict.get("batch_size", 128)
        self.nb_workers = args_dict.get("nb_workers", 64)
        if self.nb_workers is None:
            self.nb_workers = 1

        self.score_type = args_dict.get("score_type", __default_score_type__)
        self.nb_scores = get_score_numbers(self.score_type)
        self.score_rescale_mode = args_dict.get("score_rescale_mode", __default_score_rescale_mode__)
        logger.info(f"Score type has been set to {self.score_type}")
        logger.info(f"Score rescale mode has been set to {self.score_rescale_mode}")

        self.exclude_special_designs = args_dict.get("exclude_special_designs", True)

        # Augmentation by default
        self.augment_data = True
        if args_dict.get("no_data_augmentation", False):
            self.augment_data = False

        self.fast_init = fast_init
