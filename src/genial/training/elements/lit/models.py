# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

from typing import Any
from pathlib import Path
from loguru import logger

import matplotlib.pyplot as plt

import matplotlib


import lightning as L
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from genial.training.elements.models import FullTransformer
from genial.training.elements.configs import ModelConfig
from genial.training.elements.metrics import metrics
import genial.training.elements.optimizers.lr_schedulers as lr_schedulers

from torch.optim import Adam
from genial.training.elements.optimizers.optimizers import Lamb
from genial.training.elements.optimizers.ext.soap.soap import SOAP

from scipy.stats.mstats import linregress
import math
import numpy as np

# Set backend to MPLBACKEND="Agg"
matplotlib.use("Agg")


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_steps_per_epoch, num_cycles_p_training=0.5, max_epochs=6000
):
    def lr_lambda(current_step):
        # Linear Warmup Phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Linear decay phase with cosine annealing
        epochs_per_full_cycle = num_cycles_p_training * max_epochs
        overall_progress = float(current_step - num_warmup_steps) / float(max(1, max_epochs * num_steps_per_epoch))
        epoch_progress = float(current_step - num_warmup_steps) / float(
            max(1, epochs_per_full_cycle * num_steps_per_epoch)
        )
        return max(
            1.0e-6, 0.5 * (1.0 + math.cos(math.pi * epoch_progress) * (1 - overall_progress)) - 0.5 * overall_progress
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def weighted_mse_loss(y, y_expected, reduction, set_type):
    # For reporting purposes, we only adjust the weight on the training set
    if set_type != "train":
        return torch.nn.functional.mse_loss(y, y_expected, reduction=reduction)
    # Custom weight function. Equals 1 when y_expected >= -2.3. Ow, increase exponentially from 1 (at y_expected = -2.3)
    # to about 3.5 for the encoding with the lowest number of transistors.
    weight = torch.where(y_expected < -2.3, torch.exp(-y_expected - 1.3) / math.exp(1), 1)
    # Get unweighted mse loss
    mse_loss_vec = torch.nn.functional.mse_loss(y, y_expected, reduction="none")
    # Multiply by the weight
    weighted_mse_loss_vec = weight * mse_loss_vec
    # Reduce tensor
    if reduction == "none":
        return weighted_mse_loss_vec
    elif reduction == "sum":
        return weighted_mse_loss_vec.sum()
    elif reduction == "mean":
        return weighted_mse_loss_vec.mean()
    else:
        raise ValueError('reduction must be "none", "sum", or "mean"')


class AbstractLitModule(L.LightningModule):
    __available_criterion_types__ = [
        "mse_loss",
        "huber_loss",
        "weighted_mse_loss",
        "cross_entropy_loss",
    ]

    __available_scheduler_types__ = [
        "cyclic_lr",
        "warmup_constant_plateau_lr",
        "warmup_constant_lindec_lr",
        "constant_lindec_lr",
    ]

    __available_optimizer_types__ = [
        "adam",
        "lamb",
        "soap",
    ]

    # All fracs are defined with respect to nb_epochs_run
    # i.e., if 100 epochs run, 0.1 results in 0.1*100 = 10 epochs
    __lightning_module_params__ = {
        "criterion_type": "mse_loss",
        "optimizer_type": "soap",
        "lr_scheduler_type": "cyclic_lr",
        "max_scratch_lr": 1e-3,
        "max_restart_lr": 5.0e-4,
        "base_lr_factor_scratch": 1 / 20.0,
        "base_lr_factor_restart": 1 / 20.0,
        "warmup_start_scratch_factor": 1 / 200,
        "warmup_start_restart_factor": 1 / 200,
        "warmup_epochs_frac_scratch": 0.05,
        "warmup_epochs_frac_restart": 0.01,
        "constant_epochs_frac_scratch": 0.2,
        "constant_epochs_frac_restart": 0.2,
        "patience_epochs_frac_scratch": 0.1,
        "patience_epochs_frac_restart": 0.1,
        "cyclic_up_epochs_frac_scratch": 0.05,
        "cyclic_up_epochs_frac_restart": 0.05,
        "cyclic_down_epochs_frac_scratch": 0.2,
        "cyclic_down_epochs_frac_restart": 0.2,
        "batch_size": 512,
        "freeze_embedding": False,
        "do_slope_regularization": False,
    }

    def __init__(
        self,
        meta_model: FullTransformer | torch.nn.Module,
        model_config: ModelConfig | dict[str, Any] | None = None,
        restored_steps: int = 0,
        restored_epochs: int = 0,
        do_validation_plots: bool = True,
    ):
        super().__init__()
        self.transformer = meta_model
        self.restored_steps = restored_steps
        self.restored_epochs = restored_epochs

        self.setup_config(model_config)
        self.batch_size = self.__lightning_module_params__.get("batch_size")
        logger.info(f"Lightning Module batch size has been set to {self.batch_size}")

        self.setup_criterion()

        if self.__lightning_module_params__.get("d"):
            logger.warning(f"Embedding will be frozen.")
            self.transformer.embedding.requires_grad_(False)

        self.do_validation_plots = do_validation_plots
        self.validation_preds = []
        self.validation_targets = []
        self.validation_criterion_losses = []
        self.validation_mse_losses = []

    def setup_config(self, config: ModelConfig | dict[str, Any] | None) -> None:
        """Helper function to simplify instantiation of the lightning module from different trainers, inference scripts."""
        updated_params = []
        if config is not None:
            for key in self.__lightning_module_params__:
                val = None
                if isinstance(config, ModelConfig):
                    if hasattr(config, key):
                        val = getattr(config, key)
                else:
                    val = config.get(key, self.__lightning_module_params__[key])

                if val is not None and val != self.__lightning_module_params__[key]:
                    self.__lightning_module_params__.update({key: val})
                    updated_params.append(key)

        if len(updated_params) != 0:
            logger.info(f"Parameters {updated_params} have been overwritten by a model config in Lightning Module.")
        else:
            logger.info(f"No parameters have been overwritten by a model config in Lightning Module.")

    def configure_optimizers(self):
        ### Setup Required Variables
        max_epochs = max(1, self.trainer.max_epochs)
        nb_training_steps_run = self.trainer.estimated_stepping_batches
        nb_epochs_run = max_epochs
        logger.debug(
            f"max_epochs: {max_epochs} | nb_training_steps_run: {nb_training_steps_run} | nb_epochs_run: {nb_epochs_run}"
        )
        # steps_per_epoch = nb_training_steps_run/nb_epochs_run

        if self.restored_steps == 0:
            param_key = "scratch"
            logger.info(f"Optimizer configured in starting from scratch mode")
        else:
            param_key = "restart"
            logger.info(f"Optimizer configured in restart mode")

        # Learning Rates Definition
        max_lr = self.__lightning_module_params__.get(f"max_{param_key}_lr")
        base_lr = max_lr * self.__lightning_module_params__.get(f"base_lr_factor_{param_key}")
        initial_lr = max_lr * self.__lightning_module_params__.get(f"warmup_start_{param_key}_factor")

        # Fracs
        warmup_epochs_frac = self.__lightning_module_params__.get(f"warmup_epochs_frac_{param_key}")
        constant_epochs_frac = self.__lightning_module_params__.get(f"constant_epochs_frac_{param_key}")
        patience_epochs_frac = self.__lightning_module_params__.get(f"patience_epochs_frac_{param_key}")
        cyclic_up_epochs_frac = self.__lightning_module_params__.get(f"cyclic_up_epochs_frac_{param_key}")
        cyclic_down_epochs_frac = self.__lightning_module_params__.get(f"cyclic_down_epochs_frac_{param_key}")

        # Setup kwargs for LR Scheduler configuration functions
        lr_scheduler_kwargs = {
            "max_lr": max_lr,
            "base_lr": base_lr,
            "initial_lr": initial_lr,
            "nb_epochs_run": nb_epochs_run,
        }

        ### Setup Optimizer
        if self.__lightning_module_params__.get("optimizer_type") == "lamb":
            # That should be the default
            optimizer = Lamb(self.parameters(), lr=max_lr, weight_decay=0.01)
        elif self.__lightning_module_params__.get("optimizer_type") == "soap":
            optimizer = SOAP(self.parameters(), lr=max_lr, weight_decay=0.01)
        elif self.__lightning_module_params__.get("optimizer_type") == "adam":
            optimizer = Adam(self.parameters(), lr=max_lr, weight_decay=0.01)

        ### Setup LR Scheduler
        if self.__lightning_module_params__.get("lr_scheduler_type") == "cyclic_lr":
            lr_scheduler_kwargs.update(
                {
                    "nb_training_steps_run": nb_training_steps_run,
                    "cyclic_up_epochs_frac": cyclic_up_epochs_frac,
                    "cyclic_down_epochs_frac": cyclic_down_epochs_frac,
                }
            )
            lr_scheduler_config = lr_schedulers.configure_cyclic_lr(optimizer, **lr_scheduler_kwargs)

        elif self.__lightning_module_params__.get("lr_scheduler_type") == "warmup_constant_plateau_lr":
            lr_scheduler_kwargs.update(
                {
                    "warmup_epochs_frac": warmup_epochs_frac,
                    "constant_epochs_frac": constant_epochs_frac,
                    "patience_epochs_frac": patience_epochs_frac,
                }
            )
            lr_scheduler_config = lr_schedulers.configure_wup_c_rop_lr(optimizer, **lr_scheduler_kwargs)

        elif self.__lightning_module_params__.get("lr_scheduler_type") == "warmup_constant_lindec_lr":
            lr_scheduler_kwargs.update(
                {
                    "warmup_epochs_frac": warmup_epochs_frac,
                    "constant_epochs_frac": constant_epochs_frac,
                }
            )
            lr_scheduler_config = lr_schedulers.configure_wup_c_ldec_lr(optimizer, **lr_scheduler_kwargs)

        elif self.__lightning_module_params__.get("lr_scheduler_type") == "constant_lindec_lr":
            lr_scheduler_kwargs.update(
                {
                    "constant_epochs_frac": constant_epochs_frac,
                }
            )
            lr_scheduler_config = lr_schedulers.configure_c_ldec_lr(optimizer, **lr_scheduler_kwargs)

        else:
            raise NotImplementedError(
                f"Scheduler type {self.__lightning_module_params__.get('lr_scheduler_type')} is not implemented"
            )

        logger.info(
            f"LR schedulers {[scheduler['name'] for scheduler in lr_scheduler_config]} have been set with parameters {lr_scheduler_kwargs}"
        )
        logger.info(f"Training will be run for max_epochs={self.trainer.max_epochs}")
        return [optimizer], lr_scheduler_config

    def compute_golden_metric(self) -> torch.Tensor:
        raise NotImplementedError()

    def setup_criterion(self):
        if self.__lightning_module_params__.get("criterion_type") == "mse_loss":

            def criterion(y, y_expected, kwargs):
                return torch.nn.functional.mse_loss(y, y_expected, reduction=kwargs["reduction"])

            logger.info(f"Training criterion as been set to MSE Loss")
        elif self.__lightning_module_params__.get("criterion_type") == "huber_loss":

            def criterion(y, y_expected, kwargs):
                return torch.nn.functional.huber_loss(y, y_expected, reduction=kwargs["reduction"])

            logger.info(f"Training criterion as been set to Huber Loss")
        elif self.__lightning_module_params__.get("criterion_type") == "weighted_mse_loss":

            def criterion(y, y_expected, kwargs):
                return weighted_mse_loss(y, y_expected, reduction=kwargs["reduction"], set_type=kwargs["set_type"])

            logger.info(f"Training criterion as been set to weighted MSE Loss")
        elif self.__lightning_module_params__.get("criterion_type") == "cross_entropy_loss":

            def criterion(y, y_expected, kwargs):
                return torch.nn.functional.cross_entropy(
                    y.reshape(-1, y.shape[-1]), y_expected.reshape(-1), reduction=kwargs["reduction"]
                )

            logger.info(f"Training criterion as been set to cross entropy Loss")
        elif self.__lightning_module_params__.get("criterion_type") == "custom_ssl":
            # TODO
            pass
        else:
            msg = f"criterion_type: {self.__lightning_module_params__.get('criterion_type')} is not implemented."
            logger.error(msg)
            raise ValueError(msg)
        self.criterion = criterion


class LitTransformer(AbstractLitModule):
    """Top class used to properly setup the lightning module to be used depending on the task."""

    def __init__(self, *args, **kwargs):
        self.__morph__(*args, **kwargs)

    def __morph__(self, *args, **kwargs):
        """
        Morph this top LitTransformer instance into the instance of the correct class
        """

        # Get the task
        task = LitTransformer.__get_task__(*args, **kwargs)

        # Get the correct class to instantiate based on the task
        if "custom_io_encodings" in task and "ssl" in task:
            self.__class__ = SSLDoubleClsLitTransformer
        elif "ssl" in task:
            self.__class__ = SSLLitTransformer
        else:
            self.__class__ = DefaultLitTransformer

        # Overrides the instance of this class with the one from the class to use
        instance = self.__class__(*args, **kwargs)
        self.__dict__ = instance.__dict__

    @staticmethod
    def __get_task__(*args, **kwargs) -> AbstractLitModule:
        # Get the task that will be executed
        task = None
        for arg in args:
            if isinstance(arg, ModelConfig):
                task = arg.task
        if task is None:
            model_config = kwargs.get("model_config", None)
            if model_config is not None:
                task = model_config.task
        assert task is not None

        return task


def slope_regularization(
    y_pred: torch.Tensor, y_true: torch.Tensor, nb_sores: int = 1, regularization_lambda: float = 1.0
):
    """
    Computes an additional regularization term that penalizes deviation
    from the ideal slope (1.0) in the linear regression of y_pred vs. y_true.

    Args:
        y_pred (Tensor): shape (batch_size, num_objectives)
        y_true (Tensor): shape (batch_size, num_objectives)
        slope_lambda (float): regularization strength

    Returns:
        Tensor: regularization factor
    """

    slope_regs = []
    for score_idx in range(nb_sores):
        # Mean centering
        y_true_centered = y_true[:, score_idx] - y_true[:, score_idx].mean(dim=0)
        y_pred_centered = y_pred[:, score_idx] - y_pred[:, score_idx].mean(dim=0)

        # Compute slope (β) analytically per objective
        numerator = (y_true_centered * y_pred_centered).sum(dim=0)
        denominator = (y_true_centered**2).sum(dim=0) + 1e-8  # avoid division by zero
        slopes = numerator / denominator  # shape: (num_objectives,)

        # Slope regularization: penalize deviation from slope=1
        slope_reg = ((slopes - 1.0) ** 2).mean()

        # Total loss
        slope_regs.append(regularization_lambda * slope_reg)

    return torch.sum(torch.stack(slope_regs))


class DefaultLitTransformer(AbstractLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"DefaultLitTransformer has been initialized")

    def training_step(self, batch, batch_idx):
        """Training loop"""
        x = batch["encodings"]
        values = batch["values"]
        y_expected = batch["scores"]

        y, vae_output = self.transformer(x, values)
        if y.ndim > 1 and y_expected.ndim != y.ndim:
            y_expected = y_expected.unsqueeze(1)

        if vae_output is None:
            kl_loss = 0
        else:
            kl_loss = kl_divergence_loss(vae_output)

        if self.__lightning_module_params__.get("do_slope_regularization", False):
            slope_reg = slope_regularization(y, y_expected, regularization_lambda=1.0)
        else:
            slope_reg = 0

        if self.transformer.decoder.nb_scores > 1:
            losses = []
            for idx in range(self.transformer.decoder.nb_scores):
                losses.append(
                    self.criterion(y[..., idx], y_expected[..., idx], {"reduction": "mean", "set_type": "train"})
                )
            train_loss = torch.sum(torch.stack(losses)) + kl_loss + slope_reg

        else:
            train_loss = self.criterion(y, y_expected, {"reduction": "mean", "set_type": "train"}) + kl_loss + slope_reg

        self.log("loss/train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return train_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx):
        """Validation loop"""

        x = batch["encodings"]
        values = batch["values"]
        y_expected = batch["scores"]

        y: torch.Tensor
        y, vae_output = self.transformer(x, values)

        if y.ndim != y_expected.ndim:
            if y_expected.ndim > 1:
                if y.ndim > y_expected.ndim:
                    y = y.squeeze()

            elif y.ndim > 1:
                y_expected = y_expected.unsqueeze(1)

        if vae_output is None:
            kl_loss = 0
        else:
            kl_loss = kl_divergence_loss(vae_output)

        # Calculate loss
        val_loss = self.criterion(y, y_expected, {"reduction": "none", "set_type": "valid"}) + kl_loss
        mse_loss = torch.nn.functional.mse_loss(y, y_expected, reduction="none")

        self.validation_preds.append(y.detach())
        self.validation_targets.append(y_expected.detach())
        self.validation_criterion_losses.append(val_loss.detach())
        self.validation_mse_losses.append(val_loss.detach())

        val_loss = torch.mean(val_loss, dim=0)
        mse_loss = torch.mean(mse_loss, dim=0)
        if self.transformer.decoder.nb_scores >= 2:
            val_loss = torch.sum(val_loss)
            mse_loss = torch.sum(mse_loss)

        # Note: we use mse_loss as validation loss to be able to compare their values no matter which criterion is used
        log_dict = {
            "loss/val_loss": mse_loss,
            "loss/mse_loss": mse_loss,
            "loss/kl_loss": kl_loss,
        }
        self.log_dict(log_dict, on_epoch=True)

        return log_dict

    def on_validation_epoch_end(self):
        # Concatenate all predictions and targets across batches
        all_preds = (
            torch.atleast_1d(
                torch.cat(self.validation_preds, dim=0),
            )
            .cpu()
            .numpy()
        )
        all_targets = (
            torch.atleast_1d(
                torch.cat(self.validation_targets, dim=0),
            )
            .cpu()
            .numpy()
        )
        all_mse_losses = (
            torch.atleast_1d(
                torch.cat(self.validation_mse_losses, dim=0),
            )
            .cpu()
            .numpy()
        )

        # Make sure to have at least two dimensions
        if all_targets.ndim == 1:
            all_preds = all_preds.unsqueeze(1)
            all_targets = all_targets.unsqueeze(1)
            all_mse_losses = all_mse_losses.unsqueeze(1)

        # Perform linear regression over the full validation set
        all_rvalue_losses = []
        all_slope_accuracies = []
        nb_scores = all_targets.shape[1]
        for i in range(nb_scores):
            linreg_results = linregress(
                all_preds,
                all_targets,
            )
            log_dict = {
                f"loss/rvalue_loss_{i}": 1 - linreg_results.rvalue,
                f"acc/slope_acc_{i}": 1 - (abs(linreg_results.slope - 1)) ** 1.8,
            }
            all_rvalue_losses.append(log_dict[f"loss/rvalue_loss_{i}"])
            all_slope_accuracies.append(log_dict[f"acc/slope_acc_{i}"])

        # Log more metrics
        log_dict = {
            "loss/rvalue_loss": np.mean(all_rvalue_losses),
            "acc/slope_acc": np.mean(all_slope_accuracies),
        }
        self.log_dict(log_dict)

        # Plot
        if self.do_validation_plots:
            fig, axes = plt.subplots(1, nb_scores, figsize=(nb_scores * 5, 5))
            if nb_scores == 1:
                axes = [axes]

            for idx in range(nb_scores):
                # val_loss_cpu = val_loss.cpu()
                scatter = axes[idx].scatter(all_targets[:, idx], all_preds[:, idx], c=all_mse_losses[:, idx])
                axes[idx].set_xlabel(f"Real Scores {idx}")
                axes[idx].set_ylabel("Predicted Scores")
                cbar = plt.colorbar(scatter)
                cbar.set_label("MSE Loss")
            plt.tight_layout()

            # Save original validation plot if we are finetuning the model
            if self.trainer.current_epoch == 0 and self.restored_epochs != 0:
                save_path = Path(self.logger.log_dir) / f"init_validation_plot_epochs{self.restored_epochs}.png"
                plt.title(f"restored_epochs:{self.restored_epochs} - version_number:{self.logger.version}")
                for idx in range(nb_scores):
                    to_loop = (
                        [axes[idx].title, axes[idx].xaxis.label, axes[idx].yaxis.label]
                        + axes[idx].get_xticklabels()
                        + axes[idx].get_yticklabels()
                    )
                for item in to_loop:
                    item.set_fontsize(10)
                plt.savefig(save_path)
                logger.info(f"Initial Validation plot saved at:")
                logger.info(save_path)

            if self.do_validation_plots:
                tensorboard: SummaryWriter = self.logger.experiment
                tensorboard.add_figure("pred_vs_real", plt.gcf(), self.global_step)

            plt.close()

        golden_acc = self.compute_golden_metric()
        self.log_dict({"acc/golden_acc": golden_acc})

        # Clear the lists for the next epoch
        self.validation_preds.clear()
        self.validation_targets.clear()
        self.validation_mse_losses.clear()
        self.validation_criterion_losses.clear()

    def test_step(self, batch, batch_idx):
        """Test loop"""
        x = batch["encodings"]
        values = batch["values"]
        y_expected = batch["scores"]

        y = self.transformer(x, values)
        if len(y.shape) > 1:
            y_expected = y_expected.unsqueeze(1)

        test_loss = self.criterion(y, y_expected, {"reduction": "mean", "set_type": "test"})
        self.log(
            "loss/test_loss", test_loss
        )  # That does not seem to store the value in logged metrics (but it makes the test_step return the loss in a dictionnary)

        return test_loss

    def predict_step(self, batch):
        """Inference loop"""
        x = batch["encodings"]
        values = batch["values"]
        y_expected = batch["scores"]

        y = self.transformer(x, values)
        return y, y_expected

    def compute_golden_metric(self) -> torch.Tensor:
        return metrics.get_golden_metric(self.trainer)


class SSLLitTransformer(AbstractLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_weight = 1
        logger.info(f"SSLLitTransformer has been initialized")

    def training_step(self, batch, batch_idx):
        """Training loop"""
        x = batch["encodings"]
        values = batch["values"]
        y_expected = batch["scores"]

        y, vae_output = self.transformer(x, values)

        if vae_output is None:
            kl_loss = 0
        else:
            kl_loss = kl_divergence_loss(vae_output)

        if isinstance(y_expected, list):
            ce_loss0 = torch.nn.functional.cross_entropy(y[..., 0], y_expected[0])
            ce_loss1 = torch.nn.functional.cross_entropy(y[..., 1], y_expected[1])
            ce_loss = torch.sum(torch.stack([ce_loss0, ce_loss1])) / 2
        else:
            ce_loss = torch.nn.functional.cross_entropy(y, y_expected)

        train_loss = ce_loss.mean() + self.kl_weight * kl_loss

        self.log("loss/train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx):
        """Validation loop"""

        x = batch["encodings"]
        values = batch["values"]
        y_expected = batch["scores"]

        y: torch.Tensor
        y, vae_output = self.transformer(x, values)

        if vae_output is None:
            kl_loss = 0
            mu = 0
            log_var = 0
        else:
            kl_loss = kl_divergence_loss(vae_output)
            mu = vae_output[0].mean()
            log_var = vae_output[1].mean()

        if isinstance(y_expected, list):
            ce_loss0 = torch.nn.functional.cross_entropy(y[..., 0], y_expected[0])
            ce_loss1 = torch.nn.functional.cross_entropy(y[..., 1], y_expected[1])
            ce_loss = torch.sum(torch.stack([ce_loss0, ce_loss1])) / 2
        else:
            ce_loss = torch.nn.functional.cross_entropy(y, y_expected)

        val_loss = ce_loss.mean() + self.kl_weight * kl_loss

        log_dict = {
            "loss/val_loss": torch.mean(val_loss),
            "loss/vae_mu": mu,
            "loss/vae_log_var": log_var,
        }
        self.log_dict(log_dict, on_epoch=True)

        return log_dict


class SSLDoubleClsLitTransformer(AbstractLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"SSLDoubleClsLitTransformer has been initialized")

    def training_step(self, batch, batch_idx):
        """Training loop"""
        x = batch["encodings"]
        values = batch["values"]
        y_expected = batch["scores"]

        y, z = self.transformer(x, values)

        train_loss1 = torch.nn.functional.cross_entropy(y, y_expected[0], reduction="none")
        train_loss2 = torch.nn.functional.cross_entropy(z[..., 0], y_expected[1], reduction="none")
        train_loss3 = torch.nn.functional.cross_entropy(z[..., 1], y_expected[2], reduction="none")

        train_loss = train_loss1.mean() + 0.5 * (train_loss2.mean() + train_loss3.mean())
        self.log("loss/train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx):
        """Validation loop"""

        x = batch["encodings"]
        values = batch["values"]
        y_expected = batch["scores"]

        y: torch.Tensor
        y, z = self.transformer(x, values)

        val_loss1 = torch.nn.functional.cross_entropy(y, y_expected[0], reduction="none")
        val_loss2 = torch.nn.functional.cross_entropy(z[..., 0], y_expected[1], reduction="none")
        val_loss3 = torch.nn.functional.cross_entropy(z[..., 1], y_expected[2], reduction="none")

        val_loss = val_loss1.mean() + 0.5 * (val_loss2.mean() + val_loss3.mean())

        log_dict = {
            "loss/val_loss": torch.mean(val_loss),
        }
        self.log_dict(log_dict, on_epoch=True)

        return log_dict


def kl_divergence_loss(vae_output):
    mu, logvar = vae_output
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2])
    return kl_loss.mean()
