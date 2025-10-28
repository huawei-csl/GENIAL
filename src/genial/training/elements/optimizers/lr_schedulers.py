# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

from typing import Any
import math

import torch.optim.lr_scheduler as lr_schedulers


def configure_cyclic_lr(optimizer, **lr_scheduler_kwargs) -> list[dict[str, lr_schedulers.LRScheduler | Any]]:
    step_size_up = lr_scheduler_kwargs["nb_epochs_run"] * lr_scheduler_kwargs["cyclic_up_epochs_frac"]
    step_size_down = lr_scheduler_kwargs["nb_epochs_run"] * lr_scheduler_kwargs["cyclic_down_epochs_frac"]

    cyclic_scheduler = {
        "scheduler": lr_schedulers.CyclicLR(
            optimizer,
            max_lr=lr_scheduler_kwargs["max_lr"],
            base_lr=lr_scheduler_kwargs["base_lr"],
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            mode="exp_range",
            scale_fn=None,
            scale_mode="cycle",
            gamma=0.99994,
            cycle_momentum=True,
            base_momentum=0.8,
            max_momentum=0.9,
        ),
        "interval": "epoch",
        "frequency": 1,
        "name": "lr/cyclic",
    }

    lr_scheduler_config = [cyclic_scheduler]

    return lr_scheduler_config


class WarmupConstantScheduler(lr_schedulers._LRScheduler):
    def __init__(self, optimizer, max_lr, initial_lr, warmup_epochs, constant_epochs, last_epoch=-1, **kwargs):
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.warmup_epochs = max(warmup_epochs, 2)
        self.constant_epochs = constant_epochs
        self.total_epochs = warmup_epochs + constant_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: increase LR from initial_lr to base_lr
            scale = (self.initial_lr / self.max_lr) + (self.last_epoch / (self.warmup_epochs - 1)) * (
                1.0 - (self.initial_lr / self.max_lr)
            )
            return [self.max_lr * scale for _ in self.optimizer.param_groups]
        elif self.last_epoch < self.total_epochs:
            # Constant phase: keep LR at max_lr
            return [self.max_lr for _ in self.optimizer.param_groups]
        else:
            # After constant phase: do not adjust LR, allow ReduceLROnPlateau to control
            return [group["lr"] for group in self.optimizer.param_groups]


class ConstantScheduler(lr_schedulers._LRScheduler):
    def __init__(self, optimizer, max_lr, constant_epochs, last_epoch=-1, **kwargs):
        self.max_lr = max_lr
        self.constant_epochs = constant_epochs
        self.total_epochs = constant_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.total_epochs:
            # Constant phase: keep LR at max_lr
            return [self.max_lr for _ in self.optimizer.param_groups]
        else:
            # After constant phase: do not adjust LR, allow ReduceLROnPlateau to control
            return [group["lr"] for group in self.optimizer.param_groups]


def configure_wup_c_rop_lr(optimizer, **lr_scheduler_kwargs) -> list[dict[str, lr_schedulers.LRScheduler | Any]]:
    """
    Instantiate a 3-stage scheduler:
     - warmup
     - constant
     - reduce on plateau
    """

    max_lr = lr_scheduler_kwargs["max_lr"]  # The constant learning rate after warmup
    base_lr = lr_scheduler_kwargs["base_lr"]
    initial_lr = lr_scheduler_kwargs["initial_lr"]  # Starting learning rate for warmup

    # All fracs are defined with respect to nb_epochs_run
    # i.e., if 100 epochs run, 0.1 results in 0.1*100 = 10 epochs
    warmup_epochs_frac = lr_scheduler_kwargs["warmup_epochs_frac"]  # Number of warmup epochs
    constant_epochs_frac = lr_scheduler_kwargs["constant_epochs_frac"]  # Number of constant epochs
    patience_epochs_frac = lr_scheduler_kwargs["patience_epochs_frac"]  # Patience for ReduceLROnPlateau

    warmup_epochs = math.ceil(warmup_epochs_frac * lr_scheduler_kwargs["nb_epochs_run"])
    constant_epochs = constant_epochs_frac * lr_scheduler_kwargs["nb_epochs_run"]
    patience_epochs = patience_epochs_frac * lr_scheduler_kwargs["nb_epochs_run"]

    # Instantiate the custom scheduler
    warmup_constant_scheduler = WarmupConstantScheduler(
        optimizer,
        max_lr,
        initial_lr,
        warmup_epochs,
        constant_epochs,
    )

    # Scheduler configuration for the custom scheduler
    warmup_constant_scheduler_config = {
        "scheduler": warmup_constant_scheduler,
        "interval": "epoch",
        "frequency": 1,
        "name": "lr/wup_c",
    }

    # ReduceLROnPlateau scheduler
    reduce_on_plateau_scheduler = lr_schedulers.ReduceLROnPlateau(
        optimizer, patience=patience_epochs, factor=0.2, min_lr=base_lr
    )

    # Scheduler configuration for ReduceLROnPlateau
    reduce_on_plateau_config = {
        "scheduler": reduce_on_plateau_scheduler,
        "monitor": "loss/train_loss",
        "interval": "epoch",
        "frequency": 1,
        "name": "lr/wup_c_rop",
        "strict": True,
    }

    # Return scheduler configurations
    return [warmup_constant_scheduler_config, reduce_on_plateau_config]


# Linear Decrease Scheduler
def lr_lindec(epoch, start_epoch, total_epochs, start_lr, end_lr):
    if epoch < start_epoch:
        return 1.0  # No decay before the decay start epoch
    else:
        # Linearly decay from start_lr to end_lr
        return 1 - (epoch - start_epoch) / (total_epochs - start_epoch) * (1 - (end_lr / start_lr))


def configure_wup_c_ldec_lr(optimizer, **lr_scheduler_kwargs) -> list[dict[str, lr_schedulers.LRScheduler | Any]]:
    """
    Instantiate a 3-stage scheduler:
     - warmup
     - constant
     - linear_decrease
    """

    max_lr = lr_scheduler_kwargs["max_lr"]  # The constant learning rate after warmup
    base_lr = lr_scheduler_kwargs["base_lr"]

    # All fracs are defined with respect to nb_epochs_run
    # i.e., if 100 epochs run, 0.1 results in 0.1*100 = 10 epochs
    warmup_epochs_frac = lr_scheduler_kwargs["warmup_epochs_frac"]  # Number of warmup epochs
    constant_epochs_frac = lr_scheduler_kwargs["constant_epochs_frac"]  # Number of constant epochs

    warmup_epochs = math.ceil(warmup_epochs_frac * lr_scheduler_kwargs["nb_epochs_run"])
    constant_epochs = constant_epochs_frac * lr_scheduler_kwargs["nb_epochs_run"]

    # Instantiate the custom scheduler
    warmup_constant_scheduler = ConstantScheduler(
        optimizer,
        max_lr,
        constant_epochs,
    )

    def _lr_lambda_lindec(epoch):
        return lr_lindec(
            epoch,
            start_epoch=0,  # Sequential scheduler makes it appear as if start_epoch is zero when switching milestone
            total_epochs=lr_scheduler_kwargs["nb_epochs_run"] - warmup_epochs - constant_epochs,
            start_lr=max_lr,
            end_lr=base_lr,
        )

    linear_decrease_scheduler = lr_schedulers.LambdaLR(optimizer, lr_lambda=_lr_lambda_lindec)

    # Define the sequential scheduler
    total_scheduler = lr_schedulers.SequentialLR(
        optimizer,
        schedulers=[
            warmup_constant_scheduler,
            linear_decrease_scheduler,
        ],
        milestones=[
            constant_epochs + warmup_epochs,
        ],
    )

    # Full scheduler configuration
    warmup_constant_lindec_scheduler = {
        "scheduler": total_scheduler,
        "interval": "epoch",
        "frequency": 1,
        "name": "lr/wup_c_lindec",
    }

    # Return scheduler configurations
    return [warmup_constant_lindec_scheduler]


def configure_c_ldec_lr(optimizer, **lr_scheduler_kwargs) -> list[dict[str, lr_schedulers.LRScheduler | Any]]:
    """
    Instantiate a 2-stage scheduler:
     - constant
     - linear_decrease
    """

    max_lr = lr_scheduler_kwargs["max_lr"]  # The constant learning rate after warmup
    base_lr = lr_scheduler_kwargs["base_lr"]
    initial_lr = lr_scheduler_kwargs["initial_lr"]  # Starting learning rate for warmup

    # All fracs are defined with respect to nb_epochs_run
    # i.e., if 100 epochs run, 0.1 results in 0.1*100 = 10 epochs
    # warmup_epochs_frac = lr_scheduler_kwargs["warmup_epochs_frac"] # Number of warmup epochs
    constant_epochs_frac = lr_scheduler_kwargs["constant_epochs_frac"]  # Number of constant epochs

    # warmup_epochs = math.ceil(warmup_epochs_frac * lr_scheduler_kwargs["nb_epochs_run"])
    constant_epochs = constant_epochs_frac * lr_scheduler_kwargs["nb_epochs_run"]

    # Instantiate the custom scheduler
    warmup_constant_scheduler = WarmupConstantScheduler(
        optimizer,
        max_lr,
        initial_lr,
        0,
        constant_epochs,
    )

    def _lr_lambda_lindec(epoch):
        return lr_lindec(
            epoch,
            start_epoch=0,  # Sequential scheduler makes it appear as if start_epoch is zero when switching milestone
            total_epochs=lr_scheduler_kwargs["nb_epochs_run"] - constant_epochs,
            start_lr=max_lr,
            end_lr=base_lr,
        )

    linear_decrease_scheduler = lr_schedulers.LambdaLR(optimizer, lr_lambda=_lr_lambda_lindec)

    # Define the sequential scheduler
    total_scheduler = lr_schedulers.SequentialLR(
        optimizer,
        schedulers=[
            warmup_constant_scheduler,
            linear_decrease_scheduler,
        ],
        milestones=[
            constant_epochs,
        ],
    )

    # Full scheduler configuration
    constant_lindec_scheduler = {
        "scheduler": total_scheduler,
        "interval": "epoch",
        "frequency": 1,
        "name": "lr/c_lindec",
    }

    # Return scheduler configurations
    return [constant_lindec_scheduler]


def old_scheduler_configs(optimizer, **lr_scheduler_kwargs):
    # # Lambda Function for Warmup
    # lambda_warmup = lambda step: min(step / warmup_steps, 1)

    # warmup_scheduler = {
    #     'scheduler': torch.optim.lr_scheduler.LinearLR(
    #         optimizer,
    #         start_factor=warmup_start_factor,
    #         end_factor=warmup_end_factor,
    #         total_iters=warmup_steps,
    #     ),
    #     'interval': 'step',
    #     'name': 'lr/warmup'
    # }
    # linear_scheduler = {
    #     'scheduler': torch.optim.lr_scheduler.LinearLR(
    #         optimizer,
    #         start_factor=0.95,
    #         end_factor=1.0,
    #         total_iters=10,
    #     ),
    #     'interval': 'step',
    #     'name': 'lr/linear'
    # }

    # # Does not work with other schedulings with current model
    # reduce_on_plateau_scheduler = {
    #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer,
    #         mode='min',
    #         factor=0.1,
    #         patience=10,
    #         # verbose=True
    #     ),
    #     'monitor': 'loss/val_loss',
    #     'interval': 'epoch',
    #     'name': 'lr/plateau'
    # }

    #  total_scheduler = {
    #         'scheduler': torch.optim.lr_scheduler.SequentialLR(
    #             optimizer=optimizer,
    #             schedulers=[warmup_scheduler["scheduler"], linear_scheduler["scheduler"]],
    #             milestones=[warmup_steps]),
    #         'interval': 'step',
    #         'name': 'lr/linear_reduce_lr'
    #     }
    # linear_warmup_with_cosine_annealing_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=3*steps_per_epoch,
    #     num_steps_per_epoch=steps_per_epoch,
    #     num_cycles_p_training=0.5,
    #     max_epochs=self.trainer.max_epochs
    # )

    # total_scheduler = {
    #     'scheduler': cyclic_scheduler["scheduler"],
    #     'interval': cyclic_scheduler['interval'],
    #     'name': cyclic_scheduler['name'],
    # }

    # total_scheduler = {
    #     'scheduler': linear_warmup_with_cosine_annealing_scheduler,
    #     'interval': 'step',
    #     'name': "lr/wup_cos_anneal",
    # }

    return []
