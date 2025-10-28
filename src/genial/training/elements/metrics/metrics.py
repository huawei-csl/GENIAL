# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

import torch
from torch import nn


def normalize_towards_0(metric):
    return 1 / (1 + metric)


def normalize_towards_1(metric):
    return metric / 1.0


def logged_sum_approach(norm_metrics, weights):
    golden_metric = -(
        weights["weight_0_0"] * torch.log10(norm_metrics["loss/mse_loss"])
        + weights["weight_0_1"] * torch.log10(norm_metrics["loss/rvalue_loss"])
        + weights["weight_0_2"] * torch.log10(norm_metrics["loss/slope_loss"])
        # weights["weight_1_0"]*torch.log10(norm_metrics["acc/slope_acc"])
    )
    return golden_metric


def get_golden_metric(trainer) -> torch.Tensor:
    # Golden Metric Definition
    mse_loss = trainer.callback_metrics.get("loss/mse_loss")
    rvalue_loss = trainer.callback_metrics.get("loss/rvalue_loss")
    slope_acc = trainer.callback_metrics.get("acc/slope_acc")
    slope_loss = 1 - slope_acc

    # Normalize metrics based on their optimial values
    normalized_metrics = {
        "loss/mse_loss": normalize_towards_0(mse_loss),
        "loss/rvalue_loss": normalize_towards_0(rvalue_loss),
        "loss/slope_loss": normalize_towards_0(slope_loss),
        "acc/slope_acc": normalize_towards_1(slope_acc),
    }

    # Weights for each metric (tweak these values as needed)
    weights = {
        "weight_0_0": 0.67,  # Weight for normalized_0_0
        "weight_0_1": 0.00,  # Weight for normalized_0_1
        "weight_0_2": 0.33,  # Weight for normalized_0_2
        "weight_1_0": 0.33,  # Weight for normalized_1_0
    }

    golden_acc = logged_sum_approach(normalized_metrics, weights)
    return golden_acc


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation
    All code taken from https://github.com/wolny/pytorch-3dunet/blob/4a04ccf1f1803ebebdafb6c6ee092bb954f3758d/pytorch3dunet/unet3d/losses.py#L84

    Args:
        weight (torch.Tensor): weight for each class
        mode (str): 'handmade' or 'torchmetrics'
    """

    def __init__(self, nb_target_classes: int = 2, weight=None):
        super().__init__()
        self.weight = weight
        self.eps = 1e-6

        self.nb_target_classes = nb_target_classes

        self.dice = self.compute_per_channel_dice

    def update_weights(self, weight):
        self.weight = weight

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): NxCxSpatial pred tensor
            target (torch.Tensor): NxCxSpatial target tensor

        Out:
            Dice Loss
        """
        return self.dice_loss_func(pred, target)

    def dice_loss_func(self, pred, target):
        # format pred and target
        target = target.type(torch.int)

        # compute per channel Dice coefficient
        per_channel_dice = self.compute_per_channel_dice(pred, target)

        # average Dice score across all channels/classes
        return 1.0 - torch.mean(per_channel_dice)

    def compute_per_channel_dice(self, pred, target):
        """
        Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel pred and target.
        Assumes the pred is a normalized probability, e.g. a result of Sigmoid or Softmax function.
        Args:
            pred (torch.Tensor): NxCxSpatial pred tensor
            target (torch.Tensor): NxCxSpatial target tensor

        Self:
            weight (torch.Tensor): NxC weight for each class for each batch

        Out:
            Channel-wise Dice Coefficient
        """

        # pred and target shapes must match
        assert pred.size() == target.size(), "'pred' and 'target' must have the same shape"

        # Ensure weight tensor is correct shape if provided
        if self.weight is not None:
            assert self.weight.shape[0] == pred.shape[0], "'weight' must have same batch size as 'pred' and 'target'"
            assert self.weight.shape[1] == pred.shape[1] // self.nb_target_classes, (
                "'weight' must have as many channels as 'pred' and 'target'"
            )
            self.weight = self.weight.repeat_interleave(self.nb_target_classes, dim=1)

        # Relevant discussion regarding Dice Loss meaning
        # https://stackoverflow.com/questions/68506223/correct-way-to-reduce-dimension-in-dice-loss
        max_dims = len(target.size())
        reduce_axis = [max_dims - 2, max_dims - 1]

        # compute per channel Dice Coefficient
        intersect = (pred * target).sum(dim=reduce_axis)
        if self.weight is not None:
            # Normalize the weights for each batch
            weight_sum = self.weight.sum(dim=1, keepdim=True)
            normalized_weight = self.weight / weight_sum * pred.size(1)

            # Reshape weight for broadcasting (N, C, 1, 1) and apply to each batch and channel
            normalized_weight = normalized_weight.view(self.weight.size(0), self.weight.size(1), 1, 1)
            intersect = normalized_weight * intersect  # Apply normalized weights here
        denominator = (pred * pred).sum(dim=reduce_axis) + (target * target).sum(dim=reduce_axis)
        dice_coefficient = 2 * (intersect / denominator)
        return dice_coefficient
