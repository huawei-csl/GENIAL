# %%
import matplotlib.pyplot as plt
import torch

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import numpy as np


def export_tensorboard_to_csv(logdir, output_csv):
    pass


# Example usage
log_dirpath = Path(
    "../../../output/multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only/merge_2024-09-13_09-32/trainer_out/lightning_logs/version_47"
)
# log_dirpath = Path("output/multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only/merge_2024-09-13_09-32/trainer_out/lightning_logs/version_25")
output_csv_filepath = Path("output.csv")

# Create an EventAccumulator for the log directory
event_acc = EventAccumulator(str(log_dirpath))

# Load all the event data (e.g., scalar, images, histograms, etc.)
event_acc.Reload()

# Get all the tags for the scalar data (like 'accuracy', 'loss', etc.)
scalar_tags = event_acc.Tags()["scalars"]

# Dictionary to store scalar data by step
tag_dfs = {}

# Fetch scalar values
for tag in scalar_tags:
    # Get scalar events for this tag
    scalar_events = event_acc.Scalars(tag)
    tag_data = []

    for event in scalar_events:
        event_data = {
            "step": event.step,
            tag: event.value,
        }
        tag_data.append(pd.DataFrame([event_data]))

    tag_df = pd.concat(tag_data, ignore_index=True)
    tag_dfs[tag] = tag_df

# Merge all dfs into a single one
for idx, (tag, df) in enumerate(tag_dfs.items()):
    if idx == 0:
        merged_df = df
    else:
        merged_df = pd.merge(left=merged_df, right=df, on="step", how="outer")

# %%
# Try out golden metric


# Normalization for metric_1 and metric_2 (tending to 0 but never reaching)
def normalize_towards_zero(metric):
    return 1 / (1 + metric)


# Normalization for metric_3 (tending to 1)
def normalize_towards_one(metric):
    return metric / 1.0


# Define vectorized function
vec_normalize_towards_0 = np.vectorize(normalize_towards_zero)
vec_normalize_towards_1 = np.vectorize(normalize_towards_one)

# Get metrics values
metric_0_0 = merged_df["loss/val_loss"].to_numpy()
metric_0_1 = merged_df["loss/rvalue_loss"].to_numpy()
metric_1_0 = merged_df["acc/slope_acc"].to_numpy()
# Remove square
# metric_1_0 = 1- (np.sqrt(-(metric_1_0-1)))**1.5
metric_1_0 = 1 - (np.sqrt(-(metric_1_0 - 1))) ** 1.8

lr = merged_df["lr/total"][~merged_df["loss/val_loss"].isna()].to_numpy()
# lr[np.isnan(lr)]=0

# Remove NaN values
metric_0_0 = metric_0_0[~np.isnan(metric_0_0)]
metric_0_1 = metric_0_1[~np.isnan(metric_0_1)]
metric_0_2 = 1 - metric_1_0
metric_0_2 = metric_0_2[~np.isnan(metric_0_2)]


# metric_0_1[~np.isnan(metric_0_1)] = 1.0
# metric_0_1[:450] = 0.05
metric_1_0 = metric_1_0[~np.isnan(metric_1_0)]

# Normalizing all metrics
normalized_metrics = {
    "loss/val_loss": vec_normalize_towards_0(metric_0_0),
    "loss/rvalue_loss": vec_normalize_towards_0(metric_0_1),
    "acc/slope_acc_to0": vec_normalize_towards_0(metric_0_2),
    "acc/slope_acc_to1": vec_normalize_towards_1(metric_1_0),
}

# Weights for each metric
weights = {
    "weight_0_0": 0.67,  # Weight for loss/val_loss
    "weight_0_1": 0.0,  # Weight for loss/rvalue_loss
    "weight_0_2": 0.33,  # Weight for acc/slope_acc_to0
    "weight_1_0": 0.33,  # Weight for acc/slope_acc_to1
}


# 1. Weighted Sum Approach
def weighted_sum_approach(norm_metrics, weights):
    golden_metric = (
        weights["weight_0_0"] * norm_metrics["loss/val_loss"]
        + weights["weight_0_1"] * norm_metrics["loss/rvalue_loss"]
        + weights["weight_1_0"] * norm_metrics["acc/slope_acc_to1"]
    )
    return golden_metric


# 2. Geometric Mean Approach
def geometric_mean_approach(norm_metrics, weights):
    weighted_prod = (
        norm_metrics["loss/val_loss"] ** weights["weight_0_0"]
        * norm_metrics["loss/rvalue_loss"] ** weights["weight_0_1"]
        * norm_metrics["acc/slope_acc_to1"] ** weights["weight_1_0"]
    )
    golden_metric = weighted_prod ** (1 / sum(weights.values()))
    return golden_metric


# 3. Harmonic Mean Approach
def harmonic_mean_approach(norm_metrics, weights):
    golden_metric = sum(weights.values()) / (
        (weights["weight_0_0"] / norm_metrics["loss/val_loss"])
        + (weights["weight_0_1"] / norm_metrics["loss/rvalue_loss"])
        + (weights["weight_1_0"] / norm_metrics["acc/slope_acc_to1"])
    )
    return golden_metric


def logged_sum_approach_v1(norm_metrics, weights):
    golden_metric = -(
        weights["weight_0_0"] * np.log10(norm_metrics["loss/val_loss"])
        + weights["weight_0_1"] * np.log10(norm_metrics["loss/rvalue_loss"])
        + weights["weight_1_0"] * np.log10(norm_metrics["acc/slope_acc_to1"])
    )
    return golden_metric


def logged_sum_approach_v2(norm_metrics, weights):
    golden_metric = -(
        weights["weight_0_0"] * np.log10(norm_metrics["loss/val_loss"])
        + weights["weight_0_1"] * np.log10(norm_metrics["loss/rvalue_loss"])
        + weights["weight_0_2"] * np.log10(norm_metrics["acc/slope_acc_to0"])
    )
    return golden_metric


def logged_sum_approach_v3(norm_metrics, weights):
    golden_metric = -(
        weights["weight_0_0"] * torch.log10(norm_metrics["loss/val_loss"])
        + weights["weight_0_1"] * torch.log10(norm_metrics["loss/rvalue_loss"])
        + weights["weight_0_2"] * torch.log10(norm_metrics["loss/slope_loss"])
        # weights["weight_1_0"]*torch.log10(norm_metrics["acc/slope_acc"])
    )
    return golden_metric


# Calculate the golden metric using different approaches
golden_metric_weighted_sum = weighted_sum_approach(normalized_metrics, weights)
golden_metric_geometric_mean = geometric_mean_approach(normalized_metrics, weights)
golden_metric_harmonic_mean = harmonic_mean_approach(normalized_metrics, weights)
golden_metric_logged_sum_v1 = logged_sum_approach_v1(normalized_metrics, weights)
golden_metric_logged_sum_v2 = logged_sum_approach_v2(normalized_metrics, weights)
golden_metric_logged_sum_v3 = logged_sum_approach_v3(normalized_metrics, weights)

# Plot the various golden metrics
fig, (ax0, ax1) = plt.subplots(2, 1, height_ratios=[0.35, 0.65], figsize=(15, 10))

# Plot values
ax0.plot(metric_0_0, label="loss/val_loss", c="lightseagreen")
ax0.plot(metric_0_1, label="loss/rvalue_loss", c="darkgoldenrod")
# ax0.plot(metric_1_0, label="acc/slope_acc", c="darkkhaki")

# ax0.set_ylim([0.0525, 0.061])
ax0.set_yscale("log")
ax0.legend()
ax0.set_ylabel("Loss/Acc Values")

ax01 = ax0.twinx()
ax01.plot(metric_1_0, label="acc/slope_acc", c="darkkhaki")
# ax01.set_ylim([0.95, 1.001])
# ax01.plot(lr, label="learning_rate", c="black", linestyle="None", marker="x", markersize=1)
ax01.legend()
ax01.set_ylabel("acc/slope_acc")

# Plot metric
ax1.plot(golden_metric_weighted_sum, label="weighted_sum")
ax1.plot(golden_metric_geometric_mean, label="geometric_mean")
ax1.plot(golden_metric_harmonic_mean, label="harmonic_mean")
ax1.plot(golden_metric_logged_sum_v1, label="logged_sum_v1")
ax1.plot(golden_metric_logged_sum_v2, label="logged_sum_v2")
ax1.plot(golden_metric_logged_sum_v3, label="logged_sum_v3")

# Get minimum v2

ax1.axvline(np.argmin(golden_metric_logged_sum_v2), c="red")

# ax1.set_ylim([0.014, 0.025])
ax1.set_yscale("log")
ax1.legend()
ax1.set_ylabel("Golden Value")
plt.show()

# %%
