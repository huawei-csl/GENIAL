# %%
# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

import sys
import os
import argparse
from pathlib import Path
from typing import Any
from copy import copy
import shutil
from time import gmtime, strftime, time, localtime
import json

from genial.experiment.file_parsers import process_pool_helper
from tqdm import tqdm

from loguru import logger

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import heapq

from genial.config.config_dir import ConfigDir
from genial.experiment.task_generator import DesignGenerator
from genial.experiment.binary_operators import min_value_tc
from genial.experiment.task_analyzer import analyzer_parser
from genial.experiment import file_parsers
from genial.experiment.task_recommender import EncodingRecommender
import lightning as L

from genial.training.elements.configs import DatasetConfig, ModelConfig
from genial.training.mains.trainer_enc_to_score_value import EncToScoreTrainer
from genial.training.elements.lit.models import LitTransformer
from genial.training.elements.models import FullTransformer
from genial.training.elements.datasets import SwactedDesignDatamodule
from genial.training.elements.utils import setup_analyzer
from genial.training.elements.score_tools import ScoreComputeHelper

from genial.utils.utils import (
    save_serialized_data,
    load_serialized_data,
    from_binstr_list_to_int_array,
    load_database,
    _any_duplicate,
    enc_dict_to_tensor,
)
from genial.utils.utils import send_email as _send_email
import textwrap

from torch.utils.tensorboard import SummaryWriter

from genial.utils.utils import extract_int_string_from_string
from genial.globals import global_vars

from genial.training.elements.lit.models import kl_divergence_loss


plt.rcParams["font.size"] = 20


def make_score_distribution_plot(
    db_dict: dict[str, pd.DataFrame], output_plot_dirpath: Path, args_dict: dict[str, Any]
):
    logger.info(f"Generating score distribution per iteration plot.")

    recom_df = db_dict["recom"]
    synth_df = db_dict["synth"]

    recom_df = EncodingRecommender.fix_recom_db_iter_counts(recom_df=recom_df)
    real_recom_iter_nb_serie = recom_df["real_recom_iter_nb"]

    plt.plot(recom_df["recom_iter_nb"], label="Original")
    plt.plot(recom_df["real_recom_iter_nb"], label="Corrected")
    plt.legend()

    scores = synth_df["nb_transistors"]

    # Plot box plots for each iteration
    scores = []
    iteration_nbs = []
    for real_iter_nb in np.unique(real_recom_iter_nb_serie):
        # Remember iteration number
        iteration_nbs.append(real_iter_nb)

        # Get all design numbers for each iteration
        design_numbers = set(recom_df.loc[recom_df["real_recom_iter_nb"] == real_iter_nb, "design_number"])

        # Get all scores for each iteration number
        scores.append(synth_df.loc[synth_df["design_number"].isin(design_numbers), "nb_transistors"].to_numpy())

    # Rearange the packets of score based on the iteration number
    sorted_lists = sorted(zip(iteration_nbs, scores))
    iteration_nbs, scores = zip(*sorted_lists)

    # Concatenate iterations that are very small
    score_lengths = []

    new_scores = []
    for score_vec in scores:
        score_lengths.append(len(score_vec))

        if len(score_vec) > 1000:
            new_scores.append(score_vec)
        else:
            if len(score_vec) == 1000:
                if len(new_scores[-1]) < 20000:
                    new_scores[-1] = np.concat([new_scores[-1], score_vec])
                else:
                    new_scores.append(score_vec)
            else:
                if len(new_scores[-1]) < 20000:
                    new_scores[-1] = np.concat([new_scores[-1], score_vec])
                else:
                    new_scores.append(score_vec)

    fig, ax0 = plt.subplots(1, 1, figsize=(20, 10))
    box_plot_data = ax0.boxplot(new_scores)
    ax0.set_xlabel("Loop Iteration Number")
    ax0.set_ylabel("Score Distribution")
    # ax0.set_yscale("log")

    title = f"Box Plot of Score Distribution `{'trans'}`  \n{args_dict.get('experiment_name')}\n{args_dict.get('output_dir_name')}"
    ax0.set_title(title)

    min_median = float("inf")
    for median_plot in box_plot_data["medians"]:
        median_val = median_plot.get_ydata()[0]
        min_median = min(min_median, median_val)

    plt.minorticks_on()
    plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
    plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)
    ax0.axhline(min_median, color="grey", alpha=0.5, linestyle="--")
    save_filepath = output_plot_dirpath / "loop_analysis_boxplots_scores_vs_iter_nb.png"
    plt.savefig(save_filepath, dpi=350)
    logger.info(f"Score distribution plot saved at {save_filepath}")

    return recom_df


def print_grad(grad, _input, _output):
    print("Gradient:", grad)


def load_model_and_datamodule(dir_config, yml_config_path):
    # Build saliency map of the model
    if dir_config.args_dict.get("trainer_version_number", None) is None:
        loop_config_filepath = dir_config.root_output_dir / "loop_config.json"
        trainer_version_number = json.load(open(loop_config_filepath, "r"))["trainer_version_number"]
        logger.info(f"Trainer version number has been set from loop config file to {trainer_version_number}")
    else:
        trainer_version_number = dir_config.args_dict.get("trainer_version_number", None)
        logger.info(f"Trainer version number has been set from comand line argument to {trainer_version_number}")

    model_config = ModelConfig(
        dir_config=dir_config, trainer_version_number=trainer_version_number, yml_config_path=Path(yml_config_path)
    )

    # Setup model
    model = FullTransformer(model_config=model_config)

    # Restore the model
    # checkpoint_path, ckpt_epoch = EncToScoreTrainer._find_best_checkpoint(dir_config, trainer_version_number, strategy="latest_iter_best_golden_acc")
    checkpoint_path, ckpt_epoch = EncToScoreTrainer._find_best_checkpoint(
        dir_config, trainer_version_number, strategy="best_mse_no_ssl"
    )
    lit_model = LitTransformer.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        meta_model=model,
        model_config=model_config,
        map_location="cpu",
    )

    # if False:
    lit_model.to(device=dir_config.args_dict.get("device", 2))

    dataset_config = DatasetConfig(args_dict=dir_config.args_dict)
    analyzer = setup_analyzer(**(dir_config.args_dict))

    datamodule = SwactedDesignDatamodule(dataset_config=dataset_config, analyzer=analyzer)
    datamodule.setup(stage="predict")

    return lit_model, datamodule, checkpoint_path


def get_classic_encoding_tensor(
    dir_config: ConfigDir,
):
    # Compare set of vectors with target set of vectors
    standard_enc_config_dict = DesignGenerator.generate_standard_encoding(dir_config.exp_config)
    tgt_vectors = from_binstr_list_to_int_array(list(standard_enc_config_dict["in_enc_dict"].values()))
    tgt_vectors = torch.tensor(tgt_vectors, dtype=torch.float32)

    return tgt_vectors


def make_saliency_map(
    lit_model: L.LightningModule,
    datamodule: L.LightningDataModule,
    checkpoint_path: Path,
    dir_config: ConfigDir,
    output_plot_dirpath: Path,
):
    # paths = []
    predictions = []
    encodings = []
    values = []
    expected_scores = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(datamodule.predict_dataloader()):
            _batch = {
                "encodings": batch["encodings"].to(lit_model.device),
                "values": batch["values"].to(lit_model.device),
                "scores": batch["scores"].to(lit_model.device),
            }
            y, y_expected = lit_model.predict_step(_batch)

            # Append tensors to lists
            expected_scores.append(y_expected.cpu())
            predictions.append(y.cpu())
            encodings.append(batch["encodings"].cpu())
            values.append(batch["values"].cpu())  # Rescale values for generating the correct encodings

            # keys = copy(list(_batch.keys()))
            # for key in keys:
            #     del _batch[key]
            # torch.cuda.empty_cache()
    expected_scores = torch.concatenate(expected_scores)
    predictions = torch.concatenate(predictions)
    encodings = torch.concatenate(encodings)
    values = torch.concatenate(values)

    val_loss = torch.nn.functional.mse_loss(predictions.squeeze(), expected_scores)
    logger.warning(f"Loaded model has a validation loss of {val_loss}")

    # Do saliency mapping
    test_values = torch.arange(0, values.shape[1]).unsqueeze(0).to(lit_model.device)

    saliency_maps = []
    fill_vals = [0.0, 0.5, 1.0]
    for fill_val in fill_vals:
        test_input = (
            torch.full(size=encodings[0].shape, fill_value=fill_val, requires_grad=True)
            .unsqueeze(0)
            .to(lit_model.device)
        )
        test_input = test_input.requires_grad_(True)
        test_input.retain_grad()
        # test_values = test_values.requires_grad_(True)

        lit_model.transformer.eval()
        assert not lit_model.transformer.training
        output = lit_model.transformer(test_input, test_values)
        output[0].backward(retain_graph=True)

        saliency_maps.append(test_input.grad[0].cpu())

    min_value_twos_comp = min_value_tc(int(dir_config.exp_config["input_bitwidth"]))
    assert min_value_twos_comp < 0
    test_values += min_value_twos_comp

    maxes = []
    mines = []
    for map in saliency_maps:
        maxes.append(torch.max(map))
        mines.append(torch.min(map))
    max_val = np.mean(maxes)
    min_val = np.mean(mines)

    yticks_pos = []
    ytick_labels = []
    for i, value in enumerate(test_values[0]):
        yticks_pos.append((i))
        ytick_labels.append(value.item())
        # ax.axvline((i*bitwidth+bitwidth), linewidth=2.5, color="white")
    plt.close()
    fig, axes = plt.subplots(1, 4, figsize=(15, 10))
    for idx, map in enumerate(saliency_maps):
        ax = axes[idx + 1]
        im = ax.imshow(map, vmin=min_val, vmax=max_val)
        ax.set_title(f"in_val:{fill_vals[idx]}")

        # ax.set_xticks([(i*bitwidth+bitwidth/2.0) for i in range(len(design_numbers))])
        # ax.set_xticklabels(design_numbers)

        # ax.axvline(0, linewidth=2.5, color="white")

        ax.set_yticks(yticks_pos)
        ax.set_yticklabels(ytick_labels)

    ax = axes[0]
    encoding_array = encodings[0].cpu().numpy()
    sns.heatmap(encoding_array, ax=ax, vmin=0, vmax=1, xticklabels=False, linewidths=0.1, linecolor="grey", cbar=False)
    ax.set_yticks([ytick_pos + 0.5 for ytick_pos in yticks_pos])
    ax.set_yticklabels(ytick_labels)
    trainer_version_number = extract_int_string_from_string(checkpoint_path.parent.name)
    fig.suptitle(
        f"Saliency Maps and Classic Encoding Visualization\ntrain_log_v:{trainer_version_number} | ckpt:{checkpoint_path.name}"
    )

    plt.tight_layout()
    fig.colorbar(im, ax=axes)
    golden_acc = float(checkpoint_path.name.split("_")[3])
    filepath = output_plot_dirpath / f"saliency_map_model_v{trainer_version_number}_golden_acc{golden_acc}.png"
    plt.savefig(filepath, dpi=200)
    logger.info(f"Saliency map figure saved at:")
    logger.info(filepath)


def pwd_pairwise_distance_loss(input_vectors: torch.Tensor, min_distance: float = 1.0):
    """
    Regularization loss that ensure that all vectors of the input are distant by at least `min_distance`.
    """
    batch_size, num_vectors, vector_dim = input_vectors.shape

    # Compute pairwise L2 distances between all vectors (NxN matrix)
    pairwise_distances = torch.cdist(input_vectors, input_vectors, p=2)

    # We want pairwise distances to be greater than a certain threshold (min_distance)
    distance_penalty = torch.nn.functional.relu(min_distance - pairwise_distances)

    # Create a mask to exclude diagonal elements (distances of vectors to themselves)
    eye_mask = torch.eye(num_vectors, device=input_vectors.device).bool()
    mask = eye_mask.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, num_vectors, num_vectors)

    # Sum up the penalties (only off-diagonal elements)
    distance_loss = torch.sum(distance_penalty.masked_fill(mask, 0), dim=(1, 2))  # Sum but ignore diagonal

    return distance_loss


def tgt_match_loss(input_vectors: torch.Tensor, target_vectors: torch.Tensor):
    # Compute the pairwise L2 distance between input vectors and target vectors
    pairwise_distances = torch.cdist(input_vectors, target_vectors, p=2)  # Shape: (N, N)

    # For each input vector, find the minimum distance to any target vector
    min_distances, _ = torch.min(pairwise_distances, dim=1)

    # Sum the minimum distances as the match loss
    return torch.sum(min_distances, dim=1)


def check_valid(input_vectors, target_vectors):
    pairwise_distances = torch.cdist(input_vectors, target_vectors, p=2)  # Shape: (N, N)

    min_distances, _ = torch.min(pairwise_distances, dim=1)


def attraction_repulsion_loss(input_vectors, target_vectors, epsilon=1e-4):
    """
    Ensures each predicted vector is close to some target while preventing collapse.
    """
    # Attraction: Pulls vectors to the closest target
    distances = torch.cdist(input_vectors, target_vectors, p=2)  # (4,4) matrix
    attraction_loss = (torch.min(distances, dim=1)[0]).mean(1)  # Min dist for each point

    # Repulsion: Prevents points from collapsing
    pairwise_dists = torch.cdist(input_vectors, input_vectors, p=2)
    repulsion_loss = torch.mean(1 / (pairwise_dists + epsilon), (1, 2))  # Inverse distance

    return attraction_loss + 0.001 * repulsion_loss  # Repulsion weighted down


class ScoredPerm:
    def __init__(self, score, perm):
        self.score = score  # a Python float
        self.perm = perm  # a torch.Tensor

    def __lt__(self, other):
        return self.score < other.score


def generate_prototype_patterns(
    lit_model: L.LightningModule,
    datamodule: L.LightningDataModule,
    checkpoint_path: Path,
    dir_config: ConfigDir,
    output_plot_dirpath: Path,
    data_save_path: Path,
    nb_gener_prototypes: int = 4096,
    **kwargs,
):
    """
    This function trains an input to minimize the output score of the network.
    Doing so enables to find an input image that corresponds to the ideal input patter for score minimization.

    Args:
        lit_model (L.LightningModule): _description_
        checkpoint_path (Path): _description_
        dir_config (ConfigDir): _description_
        output_plot_dirpath (Path): _description_
    """

    start_time = time()
    logger.info("Generating prototype pattern ...")

    # Setting up input preparation
    noise_scale = 1e-3
    data = next(iter(datamodule.predict_dataloader()))  # Get a sample of data to get the correct shape
    encodings = data["encodings"]
    values = data["values"]

    # Setting batching
    max_batch_size = dir_config.args_dict.get("batch_size")
    nb_gener_prototypes = nb_gener_prototypes
    steps_per_epoch = nb_gener_prototypes // max_batch_size + int(nb_gener_prototypes % max_batch_size > 0)
    logger.info(f"Input prototype generation will run with {nb_gener_prototypes} batches (steps) per epoch")

    # Prepare encoded values list (requried for embedding)
    test_values = torch.arange(0, values.shape[1]).unsqueeze(0).to(lit_model.device).requires_grad_(False)
    test_values = test_values.to(lit_model.device)

    # Prepare trainable inputs
    prototypes = (
        torch.full(size=encodings[0].shape, fill_value=0.5, requires_grad=True)
        .unsqueeze(0)
        .to(lit_model.device)
        .requires_grad_(True)
    )

    # Create duplicates by repeating the tensor
    prototypes = prototypes.repeat(nb_gener_prototypes, 1, 1)

    # Add random noise between -0.5 and 0.5 to each duplicate
    noise = (torch.rand_like(prototypes) - 0.5) * noise_scale
    prototypes = prototypes + noise

    prototypes = prototypes.to(lit_model.device)

    # Split the large input tensor into smaller batches
    prototypes = torch.unbind(prototypes, dim=0)
    _batched_prototypes = []
    for i in range(steps_per_epoch):
        # logger.info(min((i+1)*max_batch_size, nb_gener_prototypes))
        batch = prototypes[i * max_batch_size : min((i + 1) * max_batch_size, nb_gener_prototypes)]
        if len(batch) != 0:
            batch = torch.stack(batch, dim=0)
            batch.retain_grad()
            _batched_prototypes.append(batch)
    batched_prototypes = _batched_prototypes

    # prototypes.retain_grad()
    # batched_prototypes = [prototypes]

    assert len(batched_prototypes) == steps_per_epoch

    # Set up an optimizer (does Adam works well for this?)
    optimizer = torch.optim.Adam(batched_prototypes, lr=0.01)

    # Run model on input
    lit_model.transformer.eval()
    assert not lit_model.transformer.training

    # Add dropout
    # for m in lit_model.transformer.modules():
    #     if m.__class__.__name__.startswith('Dropout'):
    #         m.train()

    # Setup some loss parameters
    # loss_weight_pwd = 1.0
    # loss_weight_tgt = 1.0
    tgt_vectors = get_classic_encoding_tensor(dir_config).to(lit_model.device)

    # Setup the tensorboard writer
    timestamp = strftime("%Y-%m-%d_%H-%M", gmtime())
    tb_logs_dir = output_plot_dirpath / "tb_logs" / timestamp
    tb_logs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Tensorboard log diretory set at:")
    logger.info(f"{tb_logs_dir}")
    writer = SummaryWriter(log_dir=tb_logs_dir)
    logger.info(f"Saving tensorboard logs in: ")

    # Iterate over all epochs
    trainer_version_number = extract_int_string_from_string(checkpoint_path.parent.name)
    max_epochs = dir_config.args_dict.get("max_epochs", 10)

    all_outputs = []
    prototype_batch: torch.Tensor
    logger.info(f"Input prototype generation loop running for {max_epochs} epochs ...")
    with tqdm(total=steps_per_epoch, desc=f"Input prototype optimization") as pbar:  # Progress bar
        # Proced for all epochs
        for epoch_idx in range(max_epochs):
            # Loop over all samples of the batch
            for step_idx, prototype_batch in enumerate(batched_prototypes):  # Optimization happens inline
                # Forward pass through the model
                output, vae_output = lit_model.transformer(prototype_batch, test_values)
                if epoch_idx / max_epochs < 0.8:
                    # score_loss = ((0.8*max_epochs - epoch_idx) / (0.8*max_epochs) * 10 + 1) * output
                    score_loss = ((0.8 * max_epochs - epoch_idx) / (0.8 * max_epochs) * 30 + 1) * output
                else:
                    score_loss = output
                # att_rep_loss = attraction_repulsion_loss(prototype_batch, tgt_vectors) * 8
                att_rep_loss = attraction_repulsion_loss(prototype_batch, tgt_vectors) * 2

                if vae_output is None:
                    kl_loss = torch.zeros_like(score_loss)
                else:
                    kl_loss = 0.1 * kl_divergence_loss(vae_output)

                loss = (score_loss.flatten() + att_rep_loss.flatten() + kl_loss.flatten()).mean()

                # # The objective is to minimize the output score, so the loss is simply the output
                # if epoch_idx < max_epochs/2:
                #     loss = 8*output + loss_weight_pwd*pwd_pairwise_distance_loss(prototype_batch, min_distance=1.0)
                # else:
                #     loss = 5*output + loss_weight_pwd*pwd_pairwise_distance_loss(prototype_batch, min_distance=1.0) + loss_weight_tgt*tgt_match_loss(prototype_batch, tgt_vectors)
                #
                # # Backward pass
                # if loss.shape[0] > 1:
                #     loss = torch.sum(loss)

                # Accumulate gradients
                loss.backward()

                # # Clamp the input to keep it within a reasonable range
                # with torch.no_grad():
                #     # prototype_batch.clamp_(0.0, 1.0)
                #     prototype_batch.clamp_(0.0, 1.0)

                # print({
                #     "average_score":torch.mean(output).item(),
                #     "average_score_loss":torch.mean(score_loss).item(),
                #     "att_rep_loss":torch.mean(att_rep_loss).item(),
                #     "kl_loss":torch.mean(kl_loss).item(),
                #     "epoch":epoch_idx,
                # })

                # Do post-step processing
                info_dict = {
                    "average_score": torch.mean(output).item(),
                    "average_score_loss": torch.mean(score_loss).item(),
                    "att_rep_loss": torch.mean(att_rep_loss).item(),
                    "epoch": epoch_idx,
                }

                for key, value in info_dict.items():
                    if key != "epoch":
                        writer.add_scalar(f"loss/{key}", value, epoch_idx * steps_per_epoch + step_idx)
                pbar.set_postfix(info_dict)
                pbar.update()

                # Store final scores at last epoch
                if epoch_idx == (max_epochs - 1):
                    all_outputs.append(output.clone())

                # Optimizing the input data and each batch contains entirely different samples (with no overlap between batches), applying a single optimization step after processing all batches in one epoch
                optimizer.step()
                optimizer.zero_grad()

                if epoch_idx % 50 == 0:
                    # Draw one prototype to simplify manual verification of the input optimization process
                    cpu_prototype_input = torch.clone(prototype_batch[-1]).cpu().detach().numpy()
                    title = f"Input Prototype Visualization\ntrain_log_v:{trainer_version_number}|ckpt:{checkpoint_path.name}\niter:{epoch_idx} | score {output[-1].item():.3f}"
                    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
                    sns.heatmap(
                        cpu_prototype_input.squeeze(),
                        annot=True,
                        ax=axes,
                        yticklabels=(test_values[0] - 8).cpu().numpy(),
                    )
                    axes.set_xlabel("Bits")
                    axes.set_xlabel("Values")
                    fig.suptitle(title)
                    plt.tight_layout()
                    writer.add_figure("prototype_viz", plt.gcf(), epoch_idx * steps_per_epoch)
                    plt.close()

                # Reset progress bar
                pbar.reset()

    # Post process the data
    prototype_input = torch.cat(batched_prototypes, dim=0)
    prototype_input = prototype_input.detach()
    scores = torch.cat(all_outputs, dim=0)

    end_time = time()
    logger.info(f"Optimizing {nb_gener_prototypes} input prototypes has taken {(end_time - start_time) / 60 / 60:.2f}h")

    # Fix prototype
    if prototype_input.ndim == 2:
        prototype_input = prototype_input.unsqueeze(0)

    data, wrong_indexes = fix_prototypes(
        torch.tensor(prototype_input),
        scores=scores,
        tgt_vectors=tgt_vectors,
        round_threshold=1e-3,
        data_save_path=data_save_path,
    )

    return data


def any_duplicates(tensor: torch.Tensor) -> bool:
    """
    This function uses the cdist function to check if any two rows are duplicates of each other in a 2D tensor.
    """
    # Compute pairwise distances using cdist
    distances = torch.cdist(tensor, tensor, p=2)

    # A row is a duplicate if the distance to another row is zero (ignoring self-distance)
    duplicates = (distances == 0).float().triu(diagonal=1)

    # Get the indices of duplicate rows
    duplicate_indices = torch.nonzero(duplicates, as_tuple=False)

    return len(duplicate_indices) != 0


def fix_prototypes(
    prototypes: torch.Tensor,
    scores: torch.Tensor,
    tgt_vectors: torch.Tensor,
    data_save_path: Path,
    round_threshold: float = 1e-3,
    fix_strategy_version: int = 3,
) -> torch.Tensor | None:
    """
    Returns the fixed prototype encoding dictionnary.
    This function rounds the generated `prototype` 0.0 or 1.0 values for all values whose distance to either 0 or 1 is less than `threshold`.
    It then checks if there are any duplicates within these rounded vectors.
    If there are, it returns None.
    Otherwise, it then replaces the remaining (non entirely rounded) vectors with their closest vectors in `tgt_vectors`.

    Args:
        prototype (torch.Tensor): Prototype tensor with shape (batch_size, nb_values, bitwidth)
        tgt_vectors (torch.Tensor): _description_
        round_threshold (float, optional): _description_. Defaults to 1e-3.

    Returns:
        _output_raw_protos (torch.Tensor): CPU-based original prototypes associated (aligned) with the fixed prototypes
        _output_fixed_protos (torch.Tensor): CPU-based fixed prototypes
    """

    # torch.save(prototypes, 'prototypes.pth')
    # torch.save(scores, 'scores.pth')
    # torch.save(tgt_vectors, 'tgt_vectors.pth')

    # prototypes = torch.load('prototypes.pth')
    # scores = torch.load('scores.pth')
    # tgt_vectors = torch.load('tgt_vectors.pth')

    start_time = time()
    logger.info(f"Fixing generated prototypes with strategy version {fix_strategy_version}...")
    assert prototypes.ndim == 3

    raw_protos = torch.clone(prototypes)
    fixed_final_protos = torch.clone(prototypes)
    tgt_vectors = tgt_vectors.to(fixed_final_protos.device)

    # Convert all values close to zero to 0
    indices_0 = torch.nonzero(torch.abs(prototypes - 0) < round_threshold, as_tuple=True)
    fixed_final_protos[indices_0] = 0

    # Convert all values close to one to 1
    indices_1 = torch.nonzero(torch.abs(prototypes - 1) < round_threshold, as_tuple=True)
    fixed_final_protos[indices_1] = 1

    _output_raw_protos = []
    _output_fixed_protos = []
    _output_scores = []
    wrong_count = 0
    wrong_indexes = []
    with tqdm(total=len(fixed_final_protos), desc="Fixing prototypes") as pbar:
        for idx, fixed_final_proto in enumerate(fixed_final_protos):
            # Match returns a NxN vector
            matches = (fixed_final_proto[:, None, :] == tgt_vectors).all(-1)
            # Indices of vectors matching with the other vector
            # (matchings_rows_proto is the list of indices of proto vectors that match any tgt vector)
            # (matching_rows_tgt_vectors is the list of corresponding indices of matching vectors in tgt vectors)
            matching_rows_proto, matching_rows_tgt_vectors = torch.nonzero(matches, as_tuple=True)

            # From here, things cannot be batched anymore
            # Get list of missing matches

            missing_indexes_proto = list(set(range(len(fixed_final_proto))) - set(matching_rows_proto.tolist()))
            missing_indexes_tgt_vectors = list(set(range(len(tgt_vectors))) - set(matching_rows_tgt_vectors.tolist()))

            # Check duplicate vectors
            indexes, counts = torch.unique(matching_rows_tgt_vectors, return_counts=True)
            # logger.info(matching_rows_tgt_vectors)
            if counts.sum() > len(counts):
                # duplicates found (unused list): indexes[counts > 1]
                # _output_fixed_protos.append(None)
                # _output_fixed_protos.append(None)
                wrong_count += 0
                wrong_indexes.append(idx)
                # TODO deal with duplicates?
            else:
                # Here, we want to replace remaining prototype vectors with the closest target vector
                if fix_strategy_version == 1:
                    if len(missing_indexes_tgt_vectors) > 0:
                        # Measure the distance between each remaining vector and the remaining target vectors
                        distances = torch.cdist(
                            fixed_final_proto[missing_indexes_proto, :],
                            tgt_vectors[missing_indexes_tgt_vectors, :],
                            p=2,
                        )

                        # Rank the distances globally and by column
                        ranked_tensor_global = torch.argsort(torch.argsort(distances.flatten())).reshape(
                            distances.shape
                        )
                        ranked_tensor_bycolumns = torch.argsort(torch.argsort(distances, dim=0), dim=0)

                        # The vectors with lowest scores have higher priority
                        validated_proto_vectors_idx = []
                        while len(validated_proto_vectors_idx) < len(missing_indexes_proto):
                            # The global ranking defines which target vector will be used first
                            global_minimum_index = torch.argmin(ranked_tensor_global)
                            associated_tgt_index = global_minimum_index % distances.shape[1]

                            # The column ranking defines which proto vector will be associated with the selected target vector
                            associated_proto_index = torch.argmin(ranked_tensor_bycolumns[:, associated_tgt_index])

                            # Retrieve the corresponding vector indices
                            proto_vector_idx = missing_indexes_proto[associated_proto_index]
                            tgt_vector_idx = missing_indexes_tgt_vectors[associated_tgt_index]

                            # If the selected proto vector has already been updated, skip them
                            if proto_vector_idx not in validated_proto_vectors_idx:
                                fixed_final_proto[proto_vector_idx, :] = tgt_vectors[tgt_vector_idx, :]
                                validated_proto_vectors_idx.append(proto_vector_idx)

                            # Ensure that the set
                            ranked_tensor_global[associated_proto_index, associated_tgt_index] = (
                                ranked_tensor_global.nelement()
                            )
                            ranked_tensor_bycolumns[associated_proto_index, associated_tgt_index] = len(
                                ranked_tensor_global
                            )

                elif fix_strategy_version == 2:
                    while len(missing_indexes_tgt_vectors) > 0:
                        distances = torch.cdist(
                            fixed_final_proto[missing_indexes_proto, :],
                            tgt_vectors[missing_indexes_tgt_vectors, :],
                            p=2,
                        )
                        min_index = torch.argmin(distances)
                        proto_idx, tgt_idx = torch.unravel_index(
                            min_index, distances.shape
                        )  # TODO check if this is correct
                        fixed_final_proto[missing_indexes_proto[proto_idx]] = tgt_vectors[
                            missing_indexes_tgt_vectors[tgt_idx]
                        ]
                        missing_indexes_tgt_vectors.pop(tgt_idx)
                        missing_indexes_proto.pop(proto_idx)

                elif fix_strategy_version == 3:
                    # TODO: should we re-evaluate distance at each loop iteration?
                    distances = torch.cdist(
                        fixed_final_proto[missing_indexes_proto, :], tgt_vectors[missing_indexes_tgt_vectors, :], p=2
                    )
                    distances = torch.nn.functional.softmax(distances, dim=-1)
                    while len(missing_indexes_tgt_vectors) > 0:
                        min_index = torch.argmin(distances)
                        proto_idx, tgt_idx = torch.unravel_index(min_index, distances.shape)
                        fixed_final_proto[missing_indexes_proto[proto_idx]] = tgt_vectors[
                            missing_indexes_tgt_vectors[tgt_idx]
                        ]
                        missing_indexes_tgt_vectors.pop(tgt_idx)
                        missing_indexes_proto.pop(proto_idx)
                        tgt_mask = torch.arange(distances.size(-1), device=tgt_idx.device) != tgt_idx
                        proto_mask = torch.arange(distances.size(-2), device=tgt_idx.device) != proto_idx
                        distances = distances[proto_mask][:, tgt_mask]

                if not any_duplicates(fixed_final_proto):
                    _output_fixed_protos.append(fixed_final_proto.cpu())
                    _output_raw_protos.append(raw_protos[idx].cpu())
                    _output_scores.append(scores[idx].cpu())
                else:
                    wrong_count += 0
                    wrong_indexes.append(idx)
                    # logger.info("Fixed proto had a duplicate")
            pbar.update()

    logger.warning(f"{len(_output_fixed_protos)} protoypes have been fixed or validated.")
    if wrong_count > 0:
        logger.warning(f"{wrong_count} protoypes have been discarded.")

    end_time = time()
    logger.info(
        f"Fixing {len(_output_fixed_protos) + wrong_count} prototypes has taken {(end_time - start_time) / 60 / 60:.2f}h"
    )

    data = {
        "prototype": torch.stack(_output_raw_protos, dim=0),
        "fixed_prototype": torch.stack(_output_fixed_protos, dim=0),
        "score": torch.stack(_output_scores, dim=0),
    }

    # Save data
    timestamp = strftime("%Y-%m-%d_%H-%M", gmtime())
    _data_save_path = data_save_path / f"prototype_data_{len(_output_raw_protos)}_{timestamp}_.npz"
    try:
        save_serialized_data(_data_save_path, data)
        logger.info(f"All data saved at:")
        logger.info(f"{_data_save_path}")
    except Exception:
        logger.error(f"There was an error when saving the data. Prototype generation data has not been saved.")

    return data, wrong_indexes


def second_argmin(tensor):
    # Get the index of the minimum value
    min_index = tensor.argmin()

    # Mask the minimum value and find the index of the new minimum
    masked_tensor = tensor.clone()
    proto_idx, tgt_idx = torch.unravel_index(min_index, masked_tensor.shape)
    masked_tensor[proto_idx, tgt_idx] = float("inf")  # Set the minimum value to inf
    second_min_index = masked_tensor.argmin()

    return second_min_index, masked_tensor[second_min_index]


def new_fix_prototypes(
    prototypes: torch.Tensor,
    scores: torch.Tensor,
    tgt_vectors: torch.Tensor,
    data_save_path: Path,
    round_threshold: float = 1e-3,
    fix_strategy_version: int = 3,
) -> torch.Tensor | None:
    """
    Returns the fixed prototype encoding dictionnary.
    This function rounds the generated `prototype` 0.0 or 1.0 values for all values whose distance to either 0 or 1 is less than `threshold`.
    It then checks if there are any duplicates within these rounded vectors.
    If there are, it returns None.
    Otherwise, it then replaces the remaining (non entirely rounded) vectors with their closest vectors in `tgt_vectors`.

    Args:
        prototype (torch.Tensor): Prototype tensor with shape (batch_size, nb_values, bitwidth)
        tgt_vectors (torch.Tensor): _description_
        round_threshold (float, optional): _description_. Defaults to 1e-3.

    Returns:
        _output_raw_protos (torch.Tensor): CPU-based original prototypes associated (aligned) with the fixed prototypes
        _output_fixed_protos (torch.Tensor): CPU-based fixed prototypes
    """

    # torch.save(prototypes, 'prototypes.pth')
    # torch.save(scores, 'scores.pth')
    # torch.save(tgt_vectors, 'tgt_vectors.pth')

    # prototypes = torch.load('prototypes.pth')
    # scores = torch.load('scores.pth')
    # tgt_vectors = torch.load('tgt_vectors.pth')

    start_time = time()
    logger.info(f"Fixing generated prototypes with strategy version {fix_strategy_version}...")
    assert prototypes.ndim == 3

    raw_protos = torch.clone(prototypes)
    fixed_final_protos = torch.clone(prototypes)
    tgt_vectors = tgt_vectors.to(fixed_final_protos.device)

    # Convert all values close to zero to 0
    indices_0 = torch.nonzero(torch.abs(prototypes - 0) < round_threshold, as_tuple=True)
    fixed_final_protos[indices_0] = 0

    # Convert all values close to one to 1
    indices_1 = torch.nonzero(torch.abs(prototypes - 1) < round_threshold, as_tuple=True)
    fixed_final_protos[indices_1] = 1

    _output_raw_protos = []
    _output_fixed_protos = []
    _output_scores = []
    wrong_count = 0
    wrong_indexes = []

    good_bifurcations = {
        "0_0_1_0",
        "0_0_2_0",
        "0_1_0_1",
        "0_0_2_1",
        "0_0_0_0",
        "0_1_1_0",
        "0_2_2_1",
        "0_0_1_1",
        "0_2_1_0",
        "0_0_0_1",
        "1_0_0_0",
        "0_2_0_0",
        "0_1_2_0",
        "2_0_0_0",
        "0_1_0_0",
    }
    with tqdm(total=len(fixed_final_protos), desc="Fixing prototypes") as pbar:
        for idx, fixed_final_proto in enumerate(fixed_final_protos):
            # Match returns a NxN vector
            matches = (fixed_final_proto[:, None, :] == tgt_vectors).all(-1)
            # Indices of vectors matching with the other vector
            # (matchings_rows_proto is the list of indices of proto vectors that match any tgt vector)
            # (matching_rows_tgt_vectors is the list of corresponding indices of matching vectors in tgt vectors)
            matching_rows_proto, matching_rows_tgt_vectors = torch.nonzero(matches, as_tuple=True)

            # From here, things cannot be batched anymore
            # Get list of missing matches

            missing_indexes_proto = list(set(range(len(fixed_final_proto))) - set(matching_rows_proto.tolist()))
            missing_indexes_tgt_vectors = list(set(range(len(tgt_vectors))) - set(matching_rows_tgt_vectors.tolist()))

            # Check duplicate vectors
            indexes, counts = torch.unique(matching_rows_tgt_vectors, return_counts=True)
            # logger.info(matching_rows_tgt_vectors)
            if counts.sum() > len(counts):
                # duplicates found (unused list): indexes[counts > 1]
                # _output_fixed_protos.append(None)
                # _output_fixed_protos.append(None)
                wrong_count += 0
                wrong_indexes.append(idx)
                # TODO deal with duplicates?
            else:
                # New strat with multiple outputs
                # TODO: should we re-evaluate distance at each loop iteration?
                distances = torch.cdist(
                    fixed_final_proto[missing_indexes_proto, :], tgt_vectors[missing_indexes_tgt_vectors, :], p=2
                )
                distances = torch.nn.functional.softmax(distances, dim=-1)
                while len(missing_indexes_tgt_vectors) > 5:
                    min_index = torch.argmin(distances)
                    proto_idx, tgt_idx = torch.unravel_index(min_index, distances.shape)
                    fixed_final_proto[missing_indexes_proto[proto_idx]] = tgt_vectors[
                        missing_indexes_tgt_vectors[tgt_idx]
                    ]
                    missing_indexes_tgt_vectors.pop(tgt_idx)
                    missing_indexes_proto.pop(proto_idx)
                    tgt_mask = torch.arange(distances.size(-1), device=tgt_idx.device) != tgt_idx
                    proto_mask = torch.arange(distances.size(-2), device=tgt_idx.device) != proto_idx
                    distances = distances[proto_mask][:, tgt_mask]

                distances_copy = distances.clone()
                missing_indexes_proto_copy = missing_indexes_proto.copy()
                missing_indexes_tgt_vectors_copy = missing_indexes_tgt_vectors.copy()

                for index5 in range(3):
                    for index4 in range(3):
                        for index3 in range(3):
                            for index2 in range(2):
                                bifurcation = f"{index5}_{index4}_{index3}_{index2}"

                                if bifurcation not in good_bifurcations:
                                    continue

                                distances = distances_copy.clone()
                                missing_indexes_proto = missing_indexes_proto_copy.copy()
                                missing_indexes_tgt_vectors = missing_indexes_tgt_vectors_copy.copy()
                                while len(missing_indexes_tgt_vectors) > 0:
                                    min_index = torch.argmin(distances)
                                    proto_idx, tgt_idx = torch.unravel_index(min_index, distances.shape)
                                    if distances.shape[0] == 5:
                                        proto_idx = distances[:, tgt_idx].topk(index5 + 1, largest=False)[1][-1]
                                    elif distances.shape[0] == 4:
                                        proto_idx = distances[:, tgt_idx].topk(index4 + 1, largest=False)[1][-1]
                                    elif distances.shape[0] == 3:
                                        proto_idx = distances[:, tgt_idx].topk(index3 + 1, largest=False)[1][-1]
                                    elif distances.shape[0] == 2:
                                        proto_idx = distances[:, tgt_idx].topk(index2 + 1, largest=False)[1][-1]

                                    fixed_final_proto[missing_indexes_proto[proto_idx]] = tgt_vectors[
                                        missing_indexes_tgt_vectors[tgt_idx]
                                    ]
                                    missing_indexes_tgt_vectors.pop(tgt_idx)
                                    missing_indexes_proto.pop(proto_idx)
                                    tgt_mask = torch.arange(distances.size(-1), device=tgt_idx.device) != tgt_idx
                                    proto_mask = torch.arange(distances.size(-2), device=tgt_idx.device) != proto_idx
                                    distances = distances[proto_mask][:, tgt_mask]

                                if not any_duplicates(fixed_final_proto):
                                    _output_fixed_protos.append(fixed_final_proto.cpu())
                                    _output_raw_protos.append(raw_protos[idx].cpu())
                                    _output_scores.append(scores[idx].cpu())
                                else:
                                    wrong_count += 0
                                    wrong_indexes.append(idx)
                                    # logger.info("Fixed proto had a duplicate")
            pbar.update()

    logger.warning(f"{len(_output_fixed_protos)} protoypes have been fixed or validated.")
    if wrong_count > 0:
        logger.warning(f"{wrong_count} protoypes have been discarded.")

    end_time = time()
    logger.info(
        f"Fixing {len(_output_fixed_protos) + wrong_count} prototypes has taken {(end_time - start_time) / 60 / 60:.2f}h"
    )

    data = {
        "prototype": torch.stack(_output_raw_protos, dim=0),
        "fixed_prototype": torch.stack(_output_fixed_protos, dim=0),
        "score": torch.stack(_output_scores, dim=0),
    }

    # Save data
    timestamp = strftime("%Y-%m-%d_%H-%M", gmtime())
    _data_save_path = data_save_path / f"prototype_data_{len(_output_raw_protos)}_{timestamp}_.npz"
    try:
        save_serialized_data(_data_save_path, data)
        logger.info(f"All data saved at:")
        logger.info(f"{_data_save_path}")
    except Exception:
        logger.error(f"There was an error when saving the data. Prototype generation data has not been saved.")

    return data, wrong_indexes


def patch_prototypes(prototype_data_path: Path, dir_config: ConfigDir, fix_strategy_version: int = 3):
    if prototype_data_path.exists():
        protoype_pattern_data = load_serialized_data(prototype_data_path)
    else:
        raise ValueError(f"Provided prototype data path does not exists.")

    if protoype_pattern_data["prototype"].shape[0] == protoype_pattern_data["score"].shape[0]:
        pass
    else:
        pass

    # original_prototype_inputs = torch.stack(protoype_pattern_data["prototype"], dim=0)
    original_prototype_inputs = protoype_pattern_data["prototype"]
    data, wrong_indexes = fix_prototypes(
        original_prototype_inputs,
        protoype_pattern_data["score"],
        tgt_vectors=get_classic_encoding_tensor(dir_config),
        round_threshold=1e-3,
        fix_strategy_version=fix_strategy_version,
        data_save_path=dir_config.analysis_out_dir,
    )

    return data


def copy_designs_main(src_dir_config: ConfigDir) -> ConfigDir:
    """
    Copy some designs from one folder to another.
    Only designs that meet criterion requirements will be copied.
    The directory configurations must be instantiated beforehand.
    """

    if src_dir_config.args_dict.get("dst_output_dir_name", None) is None:
        raise ValueError(f"`do_copy_designs` was received but `dst_output_dir_name` was not specified.")
    # Instantiate the destination configuration directory
    dst_args_dict = copy(src_dir_config.args_dict)
    dst_args_dict.update({"output_dir_name": src_dir_config.args_dict["dst_output_dir_name"]})
    logger.info(f"Setting up destination output directory.")
    dst_dir_config = ConfigDir(is_analysis=False, **dst_args_dict)

    criterion_name = src_dir_config.args_dict.get("criterion_name", None)
    criterion_threshold = src_dir_config.args_dict.get("criterion_threshold", None)
    criterion_minimum = src_dir_config.args_dict.get("criterion_minimum", None)

    if criterion_threshold is None and criterion_minimum is None:
        logger.error(f"Either `criterion_threshold` or `criterion_threshold` should be specified")
        raise ValueError()
    elif (criterion_threshold is not None) and (criterion_minimum is not None):
        logger.error(f"Only one of `criterion_threshold` or `criterion_threshold` should be specified")
        raise ValueError()

    # Open synth DB from src directory
    synth_db_path = src_dir_config.analysis_out_dir / "synth_analysis.db.pqt"
    synth_df = load_database(synth_db_path)

    # Get best designs only
    if criterion_threshold is not None:
        logger.info(f"Filtering out the design based on criterion value")
        best_synth_df = synth_df[synth_df[criterion_name].astype(int) <= criterion_threshold]
    elif criterion_minimum is not None:
        logger.info(f"Filtering out the design based on criterion value")
        best_synth_df = synth_df[synth_df[criterion_name].astype(int) >= criterion_minimum]
    else:
        logger.info(f"Criterion value was not set, all designs will be copied")

    best_design_numbers_list = best_synth_df["design_number"].to_list()

    # Get the generation and synthesis directory paths
    logger.info(f"Building the full list of valid and gener design paths to be copied ...")
    to_copy_path_lists = {
        "gener": file_parsers.get_list_of_gener_designs_dirpath(
            src_dir_config, filter_design_numbers=best_design_numbers_list, filter_mode="include"
        ),
        "synth": file_parsers.get_list_of_synth_designs_dirpath(
            src_dir_config, filter_design_numbers=best_design_numbers_list, filter_mode="include"
        ),
    }
    dst_path_roots = {
        "gener": dst_dir_config.generation_out_dir,
        "synth": dst_dir_config.synth_out_dir,
    }

    # Copy all specified directories to the destination directory
    for dir_type in ["gener", "synth"]:
        logger.info(f"Copying {len(to_copy_path_lists[dir_type])} {dir_type} directories to")
        logger.info(f"{dst_path_roots[dir_type]}")
        for src_path in to_copy_path_lists[dir_type]:
            dst_path = dst_path_roots[dir_type] / src_path.name
            shutil.copytree(src_path, dst_path)
        logger.info(f"{dir_type.title()} copy done.")

    # Also copy the special_designs json file
    src_special_design_filepath = src_dir_config.special_designs_filepath
    dst_special_design_filepath = dst_dir_config.special_designs_filepath

    try:
        shutil.copyfile(src_special_design_filepath, dst_special_design_filepath)
    except FileNotFoundError:
        logger.warning(f"Special Design json file was not found in {src_dir_config.root_output_dir}")

    logger.info(f"All files have been copied to:")
    logger.info(dst_dir_config.root_output_dir)

    return dst_dir_config


def has_duplicate_batched(input_vectors: torch.Tensor):
    batch_size, num_vectors, _ = input_vectors.shape

    # Compute pairwise L2 distances between all vectors (NxN matrix)
    pairwise_distances = torch.cdist(input_vectors, input_vectors, p=2)

    # Create a mask to exclude diagonal elements (distances of vectors to themselves)
    eye_mask = torch.eye(num_vectors, device=input_vectors.device).bool()
    mask = eye_mask.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, num_vectors, num_vectors)

    # Set diagonal elememts to 1 to avoid triggering on "with-self" duplicates
    pairwise_distances = pairwise_distances.masked_fill(mask, 1.0)

    # Find batch indexes with at least one zero
    has_zero = (pairwise_distances == 0).any(dim=(1, 2))
    batch_indices_with_zero = torch.nonzero(has_zero, as_tuple=False).squeeze()

    return batch_indices_with_zero, has_zero


def get_all_encodings_and_duplicates(dir_config: ConfigDir) -> tuple[pd.DataFrame]:
    logger.info(f"Extracting all encodings to find any duplicate ...")

    encoding_dicts_df = file_parsers.read_all_existing_encodings_v2(
        dir_config.root_output_dir, dir_config.bulk_flow_dirname
    )

    logger.info(f"Check all encodings ...")

    with tqdm(
        total=len(encoding_dicts_df), desc=f"x256|Check duplicates and create index dataframe"
    ) as pbar:  # Progress bar
        duplicate_rows = process_pool_helper(
            func=_any_duplicate,
            func_args_gen=((row,) for idx, row in encoding_dicts_df.iterrows()),
            max_workers=global_vars.get("nb_workers"),
            pbar=pbar,
        )

    logger.info(f"Assembling duplicate index ...")
    duplicates_df = pd.concat(duplicate_rows, ignore_index=True)
    duplicates_df = duplicates_df.set_index("idx")
    duplicates_df.index.name = None

    return encoding_dicts_df, duplicates_df


def _remove_design_numbers_from_db(filepath: Path, design_numbers: set[str]):
    for design_number in design_numbers:
        assert isinstance(design_number, str)

    if filepath.exists():
        df = load_database(filepath)
    else:
        logger.info(f"Database does not exists: {filepath}")
        logger.info(f"It was skipped.")
        return None

    df_filtered = df[~df["design_number"].isin(design_numbers)]
    df_filtered.to_parquet(filepath)
    logger.info(f"Database has been overwritten: {filepath}")

    return None


def _remove_design_numbers_from_split_dict(filepath: Path, design_numbers: set[str]):
    for design_number in design_numbers:
        assert isinstance(design_number, str)

    if not filepath.exists():
        logger.warning(f"json split dictionnary does not exists: {filepath}")
        logger.info(f"It was skipped.")
        return None

    train_split_dict = json.load(open(filepath, "r"))

    new_split_dict = dict()
    for key, design_number_list in train_split_dict.items():
        new_split_dict[key] = list(set(design_number_list) - set(design_numbers))

    json.dump(new_split_dict, fp=open(filepath, "w"), indent=4)
    logger.info(f"Split dictionnary has been overwritten: {filepath}")


def delete_designs(dir_config: ConfigDir, design_numbers: set[str]) -> None:
    for design_number in design_numbers:
        assert isinstance(design_number, str)

        # Delete generated design
        dirpath = dir_config.generation_out_dir / f"res_{design_number}"
        if dirpath.exists():
            shutil.rmtree(dirpath)

        # Delete synthed design
        dirpath = dir_config.synth_out_dir / f"res_{design_number}"
        if dirpath.exists():
            shutil.rmtree(dirpath)

        # Delete swacted design
        dirpath = dir_config.swact_out_dir / f"res_{design_number}"
        if dirpath.exists():
            shutil.rmtree(dirpath)

        logger.info(f"Erased all `res_` directories for design {design_number}")

    # Delete design in synth_df
    filepath = dir_config.analysis_out_dir / "synth_analysis.db.pqt"
    _remove_design_numbers_from_db(filepath, design_numbers)

    # Delete design in swact_df
    filepath = dir_config.analysis_out_dir / "swact_analysis.db.pqt"
    _remove_design_numbers_from_db(filepath, design_numbers)

    # Delete design in encodings_dicts.db
    filepath = dir_config.root_output_dir / "encodings_dicts.db.pqt"
    _remove_design_numbers_from_db(filepath, design_numbers)

    # Delete design in valid_designs.db
    filepath = dir_config.root_output_dir / "valid_designs.db.pqt"
    _remove_design_numbers_from_db(filepath, design_numbers)

    # Delete in dataset_split.json
    filepath = dir_config.trainer_out_root_dir / "dataset_split.json"
    _remove_design_numbers_from_split_dict(filepath, design_numbers)


def do_purge_designs_main(dir_config: ConfigDir):
    """
    This function find and remove designs with duplicate encodings from the entire output directory.
    Those designs are removed in the files and all (TODO:check this) databases where they might be referenced.
    WARNING: this function actually deletes files, so make sure you know what you are doing before using it.
    """

    encoding_dicts_df, duplicates_df = get_all_encodings_and_duplicates(dir_config)
    to_delete_design_numbers = set(duplicates_df[duplicates_df["has_duplicates"]]["design_number"])

    # Write down list of designs that have been deleted
    with open(dir_config.root_output_dir / "deleted_design_numbers.txt", "a") as f:
        f.write(f"timestamp:{strftime('%Y-%m-%d_%H-%M', gmtime())}\n")
        f.write("\n".join(to_delete_design_numbers))

    logger.info(f"the following file has been updated with list of design numbers to delete:")
    logger.info(dir_config.root_output_dir / "deleted_design_numbers.txt")

    # Actually delete the designs
    percentage = (
        0.0
        if len(to_delete_design_numbers) == 0
        else len(encoding_dicts_df) * 1.0 / len(to_delete_design_numbers) * 100
    )
    logger.warning(
        f"This operation is going to erase {len(to_delete_design_numbers)} designs from the {len(encoding_dicts_df)} existing designs ({percentage:.3f}%)"
    )
    # logger.warning("Do you want to continue?")
    # validation = input("[Y] or [N]")
    # if validation == "Y":
    delete_designs(dir_config=dir_config, design_numbers=to_delete_design_numbers)

    return to_delete_design_numbers


def do_score_iter_plot_main(dir_config: ConfigDir, output_plot_dirpath: Path):
    # Open Analysis databases
    synth_db_filepath: Path = dir_config.analysis_out_dir / "synth_analysis.db.pqt"
    synth_df = load_database(synth_db_filepath)

    # Open recommendation database
    recom_db_filepath: Path = dir_config.root_output_dir / "recommendation.db.pqt"
    if recom_db_filepath.exists():
        recom_df = load_database(recom_db_filepath)
    else:
        recom_df = pd.DataFrame()
        logger.warning(f"Recommendation db has not been found.")
        logger.info(f"Score versus loop iteration plotting skipped.")
        return None

    # Open training database
    train_db_filepath: Path = dir_config.root_output_dir / "training.db.pqt"
    if train_db_filepath.exists():
        train_df = load_database(train_db_filepath)
    else:
        train_df = pd.DataFrame()
        logger.warning(f"Training db has not been found.")

    # Build databases dict
    db_dict = {
        "recom": recom_df,
        "synth": synth_df,
        "train": train_df,
    }

    recom_df = make_score_distribution_plot(
        db_dict=db_dict,
        output_plot_dirpath=output_plot_dirpath,
        args_dict=dir_config.args_dict,
    )


def do_prototype_generation_main(dir_config: ConfigDir, model_workers_kwargs: dict[Any]) -> list[Path]:
    data = next(
        iter(model_workers_kwargs["datamodule"].predict_dataloader())
    )  # Get a sample of data to get the correct shape
    values = data["values"]

    lit_model = model_workers_kwargs["lit_model"]
    lit_model.eval()

    tc_enc = {
        -8: "1000",
        -7: "1001",
        -6: "1010",
        -5: "1011",
        -4: "1100",
        -3: "1101",
        -2: "1110",
        -1: "1111",
        0: "0000",
        1: "0001",
        2: "0010",
        3: "0011",
        4: "0100",
        5: "0101",
        6: "0110",
        7: "0111",
    }
    tc_enc_tensor = enc_dict_to_tensor(tc_enc)

    # Prepare encoded values list (requried for embedding)
    test_values = torch.arange(0, values.shape[1]).unsqueeze(0).to(lit_model.device).requires_grad_(False)
    test_values = test_values.to(lit_model.device)

    n_samples = 5000

    top_k = 10_000

    top_permutation = []

    time_check = time()

    iter_count = 0

    while True:
        perms = [torch.randperm(16, dtype=torch.int32) for _ in range(n_samples)]
        batch = torch.stack([tc_enc_tensor[p] for p in perms])
        batch = batch.to(lit_model.device)

        score_pred = lit_model.transformer(batch, test_values)[0]

        scores = score_pred.squeeze().detach().cpu().numpy()

        for j, p in enumerate(perms):
            score_perm_temp = ScoredPerm(-float(scores[j]), p)
            if len(top_permutation) < top_k:
                heapq.heappush(top_permutation, score_perm_temp)
            else:
                heapq.heappushpop(top_permutation, score_perm_temp)

        del batch, score_pred

        torch.cuda.empty_cache()

        iter_count += 1

        if iter_count % 100 == 0:
            logger.info(f"Iter {iter_count} - Time {time() - time_check:.3f}")

        if time() - time_check > 60 * 25:
            break

    tc_enc_list = list(tc_enc.values())

    encodings_input_list = [
        {j: tc_enc_list[score_perm.perm[i]] for i, j in enumerate(range(-8, 8))} for score_perm in top_permutation
    ]

    # Use a template for the output encoding.
    encodings_output = {
        -56: "11001000",
        -49: "11001111",
        -48: "11010000",
        -42: "11010110",
        -40: "11011000",
        -36: "11011100",
        -35: "11011101",
        -32: "11100000",
        -30: "11100010",
        -28: "11100100",
        -25: "11100111",
        -24: "11101000",
        -21: "11101011",
        -20: "11101100",
        -18: "11101110",
        -16: "11110000",
        -15: "11110001",
        -14: "11110010",
        -12: "11110100",
        -10: "11110110",
        -9: "11110111",
        -8: "11111000",
        -7: "11111001",
        -6: "11111010",
        -5: "11111011",
        -4: "11111100",
        -3: "11111101",
        -2: "11111110",
        -1: "11111111",
        0: "00000000",
        1: "00000001",
        2: "00000010",
        3: "00000011",
        4: "00000100",
        5: "00000101",
        6: "00000110",
        7: "00000111",
        8: "00001000",
        9: "00001001",
        10: "00001010",
        12: "00001100",
        14: "00001110",
        15: "00001111",
        16: "00010000",
        18: "00010010",
        20: "00010100",
        21: "00010101",
        24: "00011000",
        25: "00011001",
        28: "00011100",
        30: "00011110",
        32: "00100000",
        35: "00100011",
        36: "00100100",
        40: "00101000",
        42: "00101010",
        48: "00110000",
        49: "00110001",
        56: "00111000",
        64: "01000000",
    }

    # Store the input and output encodings in a list.
    encoding_dicts_list = [{"in_enc_dict": d, "out_enc_dict": encodings_output} for d in encodings_input_list]

    # Remove duplicates
    encodings_df = pd.DataFrame([{"proto_str": str(p)} for p in encoding_dicts_list])
    index_to_keep = set(encodings_df.drop_duplicates().index.tolist())
    encoding_dicts_list = [p for i, p in enumerate(encoding_dicts_list) if i in index_to_keep]

    # Setup Generation
    gener_args_dict = dir_config.args_dict

    if dir_config.args_dict.get("dst_output_dir_name") is not None:
        # Update args dict for generation. Prototype data and generated design will be stored there.
        gener_args_dict.update({"output_dir_name": dir_config.args_dict.get("dst_output_dir_name")})

    # Setup Generation
    gener_dir_config = ConfigDir(is_analysis=False, **gener_args_dict)

    # Find which design numbers to use
    nb_protos = len(encoding_dicts_list)
    design_numbers = file_parsers.get_list_of_gener_designs_number(gener_dir_config)
    design_numbers_int = [int(design_number) for design_number in design_numbers]

    if len(design_numbers_int) == 0:
        max_known_design_number = -1
    else:
        max_known_design_number = max(design_numbers_int)
    new_design_numbers = [str(i) for i in range(max_known_design_number + 1, max_known_design_number + 1 + nb_protos)]

    # Actally generate the prototypes
    design_generator = DesignGenerator(gener_dir_config)
    logger.info(f"Generating new designs, starting with design_number {max_known_design_number + 1}")
    generated_design_paths_list, generated_design_config_dicts_list = design_generator.generate_all_designs(
        nb_to_generate=nb_protos,
        design_numbers=new_design_numbers,
        output_dir_path=gener_dir_config.generation_out_dir,
        existing_encodings_dicts=encoding_dicts_list,
    )

    # start_time = time()
    # # try:
    # # Do input prototype generation
    # gener_args_dict = dir_config.args_dict
    #
    # if dir_config.args_dict.get("dst_output_dir_name") is not None:
    #     # Update args dict for generation. Prototype data and generated design will be stored there.
    #     gener_args_dict.update({"output_dir_name": dir_config.args_dict.get("dst_output_dir_name")})
    #
    # # Setup Generation
    # gener_dir_config = ConfigDir(is_analysis=False, **gener_args_dict)
    # logger.warning(f"All prototype related generation files will be saved in the following root output directory:")
    # logger.warning(gener_dir_config.root_output_dir)
    #
    # if dir_config.args_dict.get("prototype_data_path", None) is None:
    #     # No path to prototypes provided: we need to generate the prototypes
    #     protoype_pattern_data = generate_prototype_patterns(
    #         **model_workers_kwargs,
    #         nb_gener_prototypes=dir_config.args_dict.get("nb_gener_prototypes", False),
    #         data_save_path=gener_dir_config.analysis_out_dir,
    #     )
    # else:
    #     # Path to prototypes was provided: we want to fix the generated prototypes again
    #     logger.info(f"Prototypes for generating designs will be loaded from the data file specified ...")
    #     prototype_data_path = Path(dir_config.args_dict.get("prototype_data_path", None))
    #     protoype_pattern_data = patch_prototypes(prototype_data_path, dir_config=gener_dir_config)

    # # From tensors to dictionnary
    # proto_encoding_dicts_list = convert_proto_to_encoding_dictionary(
    #     dir_config=dir_config, fixed_prototypes_batched=protoype_pattern_data["fixed_prototype"]
    # )

    # # Remove duplicates
    # prototype_df = pd.DataFrame([{"proto_str": str(p)} for p in proto_encoding_dicts_list])
    # index_to_keep = set(prototype_df.drop_duplicates().index.tolist())
    # proto_encoding_dicts_list = [p for i, p in enumerate(proto_encoding_dicts_list) if i in index_to_keep]
    #
    # # Find which design numbers to use
    # nb_protos = len(proto_encoding_dicts_list)
    # design_numbers = file_parsers.get_list_of_gener_designs_number(gener_dir_config)
    # design_numbers_int = [int(design_number) for design_number in design_numbers]
    #
    # if len(design_numbers_int) == 0:
    #     max_known_design_number = -1
    # else:
    #     max_known_design_number = max(design_numbers_int)
    # new_design_numbers = [str(i) for i in range(max_known_design_number + 1, max_known_design_number + 1 + nb_protos)]
    #
    # # Actally generate the prototypes
    # design_generator = DesignGenerator(gener_dir_config)
    # logger.info(f"Generating new designs, starting with design_number {max_known_design_number + 1}")
    # generated_design_paths_list, generated_design_config_dicts_list = design_generator.generate_all_designs(
    #     nb_to_generate=nb_protos,
    #     design_numbers=new_design_numbers,
    #     output_dir_path=gener_dir_config.generation_out_dir,
    #     existing_encodings_dicts=proto_encoding_dicts_list,
    # )
    #
    # status = "Success"
    # error_msg = ""
    #
    # # except:
    # #     status="Failed"
    # #     error_msg = traceback.format_exc()
    # #     generated_design_paths_list = None
    # #
    # send_email(
    #     config_dict=dir_config.args_dict,
    #     start_time=start_time,
    #     status=status,
    #     error_message=error_msg,
    #     calling_module="ProtoGen",
    #     root_output_dir=gener_dir_config.root_output_dir,
    # )

    return generated_design_paths_list


def load_loop_config(dir_config):
    # Read loop configuration
    loop_config_filepath = dir_config.root_output_dir / "loop_config.json"
    return json.load(open(loop_config_filepath, "r"))


def parse_args() -> dict[str, Any]:
    # Add arguments for this script
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--do_score_iter_plot", action="store_true", help="Plot the box plot of score versus iteration number"
    )
    arg_parser.add_argument(
        "--do_model_analysis", action="store_true", help="Plot the model analysis plots (i.e. semantic maps, ...)"
    )
    arg_parser.add_argument(
        "--do_prototype_pattern_gen",
        action="store_true",
        help="Train an input to minimze the model's output score, and return the optimum input.",
    )
    arg_parser.add_argument(
        "--do_copy_designs", action="store_true", help="Copy the designs to the specify destination directory."
    )
    arg_parser.add_argument(
        "--do_purge_designs",
        action="store_true",
        help="Delete designs that do not comply with any correctness check. Checks include duplicated encoding.",
    )

    # Will be used for any kind of generation
    arg_parser.add_argument(
        "--dst_output_dir_name",
        type=str,
        default=None,
        help="If set, the designs will be copied to this output directory. A standard directory structure will be setup by instantiating a directory configuration object first.",
    )

    # For prototype pattern generation
    arg_parser.add_argument("--trainer_version_number", type=int, help="Model version to load for analysis")
    arg_parser.add_argument(
        "--model_checkpoint_path", type=int, help="Direct path to the model checkpoint to load for model analysis."
    )
    arg_parser.add_argument("--yml_config_path", type=str, default=None, help="Specify which configuration to load.")
    arg_parser.add_argument(
        "--nb_gener_prototypes", type=int, help="Number of new input encoding prototypes to generate."
    )
    arg_parser.add_argument("--device", type=int, help="Device to which send the model.")
    arg_parser.add_argument("--batch_size", type=int, default=8092, help="Batch size for prototype generation.")
    arg_parser.add_argument(
        "--max_epochs", type=int, default=2000, help="Number of epochs to run input prototype generation for."
    )
    arg_parser.add_argument(
        "--score_type",
        type=str,
        default="trans",
        help="Defines the type of score used for training, which impacts the number of scores of the model (and thus the number of output heads).",
        choices=ScoreComputeHelper.transformer_map.keys(),
    )

    # For prototype pattern fixing
    arg_parser.add_argument(
        "--prototype_data_path",
        type=str,
        help="Path to the prototype data to resume design generation from. Might be helpful for fixing again prototypes that were not correctly fixed. If set, prototype generation is skipped and prototypes are loaded from the .npz file directly.",
    )

    # For copying designs
    arg_parser.add_argument(
        "--criterion_name",
        type=str,
        default="nb_transistors",
        help="Specifies the criterion name to use for deciding which designs to consider best designs.",
    )
    arg_parser.add_argument(
        "--criterion_threshold",
        type=float,
        default=None,
        help="Specifies the criterion threshold value to use for deciding which designs to consider best designs.",
    )
    arg_parser.add_argument(
        "--criterion_minimum",
        type=float,
        default=None,
        help="Specifies the criterion threshold value to use for deciding which designs to consider best designs.",
    )

    args_dict = vars(arg_parser.parse_known_args()[0])

    # Add standard analyzer arguments
    default_args_dict = analyzer_parser()
    args_dict.update(default_args_dict)

    return args_dict


def send_email(
    config_dict: dict[str:Any],
    start_time: float,
    status: str,
    error_message: str,
    calling_module: str,
    root_output_dir: Path,
):
    def define_log_file_name():
        src_dir = Path(os.getenv("SRC_DIR"))
        emaillogs_dir = src_dir / "z_email_logs" / calling_module
        if not emaillogs_dir.exists():
            emaillogs_dir.mkdir(exist_ok=True, parents=True)

        version_number = 0
        for file in emaillogs_dir.iterdir():
            if file.is_file():
                version_number = max(int(extract_int_string_from_string(file.name)), version_number)
        version_number += 1

        return version_number, emaillogs_dir / f"log_v{version_number}.log"

    if config_dict.get("send_email", False):
        end_time = time()
        duration = end_time - start_time
        formatted_duration = strftime("%H:%M:%S", gmtime(duration))

        command = " ".join(sys.argv)

        # Prepare email body
        email_body = textwrap.dedent(f"""
        
        Job Information:

        - **Proto**
        - **Start Date**: {strftime("%Y-%m-%d %H:%M:%S", localtime(start_time))}
        - **Duration**: {formatted_duration}
        - **Status**: {status}
        - **Error**: {error_message}
        - **Files**: {root_output_dir.relative_to(Path(os.getenv("WORK_DIR")))}
        - **Command Line**: {command}
        
        """)

        # Store e-mail message in a file
        log_version, log_filepath = define_log_file_name()
        log_filepath.write_text(email_body)

        # Prepare e-mail subject
        email_subject = f"{calling_module.title()} | log v{log_version} | {status}"

        email_body_short = textwrap.dedent(f"""
        
        Job Information:

        - **Proto**
        - **Start Date**: {strftime("%Y-%m-%d %H:%M:%S", localtime(start_time))}
        - **Duration**: {formatted_duration}
        - **Status**: {status}
        - **More Info in Log Version**: {log_version}
        
        """)

        _send_email(email_subject, body=email_body_short, calling_function=calling_module)
    else:
        logger.warning(f"End job e-mail was not sent because argument `--send_email` was not set.")


def main():
    args_dict = parse_args()
    dir_config = ConfigDir(is_analysis=True, **args_dict)

    # Setup output plot directory
    output_plot_dirpath = dir_config.analysis_out_dir / "plots_loop_analysis"
    if not output_plot_dirpath.exists():
        if output_plot_dirpath.parent.exists():
            output_plot_dirpath.mkdir(exist_ok=True)

    # Plot the box plot of score distribution
    if args_dict.get("do_score_iter_plot", False):
        do_score_iter_plot_main(dir_config=dir_config, output_plot_dirpath=output_plot_dirpath)
    else:
        logger.info(f"Score versus loop iteration plotting skipped.")

    # Check if a model should be loaded
    do_load_model = args_dict.get("do_model_analysis", False) or (
        args_dict.get("do_prototype_pattern_gen", False) and args_dict.get("prototype_data_path", None) is None
    )

    # Load the model and setup the kwargs associated for model related jobs
    if do_load_model:
        yml_config_path = args_dict.get("yml_config_path", None)
        if yml_config_path is None:
            raise ValueError("Specifying the yaml config file is now required!")
        logger.info("Loading model for analysis ... ")
        lit_model, datamodule, checkpoint_path = load_model_and_datamodule(dir_config, yml_config_path)
        model_workers_kwargs = dict(
            lit_model=lit_model,
            datamodule=datamodule,
            checkpoint_path=checkpoint_path,
            dir_config=dir_config,
            output_plot_dirpath=output_plot_dirpath,
        )
    else:
        model_workers_kwargs = dict()

    if args_dict.get("do_model_analysis", False):
        # Generate the saliency maps
        make_saliency_map(**model_workers_kwargs)

    if args_dict.get("do_prototype_pattern_gen", False):
        do_prototype_generation_main(dir_config=dir_config, model_workers_kwargs=model_workers_kwargs)

    if args_dict.get("do_copy_designs"):
        # Copy the  designs in the destination output directory
        copy_designs_main(dir_config)

    if args_dict.get("do_purge_designs", False):
        do_purge_designs_main(dir_config=dir_config)


if __name__ == "__main__":
    # sys.argv = ["python", "prototype_generator_v2.py", "--experiment_name", "multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only", "--output_dir_name", "loop_v2", "--synth_only", "--dst_output_dir_name", "best_prototypes", "--do_prototype_pattern_gen", "--trainer_version_number", "86", "--device", "3"]

    main()
