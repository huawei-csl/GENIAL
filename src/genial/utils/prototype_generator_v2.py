# %%
#
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
from genial.training.elements.score_tools import ScoreComputeHelper
from genial.utils.prototype_utils import convert_proto_to_encoding_dictionary
from tqdm import tqdm

from loguru import logger

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

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

from genial.utils.utils import (
    load_serialized_data,
    from_binstr_list_to_int_array,
    load_database,
    _any_duplicate,
    enc_dict_to_tensor,
)
from genial.utils.utils import send_email as _send_email
import textwrap

from torch.utils.tensorboard import SummaryWriter

from genial.utils.utils import extract_int_string_from_string, get_twos_complement_tensor
from genial.globals import global_vars


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


def load_model_and_datamodule(dir_config, yml_config_path, return_trainer_version_number: bool = False):
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

    returned = (lit_model, datamodule, checkpoint_path)
    if return_trainer_version_number:
        returned = (*returned, trainer_version_number)
    return returned


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
    trainer_version_number: int,
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
    # trainer_version_number extracted above if needed
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

    print("t")
    # attraction_loss = (torch.min(distances, dim=1)[0]).mean(1)  # Min dist for each point

    temperature = 0.1
    attraction_loss = -torch.logsumexp(-distances / temperature, dim=1).mean(1)

    # Repulsion: Prevents points from collapsing
    pairwise_dists = torch.cdist(input_vectors, input_vectors, p=2)
    repulsion_loss = torch.mean(1 / (pairwise_dists + epsilon), (1, 2))  # Inverse distance

    print(f"attraction_loss - {attraction_loss}")
    print(f"repulsion_loss - {repulsion_loss}")

    return attraction_loss + 0.001 * repulsion_loss  # Repulsion weighted down


def soft_alignment_loss(prototype_batch_norm, targets, temperature=0.1):
    sim_matrix = torch.matmul(prototype_batch_norm, targets.T)  # (bs, 16, 16)

    # Softmax over target dimension (last axis)
    soft_assignments = torch.nn.functional.softmax(sim_matrix / temperature, dim=-1)  # (bs, 16, 16)

    # Expected similarity: sum over target dimension
    expected_sim = torch.sum(soft_assignments * sim_matrix, dim=-1)  # (bs, 16)

    # We want to maximize similarity → minimize negative similarity
    loss = -expected_sim.mean(-1)  # scalar

    return loss


def sinkhorn(log_alpha, n_iter=10, temperature=1.0):
    log_alpha = log_alpha / temperature
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return log_alpha.exp()


def diversity_loss(prototype_sm, exp_param=1.0):
    # Derive hard assignments
    with torch.no_grad():
        hard = torch.zeros_like(prototype_sm)
        hard.scatter_(2, prototype_sm.argmax(dim=2, keepdim=True), 1.0)  # row-wise argmax

    # Straight-through estimator
    prototype_ste = prototype_sm + (hard - prototype_sm).detach()

    # Extract dim
    bs, design_len = prototype_sm.shape[:2]
    # Flatten
    prototype_ste_flat = prototype_ste.view(bs, -1).float()
    # Derive collision matrix
    collision_matrix = prototype_ste_flat @ prototype_ste_flat.T
    # Normalise between 0 and 1
    collision_matrix_norm = collision_matrix / design_len
    if exp_param < 1:  # Needed to avoid floating-point errors
        collision_matrix_norm = collision_matrix_norm + 1e-8
    # Control the weighting
    collision_matrix_final = collision_matrix_norm**exp_param
    # Mask the diagonal
    mask = ~torch.eye(bs, dtype=torch.bool, device=prototype_sm.device)
    # Return the mean collision score
    return collision_matrix_final[mask].mean()


def generate_prototype_patterns(
    lit_model: L.LightningModule,
    datamodule: L.LightningDataModule,
    checkpoint_path: Path,
    dir_config: ConfigDir,
    output_plot_dirpath: Path,
    data_save_path: Path,
    nb_gener_prototypes: int = 4096,
    keep_percentage: float = 0.25,
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

    logger.info("Generating prototype pattern ...")

    # Run model on input
    lit_model.transformer.eval()
    assert not lit_model.transformer.training

    # Setting up input preparation
    data = next(iter(datamodule.predict_dataloader()))  # Get a sample of data to get the correct shape
    values = data["values"]

    # Normalize keep percentage (accept 0-1 or 0-100)
    if keep_percentage > 1.0:
        keep_percentage = keep_percentage / 100.0
    keep_percentage = float(max(min(keep_percentage, 1.0), 1e-6))

    # Oversample, then we will keep a fraction later
    target_nb = nb_gener_prototypes
    nb_gener_prototypes = nb_gener_prototypes * int(1 / keep_percentage)

    # Setting batching
    max_batch_size = dir_config.args_dict.get("batch_size")
    nb_gener_prototypes = nb_gener_prototypes
    steps_per_epoch = nb_gener_prototypes // max_batch_size + int(nb_gener_prototypes % max_batch_size > 0)
    logger.info(f"Input prototype generation will run with {nb_gener_prototypes} batches (steps) per epoch")

    # Prepare encoded values list (requried for embedding)
    test_values = torch.arange(0, values.shape[1]).unsqueeze(0).to(lit_model.device).requires_grad_(False)
    test_values = test_values.to(lit_model.device)

    n_vectors = lit_model.transformer.embedding.encoding_embeddings.weight.shape[0]
    tgt_vectors = torch.cat(
        [
            lit_model.transformer.embedding.encoding_embeddings(torch.Tensor([i]).to(torch.int32).to(lit_model.device))
            for i in range(n_vectors)
        ]
    )

    # L2 normalise
    tgt_vectors_norm = torch.nn.functional.normalize(tgt_vectors, p=2, dim=-1).detach()
    # No gradient tracking needed
    tgt_vectors_norm.requires_grad = False

    # Derive design length
    design_len = len(test_values.flatten())

    prototypes = torch.empty((nb_gener_prototypes, design_len, design_len), requires_grad=True, device=lit_model.device)
    torch.nn.init.xavier_uniform_(prototypes)

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

    assert len(batched_prototypes) == steps_per_epoch

    # Set up an optimizer (does Adam works well for this?)
    optimizer = torch.optim.Adam(batched_prototypes, lr=0.01)

    # Setup the tensorboard writer
    timestamp = strftime("%Y-%m-%d_%H-%M", gmtime())
    tb_logs_dir = output_plot_dirpath / "tb_logs" / timestamp
    tb_logs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Tensorboard log diretory set at:")
    logger.info(f"{tb_logs_dir}")
    writer = SummaryWriter(log_dir=tb_logs_dir)

    # Iterate over all epochs
    # trainer_version_number = extract_int_string_from_string(checkpoint_path.parent.name)
    max_epochs = dir_config.args_dict.get("max_epochs", 2000)
    prototype_batch: torch.Tensor
    logger.info(f"Input prototype generation loop running for {max_epochs} epochs ...")

    beta_div = dir_config.args_dict.get("beta")
    alpha_fac = dir_config.args_dict.get("alpha_weight")
    sinkhorn_start_epoch = dir_config.args_dict.get("sinkhorn_start_epoch")
    assert max_epochs >= sinkhorn_start_epoch, "Sinkhorn start epoch must be smaller than max epochs"

    with tqdm(total=steps_per_epoch, desc=f"Input prototype optimization") as pbar:  # Progress bar
        # Proced for all epochs
        for epoch_idx in range(max_epochs):
            # Loop over all samples of the batch
            for step_idx, prototype_batch in enumerate(batched_prototypes):  # Optimization happens inline
                if epoch_idx < sinkhorn_start_epoch:
                    # Phase 1: no Sinkhorn, just softmax
                    prototypes_sm = torch.softmax(prototype_batch, dim=-1)
                else:
                    # Phase 2: Sinkhorn with increasing iterations and decreasing temperature
                    frac = (epoch_idx - sinkhorn_start_epoch) / (max_epochs - sinkhorn_start_epoch)  # 0 → 1
                    temp = 0.05**frac  # temperature decay
                    start_iter = 3
                    n_iter = int(start_iter + frac * (15 - start_iter))  # T: 3 → 15 iterations
                    alpha = frac * alpha_fac  # sharpness weight: 0 → 0.5

                    # Logits from prototypes, apply Sinkhorn
                    prototypes_sm = sinkhorn(prototype_batch, n_iter=n_iter, temperature=temp)

                    sharp_loss = (prototypes_sm * (1 - prototypes_sm)).sum(dim=(1, 2)).mean()

                    div_loss = diversity_loss(prototypes_sm)

                # Retrieve the embeddings
                in_emb = prototypes_sm @ tgt_vectors_norm

                # Derive metric
                x = lit_model.transformer.embedding(in_emb, test_values, embedding_input=True)
                x = lit_model.transformer.transformer_core(x)
                x = lit_model.transformer.decoder(x)

                if isinstance(x, tuple):
                    output, _ = lit_model.transformer.final_layer(x[0]), x[1]
                else:
                    output, _ = lit_model.transformer.final_layer(x), None

                if epoch_idx < sinkhorn_start_epoch:
                    loss = output.mean()
                    sharp_loss = 0
                else:
                    loss = output.mean() + alpha * sharp_loss + beta_div * div_loss
                    sharp_loss = sharp_loss.item()

                # Accumulate gradients
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                info_dict = {
                    "epoch": epoch_idx,
                    "average_score": torch.mean(output).item(),
                    "sharp_loss": sharp_loss,
                    "overall_loss": loss.item(),
                }

                for key, value in info_dict.items():
                    if key != "epoch":
                        writer.add_scalar(f"loss/{key}", value, epoch_idx * steps_per_epoch + step_idx)
                pbar.set_postfix(info_dict)
                pbar.update()
                # Reset progress bar
                pbar.reset()

    # Fix the prototypes
    bit_representations = fix_prototypes(batched_prototypes, n_iter, temp, design_len)
    proto_bit_tensor = torch.stack(bit_representations)

    # Filter the prototypes to generates based on their cost and existence in previous DB
    data = filter_generated_prototypes(
        proto_bit_tensor=proto_bit_tensor,
        lit_model=lit_model,
        dir_config=dir_config,
        keep_percentage=keep_percentage,
        values=values,
        target_nb=target_nb,
    )

    return data


def fix_prototypes(batched_prototypes, n_iter, temp, design_len):
    # Get the two's complement tensor as a template
    tc_bit_tensor = get_twos_complement_tensor(n_bits=4)

    # Create a list to store the bit representations
    bit_representations = []

    # Fix each design iteratively
    for bp in batched_prototypes:
        bp_sh = sinkhorn(bp, n_iter=n_iter, temperature=temp)
        bp_sh_norm = bp_sh / bp_sh.sum(2, keepdim=True)

        for j, p_smi in enumerate(bp_sh_norm):
            p_sm = p_smi.clone()
            trigger = False
            row_to_index_map = {}
            for _ in range(design_len - 1):  # We only go to minus 1 because the last assignment is not sampled.
                # Find the index that has the largest softmax value
                argmax_index = p_sm.argmax().item()
                # Derive the row
                rel_row = argmax_index // design_len
                # Detect prototypes that have not fully converged - These are skipped (occurs rarely).
                if p_sm[rel_row].sum() == 0:
                    trigger = True
                    break
                # Sample according to the probability distribution in the row
                # (most cases should practically be deterministic)
                sampled_index = torch.multinomial(p_sm[rel_row], num_samples=1).item()
                # Store the selected index for the current row
                row_to_index_map[rel_row] = sampled_index
                # Zero the row and column for the next iteration
                p_sm[rel_row] = 0
                p_sm[:, sampled_index] = 0

            # We skip prototypes that have not converged.
            if trigger is True:
                continue

            # The last row and columns does not need to be sampled, as it is deterministic.
            all_value_set = set(range(design_len))
            missing_key = (all_value_set - set(row_to_index_map.keys())).pop()
            missing_value = (all_value_set - set(row_to_index_map.values())).pop()
            row_to_index_map[missing_key] = missing_value

            # Sort the dictionary values
            row_to_index_map_sorted = dict(sorted(row_to_index_map.items()))

            # Append the bit representations
            bit_representations.append(torch.stack([tc_bit_tensor[k] for k in row_to_index_map_sorted.values()]))

    return bit_representations


def filter_generated_prototypes(proto_bit_tensor, lit_model, dir_config, keep_percentage, values, target_nb):
    # Deduplicate within batch
    unique_indices = deduplicate_batch(proto_bit_tensor)
    if len(unique_indices) < len(proto_bit_tensor):
        logger.info(
            f"Found {len(proto_bit_tensor) - len(unique_indices)} duplicates in generated batch; removing them."
        )
    proto_bit_tensor = proto_bit_tensor[unique_indices]

    # Filter out prototypes that already exist in previous generation DB
    try:
        analyzer = setup_analyzer(**dir_config.args_dict)
        gener_df = analyzer.gener_df if hasattr(analyzer, "gener_df") else pd.DataFrame()
    except Exception:
        logger.warning("Could not load previous generation database; skipping existence filtering.")
        gener_df = pd.DataFrame()

    if not gener_df.empty and "encodings_input" in gener_df.columns:
        existing_list = gener_df["encodings_input"].map(lambda x: enc_dict_to_tensor(eval(x))).to_list()
        if len(existing_list) > 0:
            existing_encodings_tensors = torch.stack(existing_list, dim=0)
            keep_indices = subtract_batches(proto_bit_tensor, existing_encodings_tensors)
            if len(keep_indices) < len(proto_bit_tensor):
                logger.info(
                    f"Filtered out {len(proto_bit_tensor) - len(keep_indices)} already-existing prototypes from DB."
                )
            proto_bit_tensor = proto_bit_tensor[keep_indices]
    else:
        logger.info("No previous generation DB found; using all generated prototypes.")

    # Score all remaining prototypes with the network (post-fix scoring)
    # Build values tensor once (same as during optimization)
    test_values = torch.arange(0, values.shape[1]).unsqueeze(0).to(lit_model.device).requires_grad_(False)
    score_postfix, _ = process_in_splits(
        lit_model, proto_bit_tensor, test_values, max_batch_size=dir_config.args_dict["batch_size"]
    )
    score_postfix = score_postfix.detach().cpu().view(-1)

    # Select keep_percentage best and probabilistically sample target_nb of them
    if len(score_postfix) == 0:
        logger.warning("No prototypes left after filtering; returning empty set.")
        data = {"all_prototypes": proto_bit_tensor, "costs": score_postfix}
        return data

    quant = torch.quantile(score_postfix, keep_percentage)
    best_mask = score_postfix <= quant
    true_indices = torch.where(best_mask)[0]

    filtered_scores = score_postfix[best_mask]
    # Lower scores => higher probability
    temperature = 1.0
    probs = torch.softmax(-filtered_scores / temperature, dim=0).view(-1)

    nb_samples = min(target_nb, len(true_indices))
    sampled_local_indices = torch.multinomial(probs, nb_samples, replacement=False)
    sampled_indices = true_indices[sampled_local_indices]

    final_mask = torch.zeros_like(best_mask, dtype=torch.bool)
    final_mask[sampled_indices] = True

    proto_bit_tensor = proto_bit_tensor[final_mask]
    score_postfix = score_postfix[final_mask]

    data = {"all_prototypes": proto_bit_tensor, "costs": score_postfix}

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


def process_in_splits(lit_model, input_tensor, test_values, max_batch_size=256):
    """Run the model in manageable splits to avoid OOM.

    input_tensor: binary prototypes [B, 16, 4] where each 4-bit row is a code.
    We convert to indices via a lookup against the codebook, then one-hot to embed.
    """
    outputs = []
    n_vectors = lit_model.transformer.embedding.encoding_embeddings.weight.shape[0]
    tgt_vectors = torch.cat(
        [
            lit_model.transformer.embedding.encoding_embeddings(torch.Tensor([j]).to(torch.int32).to(lit_model.device))
            for j in range(n_vectors)
        ]
    )
    tgt_vectors_norm = torch.nn.functional.normalize(tgt_vectors, p=2, dim=-1).detach()

    tc_bit_tensor = get_twos_complement_tensor(n_bits=4)
    tc_bit_tensor = tc_bit_tensor.to(lit_model.device)

    with torch.no_grad():
        for i in range(0, input_tensor.shape[0], max_batch_size):
            batch = input_tensor[i : i + max_batch_size].to(lit_model.device)
            tv = test_values.to(lit_model.device)
            # Map each 4-bit row to its index in the codebook
            # Compare and get matches of shape [B,16,16]
            matches = (batch[:, :, None, :] == tc_bit_tensor[None, None, :, :]).all(-1)
            indices = matches.float().argmax(dim=-1)  # [B, 16]
            one_hot = torch.nn.functional.one_hot(indices, num_classes=n_vectors).float()
            in_emb = one_hot @ tgt_vectors_norm  # [B,16,emb_dim]
            x = lit_model.transformer.embedding(in_emb, tv, embedding_input=True)
            x = lit_model.transformer.transformer_core(x)
            x = lit_model.transformer.decoder(x)
            if isinstance(x, tuple):
                out, _ = lit_model.transformer.final_layer(x[0]), x[1]
            else:
                out, _ = lit_model.transformer.final_layer(x), None
            outputs.append(out.detach().cpu())
    return torch.cat(outputs, dim=0), None


def subtract_batches(A: torch.Tensor, B: torch.Tensor):
    """Return indices of rows in A that are NOT present in B (exact match)."""
    # Flatten
    A_flat = A.view(A.shape[0], -1)
    B_flat = B.view(B.shape[0], -1)
    # Hash B
    B_hashes = set(hash(bytes(row.cpu().numpy())) for row in B_flat)
    # Keep only those in A that are NOT in B
    keep_indices = [i for i, row in enumerate(A_flat) if hash(bytes(row.cpu().numpy())) not in B_hashes]
    return keep_indices


def deduplicate_batch(images: torch.Tensor):
    """Keep first occurrence of each unique binary image (exact match)."""
    B = images.shape[0]
    # Flatten each image
    flat_images = images.view(B, -1)
    # Hash rows for fast lookup
    hashes = [hash(bytes(row.cpu().numpy())) for row in flat_images]
    seen = {}
    unique_indices = []
    for i, h in enumerate(hashes):
        if h not in seen:
            seen[h] = i
            unique_indices.append(i)
    return unique_indices


def patch_prototypes(prototype_data_path: Path, dir_config: ConfigDir, fix_strategy_version: int = 3):
    if prototype_data_path.exists():
        protoype_pattern_data = load_serialized_data(prototype_data_path)
    else:
        raise ValueError(f"Provided prototype data path does not exists.")

    if protoype_pattern_data["prototype"].shape[0] == protoype_pattern_data["score"].shape[0]:
        # score_ok intentionally unused; no action required
        pass
    else:
        # score_ok intentionally unused; no action required
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
    start_time = time()
    # try:
    # Do input prototype generation
    gener_args_dict = dir_config.args_dict

    if dir_config.args_dict.get("dst_output_dir_name") is not None:
        # Update args dict for generation. Prototype data and generated design will be stored there.
        gener_args_dict.update({"output_dir_name": dir_config.args_dict.get("dst_output_dir_name")})

    # Setup Generation
    gener_dir_config = ConfigDir(is_analysis=False, **gener_args_dict)
    logger.warning(f"All prototype related generation files will be saved in the following root output directory:")
    logger.warning(gener_dir_config.root_output_dir)

    if dir_config.args_dict.get("prototype_data_path", None) is None:
        # No path to prototypes provided: we need to generate the prototypes
        protoype_pattern_data = generate_prototype_patterns(
            **model_workers_kwargs,
            nb_gener_prototypes=dir_config.args_dict.get("nb_gener_prototypes", False),
            data_save_path=gener_dir_config.analysis_out_dir,
            keep_percentage=dir_config.args_dict.get("keep_percentage", 0.25),
        )
    else:
        # Path to prototypes was provided: we want to fix the generated prototypes again
        logger.info(f"Prototypes for generating designs will be loaded from the data file specified ...")
        prototype_data_path = Path(dir_config.args_dict.get("prototype_data_path", None))
        protoype_pattern_data = patch_prototypes(prototype_data_path, dir_config=gener_dir_config)

    # From tensors to dictionnary
    proto_encoding_dicts_list = convert_proto_to_encoding_dictionary(
        dir_config=dir_config, fixed_prototypes_batched=protoype_pattern_data["all_prototypes"]
    )

    # Remove duplicates
    prototype_df = pd.DataFrame([{"proto_str": str(p)} for p in proto_encoding_dicts_list])
    index_to_keep = set(prototype_df.drop_duplicates().index.tolist())
    proto_encoding_dicts_list = [p for i, p in enumerate(proto_encoding_dicts_list) if i in index_to_keep]

    # Find which design numbers to use
    nb_protos = len(proto_encoding_dicts_list)
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
        existing_encodings_dicts=proto_encoding_dicts_list,
    )

    status = "Success"
    error_msg = ""

    # except:
    #     status="Failed"
    #     error_msg = traceback.format_exc()
    #     generated_design_paths_list = None
    #
    send_email(
        config_dict=dir_config.args_dict,
        start_time=start_time,
        status=status,
        error_message=error_msg,
        calling_module="ProtoGen",
        root_output_dir=gener_dir_config.root_output_dir,
    )
    # Touch a token file to indicate prototype generation completed successfully.
    try:
        token_path = gener_dir_config.root_output_dir / "proto_generation_done.token"
        with open(token_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": int(time()),
                        "nb_generated": len(generated_design_paths_list)
                        if generated_design_paths_list is not None
                        else 0,
                        "status": status,
                    }
                )
            )
        logger.info(f"Wrote prototype completion token at: {token_path}")
    except Exception as e:
        logger.warning(f"Failed to write prototype completion token: {e}")

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
    arg_parser.add_argument(
        "--score_type",
        type=str,
        default="trans",
        choices=list(ScoreComputeHelper.transformer_map.keys()),
        help="Which design score to use for this experiment.",
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
        "--sinkhorn_start_epoch", type=int, default=100, help="Epoch where we switch from softmax to sinkhorn"
    )

    arg_parser.add_argument("--alpha_weight", type=float, default=0.5, help="alpha weight for the sharpness")

    arg_parser.add_argument("--beta", type=float, default=80.0, help="beta weight for the diversity")

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
    do_get_trainer_version_number = args_dict.get("do_model_analysis", False)

    # Load the model and setup the kwargs associated for model related jobs
    trainer_version_number = None
    if do_load_model:
        yml_config_path = args_dict.get("yml_config_path", None)
        if yml_config_path is None:
            raise ValueError("Specifying the yaml config file is now required!")
        logger.info("Loading model for analysis ... ")
        if do_get_trainer_version_number:
            lit_model, datamodule, checkpoint_path, trainer_version_number = load_model_and_datamodule(
                dir_config, yml_config_path, return_trainer_version_number=True
            )
        else:
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
        make_saliency_map(**model_workers_kwargs, trainer_version_number=trainer_version_number)

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
