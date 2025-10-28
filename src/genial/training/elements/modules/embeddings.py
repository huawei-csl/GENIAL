# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

import torch
from torch import nn

from genial.training.elements.configs import ModelConfig

from loguru import logger


class PointNetEmbedding(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.task = model_config.task
        self.n_cls_token = model_config.n_cls_token
        self.encoding_width = model_config.encoding_width
        self.d_model = model_config.d_model
        self.nb_scores = model_config.nb_scores

        # Define input and output neuron count
        half = self.d_model // 2
        quarter = self.d_model // 4

        # Point Net Embedding
        self.conv1 = nn.Conv1d(in_channels=self.encoding_width, out_channels=quarter, kernel_size=1)
        self.conv2 = nn.Conv1d(quarter, half, 1)
        self.conv3 = nn.Conv1d(half, self.d_model, 1)

        self.bn1 = nn.BatchNorm1d(quarter)
        self.bn2 = nn.BatchNorm1d(half)
        self.bn3 = nn.BatchNorm1d(self.d_model)

        # Positional Encoding
        sequence_length = 2**self.encoding_width
        if self.n_cls_token:
            self.cls_token = nn.Parameter(torch.randn(self.n_cls_token, self.d_model))
            sequence_length += self.n_cls_token
            logger.info(
                f"n_cls_token={self.n_cls_token} has been received | Model will evaluate the score from the unique CLS token prepended to the token sequence."
            )

        # Score Value
        if "synthv0_to_synthv3" in self.task:
            # Module to upscale other score value to token width
            self.score_to_token = nn.Linear(1, self.d_model)
            sequence_length += 1
            logger.info(f"Task is: {self.task} | Model will prepend the score value to the encoding tokens.")

        # Positional Encoding
        self.positional_encodings = nn.Embedding(num_embeddings=sequence_length, embedding_dim=self.d_model)

    def forward(self, x: torch.Tensor, values: torch.Tensor):
        # Split other score from encoding if present
        if "synthv0_to_synthv3" in self.task:
            score = x[1].unsqueeze(1)  # (B, 1)
            # Upscale Score Value to Token Width
            score_token = self.score_to_token(score).unsqueeze(1)
            x = x[0]

        # Point Net Embedding
        # We want to convolve over the sequence length (L)
        # With a single token taken per filter (to simply project them to a different dimension)
        # So, we permute the (B, L, D) to (B, D, L), so that one filter sees D as the channels, and each filter treats a single token at a time
        x = x.permute(0, 2, 1)

        # Scaling up
        x = nn.functional.gelu(self.bn1(self.conv1(x)))
        x = nn.functional.gelu(self.bn2(self.conv2(x)))
        x = nn.functional.gelu(self.bn3(self.conv3(x)))

        # Permuting back for transformer core
        x = x.permute(0, 2, 1)

        # Pre-pend the sequence length with the score token if needed
        if "synthv0_to_synthv3" in self.task:
            x = torch.cat([score_token, x], dim=1)

            # Update values
            to_concat = values.max(dim=1, keepdim=True).values + 1
            values = torch.cat([values, to_concat], dim=1)

        # Re-prepend with the CLS token(s)
        if self.n_cls_token:
            # Concatenate cls token. Note: there can be multiple cls tokens
            expanded_cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((expanded_cls_token, x), dim=1)

            # Update values
            to_concat = values.max(dim=1, keepdim=True).values + 1
            values = torch.cat([values, to_concat], dim=1)

        # Positional Encoding
        x = x + self.positional_encodings(values)

        return x


class PointNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Define input and output neuron count
        half = out_channels // 2
        quarter = out_channels // 4

        # Point Net Embedding
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=quarter, kernel_size=1)
        self.conv2 = nn.Conv1d(quarter, half, 1)
        self.conv3 = nn.Conv1d(half, out_channels, 1)

        self.bn1 = nn.BatchNorm1d(quarter)
        self.bn2 = nn.BatchNorm1d(half)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor):
        # Scaling up
        x = nn.functional.gelu(self.bn1(self.conv1(x)))
        x = nn.functional.gelu(self.bn2(self.conv2(x)))
        x = nn.functional.gelu(self.bn3(self.conv3(x)))

        return x


class PointNetEmbeddingV2(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.task = model_config.task
        self.n_cls_token = model_config.n_cls_token
        self.encoding_width = model_config.encoding_width
        self.d_model = model_config.d_model
        self.nb_scores = model_config.nb_scores

        self.pointnet_in_enc = PointNet(in_channels=self.encoding_width, out_channels=self.d_model)

        # Positional Encoding
        seq_length = 2**self.encoding_width
        if self.n_cls_token:
            self.cls_token = nn.Parameter(torch.randn(self.n_cls_token, self.d_model))
            seq_length += self.n_cls_token
            logger.info(
                f"n_cls_token={self.n_cls_token} has been received | Model will evaluate the score from the {self.cls_token.shape[0]} CLS token(s) apprended to the token sequence."
            )

        # Score Value for Another Synth Method
        if "synthv0_to_synthv3" in self.task:
            # Module to upscale other score value to token width
            self.score_to_token = nn.Linear(1, self.d_model)
            seq_length += 1
            logger.info(f"Task is: {self.task} | Model will prepend the score value to the encoding tokens.")

        # Output Encoding
        if "custom_io_encodings" in self.task:
            self.encoding_width_output = model_config.encoding_width_output
            seq_length += 60
            self.pointnet_out_enc = PointNet(in_channels=self.encoding_width_output, out_channels=self.d_model)

        # Positional Encoding
        self.positional_encodings = nn.Embedding(num_embeddings=seq_length, embedding_dim=self.d_model)

        logger.error(f"PointNet Embedding V2 with task {self.task} has been setup.")

    def forward(self, x: list[torch.Tensor] | torch.Tensor, values: torch.Tensor):
        # Split other score from encoding if present
        if "synthv0_to_synthv3" in self.task:
            # Data format: x is [input_encoding_tensor, score] (provided by Mixed Synthed Dataset)
            score = x[1].unsqueeze(1)  # (B, 1)
            score_token = self.score_to_token(score).unsqueeze(1)
            # Pursue with input_encoding_tensor
            x = x[0]

        if "custom_io_encodings" in self.task:
            # Data format: x is [input_encoding_tensor, output_encoding_tensor]
            x_out = x[1]
            x = x[0]

            # Apply PointNet to Output Encoding Tensor
            x_out = x_out.permute(0, 2, 1)
            x_out = self.pointnet_out_enc(x_out)
            # Permuting back for transformer core
            x_out = x_out.permute(0, 2, 1)

        # Point Net Embedding
        # We want to convolve over the sequence length (L)
        # With a single token taken per filter (to simply project them to a different dimension)
        # So, we permute the (B, L, D) to (B, D, L), so that one filter sees D as the channels, and each filter treats a single token at a time
        x = x.permute(0, 2, 1)
        x = self.pointnet_in_enc(x)
        # Permuting back for transformer core
        x = x.permute(0, 2, 1)

        if "custom_io_encodings" in self.task:
            x = torch.cat([x, x_out], dim=1)

        # Pre-pend the sequence length with the score token if needed
        if "synthv0_to_synthv3" in self.task:
            x = torch.cat([score_token, x], dim=1)

            # Update values
            to_concat = values.max(dim=1, keepdim=True).values + 1
            values = torch.cat([values, to_concat], dim=1)

        # Re-prepend
        if self.n_cls_token:
            # Concatenate cls token
            expanded_cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((expanded_cls_token, x), dim=1)

            # Update values
            if "custom_io_encodings" in self.task and "ssl" in self.task:
                to_concat_in = values[0].max(dim=1, keepdim=True).values + 1
                values[0] = torch.cat([values[0], to_concat_in], dim=1)
                to_concat_out = values[1].max(dim=1, keepdim=True).values + 1
                values[1] = torch.cat([values[1], to_concat_out], dim=1) + values[0].max(dim=1, keepdim=True).values + 1
                values = torch.cat(values, dim=1)
            else:
                to_concat_temp = values.max(dim=1, keepdim=True).values + 1
                to_concat = [values]
                # Add as many values as there are cls_token
                for i in range(self.n_cls_token):
                    to_concat.append(to_concat_temp + i)
                values = torch.cat(to_concat, dim=1)

        # Positional Encoding
        x = x + self.positional_encodings(values)[:, : x.shape[-2], :]

        return x


class EmbeddingInsteadPointNet(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.task = model_config.task
        self.n_cls_token = model_config.n_cls_token
        self.encoding_width = model_config.encoding_width
        self.d_model = model_config.d_model

        # CLS embedding
        nb_values = 2**self.encoding_width

        # Encoding embedding
        self.encoding_embeddings = nn.Embedding(num_embeddings=nb_values, embedding_dim=self.d_model)

        if self.n_cls_token:
            self.cls_token = nn.Parameter(torch.randn(self.n_cls_token, self.d_model))
            nb_values += self.n_cls_token
            logger.info(
                f"n_cls_token={self.n_cls_token} has been received | Model will evaluate the score from the unique CLS token apprended to the token sequence."
            )

        # Bit weights used to convert bit encodings into index for the encoding embeddings retrieval
        self.register_buffer(
            "bit_weights", torch.tensor([2**i for i in reversed(range(self.encoding_width))], dtype=torch.float)
        )

        # Positional Encoding
        self.positional_encodings = nn.Embedding(num_embeddings=nb_values, embedding_dim=self.d_model)

        logger.error(f"EmbeddingInsteadPointNet with task {self.task} has been setup.")

    def forward(self, x: list[torch.Tensor] | torch.Tensor, values: torch.Tensor, embedding_input=False):
        if not embedding_input:
            # Retrieve the embedding for each encoding
            x = self.encoding_embeddings(torch.matmul(x, self.bit_weights).type(torch.int))
            # L2-normalize embeddings to unit norm
            x = torch.nn.functional.normalize(x, p=2, dim=-1)

        # Re-prepend
        if self.n_cls_token:
            # Concatenate cls token
            expanded_cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((expanded_cls_token, x), dim=1)
            to_concat_temp = values.max(dim=1, keepdim=True).values + 1
            to_concat = [values]
            for i in range(self.n_cls_token):
                to_concat.append(to_concat_temp + i)
            values = torch.cat(to_concat, dim=1)

        # Positional Encoding
        x = x + self.positional_encodings(values)[:, : x.shape[-2], :]

        return x
