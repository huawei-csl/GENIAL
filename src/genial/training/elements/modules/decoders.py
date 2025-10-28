# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

import torch
from torch import nn
import math

from genial.training.elements.configs import ModelConfig
from collections import OrderedDict

from loguru import logger


class AbstractLinearDecoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int):
        super().__init__()

        size_00 = d_model * seq_length
        size_01 = d_model * seq_length // 2
        size_10 = d_model * seq_length // 2
        size_11 = d_model * seq_length // 4
        size_20 = d_model * seq_length // 4
        size_21 = 1 * seq_length
        size_30 = 1 * seq_length
        size_31 = 1

        # Point Net Embedding
        self.output_head_0 = nn.Linear(size_00, size_01)
        self.output_head_1 = nn.Linear(in_features=size_10, out_features=size_11)
        self.output_head_2 = nn.Linear(in_features=size_20, out_features=size_21)
        self.output_head_3 = nn.Linear(in_features=size_30, out_features=size_31)

        self.bn0 = nn.BatchNorm1d(size_01)
        self.bn1 = nn.BatchNorm1d(size_11)
        self.bn2 = nn.BatchNorm1d(size_21)

        self.activation_0 = nn.GELU()
        self.activation_1 = nn.GELU()
        self.activation_2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LinearDecoder expects the output of a Transformer model as input
        # Hence flattens (B, L, D) -> (B, L*D) before applying the linear layers
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = self.output_head_0(x)
        x = self.bn0(x)
        x = self.activation_0(x)

        x = self.output_head_1(x)
        x = self.bn1(x)
        x = self.activation_1(x)

        x = self.output_head_2(x)
        x = self.bn2(x)
        x = self.activation_2(x)

        x = self.output_head_3(x)
        # x = self.final_layer(x)

        return x


class _LinearDecoder(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.linear_decoder = AbstractLinearDecoding(
            d_model=model_config.d_model, seq_length=model_config.max_sequence_length
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_decoder(x)


class ReducingLinearDecoder(nn.Module):
    def __init__(self, model_config: ModelConfig):
        """
        This decoder first reduces the output dimension to (B, L=16, D=256)
        And then applies the usual Linear Decoder model which flattens the sequence of tokens
        """
        super().__init__()

        # TODO: clean this

        # Reducing along the sequence length
        D = model_config.d_model  # Number of feature dimensions
        L = model_config.max_sequence_length

        self.depthwise_conv1d_0 = torch.nn.Conv1d(
            in_channels=D,
            out_channels=D,
            kernel_size=3,  # How many encodings to cross correlate with one convolution layer convolution
            stride=2,
            padding=2,
            dilation=2,
            groups=1,  # Depthwise convolution - make sure the different features of the encodings do not interact together
            # groups=model_config.encoding_width  # Depthwise convolution - make sure the different features of the encodings do not interact together
        )
        self.pointwise_conv1d_0 = torch.nn.Conv1d(
            in_channels=D, out_channels=D // 2, kernel_size=1, stride=1, padding=0
        )
        D = D // 2
        L = L // 2
        self.bn0 = nn.BatchNorm1d(D)
        self.activation_0 = torch.nn.GELU()

        self.depthwise_conv1d_1 = torch.nn.Conv1d(
            in_channels=D,
            out_channels=D,
            kernel_size=3,  # How many encodings to cross correlate with one convolution layer convolution
            stride=2,
            padding=2,
            dilation=2,
            groups=1,  # Depthwise convolution - make sure the different features of the encodings do not interact together
            # groups=model_config.encoding_width  # Depthwise convolution - make sure the different features of the encodings do not interact together
        )
        # self.pointwise_conv1d_1 = torch.nn.Conv1d(
        #     in_channels=D,
        #     out_channels=D//2,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0
        # )
        # D = D//2
        L = L // 2
        self.bn1 = nn.BatchNorm1d(D)
        self.activation_1 = torch.nn.GELU()

        self.depthwise_conv1d_2 = torch.nn.Conv1d(
            in_channels=D,
            out_channels=D,
            kernel_size=3,  # How many encodings to cross correlate with one convolution layer convolution
            stride=2,
            padding=2,
            dilation=2,
            groups=1,  # Depthwise convolution - make sure the different features of the encodings do not interact together
            # groups=model_config.encoding_width  # Depthwise convolution - make sure the different features of the encodings do not interact together
        )
        # self.pointwise_conv1d_2 = torch.nn.Conv1d(
        #     in_channels=D,
        #     out_channels=D//2,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0
        # )
        # D = D//2
        L = L // 2
        self.bn2 = nn.BatchNorm1d(D)
        self.activation_2 = torch.nn.GELU()

        self.depthwise_conv1d_3 = torch.nn.Conv1d(
            in_channels=D,
            out_channels=D,
            kernel_size=3,  # How many encodings to cross correlate with one convolution layer convolution
            stride=2,
            padding=2,
            dilation=2,
            groups=1,  # Depthwise convolution - make sure the different features of the encodings do not interact together
            # groups=model_config.encoding_width  # Depthwise convolution - make sure the different features of the encodings do not interact together
        )
        # self.pointwise_conv1d_3 = torch.nn.Conv1d(
        #     in_channels=D,
        #     out_channels=D//2,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0
        # )
        # D = D//2
        L = L // 2
        self.bn3 = nn.BatchNorm1d(D)
        self.activation_3 = torch.nn.GELU()

        self.linear_decoder = AbstractLinearDecoding(d_model=D, seq_length=L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute to (B, D, L)
        x = x.permute(0, 2, 1)

        x = self.depthwise_conv1d_0(x)
        x = self.pointwise_conv1d_0(x)
        x = self.bn0(x)
        x = self.activation_0(x)

        x = self.depthwise_conv1d_1(x)
        # x = self.pointwise_conv1d_1(x)
        x = self.bn1(x)
        x = self.activation_1(x)

        x = self.depthwise_conv1d_2(x)
        # x = self.pointwise_conv1d_2(x)
        x = self.bn2(x)
        x = self.activation_2(x)

        x = self.depthwise_conv1d_3(x)
        # x = self.pointwise_conv1d_3(x)
        x = self.bn3(x)
        x = self.activation_3(x)

        # Permute back to (B, L_reduced, D_reduced)
        x = x.permute(0, 2, 1)

        return self.linear_decoder(x)


class PointNetLinearDecoder(nn.Module):
    def __init__(self, model_config: ModelConfig):
        """
        This decoder first reduces the output dimension to (B, L=16, D=256)
        And then applies the usual Linear Decoder model which flattens the sequence of tokens
        """
        super().__init__()

        # TODO: clean this

        # Reducing along the sequence length
        self.d_model = model_config.d_model  # Number of feature dimensions
        self.max_seq_len = model_config.max_sequence_length

        # Setup number of reductions
        self.target_flattened_width = 4096

        # Make sure that the maximum sequence length is a multiple of the target target_flattened_width
        assert self.target_flattened_width % model_config.max_sequence_length == 0
        self.target_token_width = self.target_flattened_width // self.max_seq_len
        self.number_of_stack = int(math.log2(self.d_model // self.target_token_width))

        self.pointnet = torch.nn.Sequential()

        dim = self.d_model
        for i in range(self.number_of_stack):
            self.pointnet.append(nn.Conv1d(in_channels=dim, out_channels=dim // 2, kernel_size=1))
            self.pointnet.append(nn.BatchNorm1d(dim // 2))
            self.pointnet.append(nn.GELU())
            dim //= 2
        assert dim == self.target_token_width

        self.linear_decoder = AbstractLinearDecoding(d_model=dim, seq_length=self.max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Point Net Embedding
        x = x.permute(0, 2, 1)  # From (B, L, D) to (B, D, L) | D are the "channels" of the conv1d kernels
        x = self.pointnet(x)
        x = x.permute(0, 2, 1)

        # Flatten and project to scalar
        x = self.linear_decoder(x)

        return x


class AttentionDecoder(nn.Module):
    def __init__(self, model_config: ModelConfig):
        """Piece of code kept for retro compatibility, new transformer models should use _LinearDecoder."""
        super().__init__()

        self.vae_constraint = False
        self.d_model = model_config.d_model
        self.kv_seq_len = model_config.max_sequence_length
        if model_config.n_cls_token:
            self.kv_seq_len += model_config.n_cls_token
        self.nb_scores = model_config.nb_scores
        self.q_seq_len = self.nb_scores

        self.activation = nn.GELU()

        decoder_layer_kwargs = {
            "d_model": model_config.d_model,
            "nhead": model_config.nhead,
            "dim_feedforward": model_config.dim_feedforward,
            "dropout": model_config.dropout,
            "activation": model_config.activation,
            "layer_norm_eps": 1e-05,
            "batch_first": True,
            "norm_first": False,
            "bias": True,
        }
        decoder_layer = nn.TransformerDecoderLayer(**decoder_layer_kwargs)

        self.latent_query = nn.Parameter(torch.randn(1, self.q_seq_len, self.d_model))

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=model_config.num_decoder_layers,
        )

        full = self.d_model
        half = full // 2
        self.output_heads = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            ("linear0", nn.Linear(full, half)),
                            ("bn0", nn.BatchNorm1d(half)),
                            ("act0", nn.GELU()),
                            ("drop0", nn.Dropout(0.1)),
                            ("linear1", nn.Linear(half, 1)),
                        ]
                    )
                )
                for idx in range(self.nb_scores)
            ]
        )

        if model_config.vae_constraint:
            if self.nb_scores > 1:
                raise NotImplementedError(f"The Decoder dose not support both multi scores and VAE constraints.")
            self.vae_constraint = True
            self.one_n_half_size = int(self.d_model * 1.5)
            self.mean_var_pred_0 = nn.Linear(self.d_model, self.one_n_half_size)
            self.mean_var_bn = nn.BatchNorm1d(self.one_n_half_size)
            self.mean_var_pred_1 = nn.Linear(self.one_n_half_size, 2 * self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vae_output = None
        if self.vae_constraint:
            bs = x.shape[0]
            x1 = self.mean_var_pred_0(x)
            x2 = self.mean_var_bn(x1.reshape(-1, self.one_n_half_size)).reshape(bs, -1, self.one_n_half_size)
            x3 = self.mean_var_pred_1(x2)
            mu, logvar = x3.chunk(2, dim=-1)
            std_cls = torch.exp(0.5 * logvar[:, 0])
            mu_cls = mu[:, 0]
            eps = torch.randn_like(std_cls)
            x = mu_cls + eps * std_cls
            vae_output = mu, logvar
        else:
            vae_output = None

        x = self.transformer_decoder(
            tgt=torch.tile(self.latent_query, (x.shape[0], 1, 1)),
            memory=x,
        )

        _xs = [self.output_heads[idx](x[:, idx, :]) for idx in range(self.nb_scores)]
        x = torch.cat(_xs, dim=1)

        return x, vae_output


class MultiTargetLinearDecoder(nn.Module):
    def __init__(self, model_config: ModelConfig):
        """
        A multi target version of the deafult linear decoder.
        """
        super().__init__()

        self.vae_constraint = False
        self.d_model = model_config.d_model
        self.n_cls_token = model_config.n_cls_token
        self.nb_scores = model_config.nb_scores

        self.activation = nn.GELU()

        assert self.n_cls_token >= 1, (
            "Number of cls tokens should be at least 1 when using the MultiTargetLinearDecoder"
        )

        # By default if there are several cls tokens, they get concatenated in the decoding phase into a single token
        full = int(self.d_model * self.n_cls_token)
        half = full // 2

        self.output_heads = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            ("linear0", nn.Linear(full, half)),
                            ("bn0", nn.BatchNorm1d(half)),
                            ("act0", nn.GELU()),
                            ("drop0", nn.Dropout(0.1)),
                            ("linear1", nn.Linear(half, 1)),
                        ]
                    )
                )
                for idx in range(self.nb_scores)
            ]
        )

        if model_config.vae_constraint:
            if self.nb_scores > 1:
                raise NotImplementedError(f"The Decoder dose not support multi scores and vae constraints.")
            self.vae_constraint = True
            self.one_n_half_size = int(self.d_model * 1.5)
            self.mean_var_pred_0 = nn.Linear(self.d_model, self.one_n_half_size)
            self.mean_var_bn = nn.BatchNorm1d(self.one_n_half_size)
            self.mean_var_pred_1 = nn.Linear(self.one_n_half_size, 2 * self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vae_output = None
        if self.n_cls_token == 1:
            if self.vae_constraint:
                bs = x.shape[0]
                x1 = self.mean_var_pred_0(x)
                x2 = self.mean_var_bn(x1.reshape(-1, self.one_n_half_size)).reshape(bs, -1, self.one_n_half_size)
                x3 = self.mean_var_pred_1(x2)
                mu, logvar = x3.chunk(2, dim=-1)
                std_cls = torch.exp(0.5 * logvar[:, 0])
                mu_cls = mu[:, 0]
                eps = torch.randn_like(std_cls)
                x = mu_cls + eps * std_cls
                vae_output = mu, logvar

            else:
                x = x[:, 0, :]
        else:
            raise ValueError(f"Unsupported for n_cls_token > 1.")

        _xs = [self.output_heads[idx](x) for idx in range(self.nb_scores)]
        x = torch.cat(_xs, dim=1)

        return x, vae_output


class LinearDecoder(nn.Module):
    def __init__(self, model_config: ModelConfig):
        """Piece of code kept for retro compatibility, new transformer models should use _LinearDecoder."""
        super().__init__()

        self.vae_constraint = False
        self.d_model = model_config.d_model
        self.n_cls_token = model_config.n_cls_token
        seq_len = model_config.max_sequence_length
        self.nb_scores = model_config.nb_scores

        self.activation = nn.GELU()

        if self.n_cls_token:
            dim = int(self.d_model * self.n_cls_token)
            half = dim // 2
            self.output_head_0 = nn.Linear(dim, half)
            self.output_head_1 = nn.Linear(half, 1)
            self.bn0 = nn.BatchNorm1d(half)
            self.dropout0 = nn.Dropout(0.1)

        else:
            full = model_config.d_model * seq_len
            half = full // 2
            quarter = full // 4

            # Point Net Embedding
            self.output_head_0 = nn.Linear(full, half)
            self.output_head_1 = nn.Linear(half, quarter)
            self.output_head_2 = nn.Linear(quarter, seq_len)
            self.output_head_3 = nn.Linear(seq_len, 1)

            self.bn0 = nn.BatchNorm1d(half)
            self.bn1 = nn.BatchNorm1d(quarter)
            self.bn2 = nn.BatchNorm1d(seq_len)

        logger.info(f"Initialized model with decoder type: LinearDecoder")

        if model_config.vae_constraint:
            if self.nb_scores > 1:
                raise NotImplementedError(f"The Decoder dose not support multi scores and vae constraints.")
            self.vae_constraint = True
            self.one_n_half_size = int(self.d_model * 1.5)
            self.mean_var_pred_0 = nn.Linear(self.d_model, self.one_n_half_size)
            self.mean_var_bn = nn.BatchNorm1d(self.one_n_half_size)
            self.mean_var_pred_1 = nn.Linear(self.one_n_half_size, 2 * self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vae_output = None
        if not self.n_cls_token:
            # LinearDecoder expects the output of a Transformer model as input
            # Hence flattens (B, L, D) -> (B, L*D) before applying the linear layers
            x = torch.flatten(x, start_dim=1, end_dim=-1)

            x = self.output_head_0(x)
            x = self.bn0(x)
            x = self.activation(x)

            x = self.output_head_1(x)
            x = self.bn1(x)
            x = self.activation(x)

            x = self.output_head_2(x)
            x = self.bn2(x)
            x = self.activation(x)

            x = self.output_head_3(x)
        else:
            if self.n_cls_token == 1:
                if self.vae_constraint:
                    bs = x.shape[0]
                    x1 = self.mean_var_pred_0(x)
                    x2 = self.mean_var_bn(x1.reshape(-1, self.one_n_half_size)).reshape(bs, -1, self.one_n_half_size)
                    x3 = self.mean_var_pred_1(x2)
                    mu, logvar = x3.chunk(2, dim=-1)
                    std_cls = torch.exp(0.5 * logvar[:, 0])
                    mu_cls = mu[:, 0]
                    eps = torch.randn_like(std_cls)
                    x = mu_cls + eps * std_cls
                    vae_output = mu, logvar

                else:
                    x = x[:, 0, :]
            elif self.n_cls_token == 2:
                x = torch.cat([x[:, 0, :], x[:, 1, :]], dim=1)
            else:
                raise ValueError(f"Unsupported for n_cls_token > 2.")

            x = self.output_head_0(x)
            x = self.bn0(x)
            x = self.activation(x)
            x = self.dropout0(x)

            x = self.output_head_1(x)

        return x, vae_output


class SslDecoder(nn.Module):
    def __init__(self, model_config: ModelConfig, cls_dropout=0.2):
        """Decoder for SSL task"""
        super().__init__()
        # By default, vae constraint is set to False. Updated later in __init__ method if needed.
        self.vae_constraint = False
        self.encoding_bit = model_config.encoding_width
        self.d_model = model_config.d_model
        self.nb_scores = model_config.nb_scores

        # GELU activation is used
        self.activation = nn.GELU()
        # The dropout applied on the CLS token output
        self.dropout = nn.Dropout(cls_dropout)

        # Output size = nb_encodings ** 2 = (2 ** encoding_bit) ** 2 = 2 ** (encoding_bit * 2)
        if self.encoding_bit > 4:
            output_size = 2 * (2 ** (self.encoding_bit // 2)) * (2 ** (self.encoding_bit))
        else:
            output_size = 2 ** (self.encoding_bit * 2)

        # print(output_size)
        mid_size = int((self.d_model + output_size) // 2)
        self.output_head_0 = nn.Linear(self.d_model, mid_size)
        self.output_head_1 = nn.Linear(mid_size, output_size)
        self.bn = nn.BatchNorm1d(mid_size)

        if model_config.vae_constraint:
            if self.nb_scores > 1:
                raise NotImplementedError(f"The Decoder dose not support multi scores and vae constraints.")
            self.vae_constraint = True
            self.one_n_half_size = int(self.d_model * 1.5)
            self.mean_var_pred_0 = nn.Linear(self.d_model, self.one_n_half_size)
            self.mean_var_bn = nn.BatchNorm1d(self.one_n_half_size)
            # 2 * self.d_model such that we obtain both the mean and variance prediction
            self.mean_var_pred_1 = nn.Linear(self.one_n_half_size, 2 * self.d_model)

        logger.info(f"Initialized model with decoder type: SslDecoder")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vae_output = None
        if self.vae_constraint:
            bs = x.shape[0]
            x1 = self.mean_var_pred_0(x)
            x2 = self.mean_var_bn(x1.reshape(-1, self.one_n_half_size)).reshape(bs, -1, self.one_n_half_size)
            x3 = self.mean_var_pred_1(x2)
            mu, logvar = x3.chunk(2, dim=-1)
            std_cls = torch.exp(0.5 * logvar[:, 0])
            mu_cls = mu[:, 0]
            eps = torch.randn_like(std_cls)
            x = mu_cls + eps * std_cls
            vae_output = mu, logvar

        else:
            #  Default situation: 1 CLS token
            x = x[:, 0, :]
        # Dropout applied to CLS token output
        x = self.dropout(x)
        # Two linear layers
        x = self.output_head_0(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.output_head_1(x)

        if self.encoding_bit > 4:
            return x.reshape(x.shape[0], 2 ** (self.encoding_bit // 2), -1, 2), vae_output
        else:
            return x.reshape(x.shape[0], -1, 2**self.encoding_bit), vae_output


class SslDecoderWoCls(nn.Module):
    def __init__(self, model_config: ModelConfig, encoding_bit=4):
        """Decoder for SSL task"""
        super().__init__()

        self.vae_constraint = False
        self.encoding_bit = encoding_bit
        self.d_model = model_config.d_model
        # Output size is the number of possible pairs of encodings times the number of bits times the number of labels.
        self.output_size = int(2**encoding_bit * (2**encoding_bit - 1) * encoding_bit * 4)
        self.mid_size = int((self.d_model + self.output_size) // 2)
        self.n_cls_token = model_config.n_cls_token

        self.activation = nn.GELU()
        self.output_head_0 = nn.Linear(self.d_model, self.mid_size)
        self.output_head_1 = nn.Linear(self.mid_size, self.output_size)
        self.bn0 = nn.BatchNorm1d(self.mid_size)

        self.dropout = nn.Dropout(0.2)

        output_size2 = 2 ** (encoding_bit * 2)
        mid_size2 = int((self.d_model + output_size2) // 2)
        self.output_head_2 = nn.Linear(self.d_model, mid_size2)
        self.output_head_3 = nn.Linear(mid_size2, output_size2)
        self.bn2 = nn.BatchNorm1d(mid_size2)

        if model_config.vae_constraint:
            self.vae_constraint = True
            self.one_n_half_size = int(self.d_model * 1.5)
            self.mean_var_pred_0 = nn.Linear(self.d_model, self.one_n_half_size)
            self.mean_var_bn = nn.BatchNorm1d(self.one_n_half_size)
            self.mean_var_pred_1 = nn.Linear(self.one_n_half_size, 2 * self.d_model)

        logger.info(f"Initialized model with decoder type: SslDecoder")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vae_output = None
        if self.vae_constraint:
            bs = x.shape[0]
            x1 = self.mean_var_pred_0(x)
            x2 = self.mean_var_bn(x1.reshape(-1, self.one_n_half_size)).reshape(bs, -1, self.one_n_half_size)
            x3 = self.mean_var_pred_1(x2)
            mu, logvar = x3.chunk(2, dim=-1)
            std_cls = torch.exp(0.5 * logvar[:, 0])
            mu_cls = mu[:, 0]
            eps = torch.randn_like(std_cls)
            x = mu_cls + eps * std_cls
            vae_output = mu, logvar
        else:
            x = x[:, 0, :]
        x = self.dropout(x)
        # y = self.output_head_0(x)
        # y = self.bn0(y)
        # y = self.activation(y)
        # # x = self.dropout(x)
        # y = self.output_head_1(y)

        z = self.output_head_2(x)
        z = self.bn2(z)
        z = self.activation(z)
        z = self.output_head_3(z)

        return z.reshape(z.shape[0], -1, 16), vae_output

        # return y.reshape(y.shape[0], -1, 4), z.reshape(z.shape[0], -1, 16), vae_output


class SslDecoderDoubleCls(nn.Module):
    def __init__(self, model_config: ModelConfig):
        """Decoder for SSL task"""
        super().__init__()

        self.in_encoding_bit = model_config.encoding_width
        self.out_encoding_bit = model_config.encoding_width_output
        self.d_model = model_config.d_model
        # Output size is the number of possible pairs of encodings times the number of bits times the number of labels.
        # self.output_size = int(2**encoding_bit * (2**encoding_bit - 1) * encoding_bit * 4)
        # self.mid_size = int((self.d_model + self.output_size) // 2)
        self.n_cls_token = model_config.n_cls_token

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(0.2)

        # In encoding

        self.output_size_in = 2 ** (self.in_encoding_bit * 2)
        self.mid_size_in = int((self.d_model + self.output_size_in) // 2)

        self.output_head_in_0 = nn.Linear(self.d_model, self.mid_size_in)
        self.output_head_in_1 = nn.Linear(self.mid_size_in, self.output_size_in)
        self.bn_in = nn.BatchNorm1d(self.mid_size_in)

        # Out encoding

        self.output_size_out = 2 * 16 * 60
        self.mid_size_out = int((self.d_model + self.output_size_out) // 2)

        self.output_head_out_0 = nn.Linear(self.d_model, self.mid_size_out)
        self.output_head_out_1 = nn.Linear(self.mid_size_out, self.output_size_out)
        self.bn_out = nn.BatchNorm1d(self.mid_size_out)

        logger.info(f"Initialized model with decoder type: SslDecoderDoubleCls")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(x[:, 0, :])
        y = self.output_head_in_0(y)
        y = self.bn_in(y)
        y = self.activation(y)
        # x = self.dropout(x)
        y = self.output_head_in_1(y)

        z = self.dropout(x[:, 1, :])
        z = self.output_head_out_0(z)
        z = self.bn_out(z)
        z = self.activation(z)
        # x = self.dropout(x)
        z = self.output_head_out_1(z)

        return y.reshape(y.shape[0], -1, 2**self.in_encoding_bit), z.reshape(z.shape[0], 16, -1, 2)
