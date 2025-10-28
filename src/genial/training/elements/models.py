# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).


import torch
from torch import nn

from abc import ABCMeta

from genial.training.elements.configs import ModelConfig
from genial.training.elements.modules import embeddings, transformer_cores, decoders

from loguru import logger
import traceback


# our version of ABCMeta with required attributes
class MetaModel(ABCMeta):
    required_attributes = []

    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        for attr_name in obj.required_attributes:
            if not getattr(obj, attr_name):
                raise ValueError("required attribute (%s) not set" % attr_name)
        return obj


class AbstractModel(nn.Module, metaclass=MetaModel):
    """
    Base class for all models.
    """

    required_attributes = [
        "embedding",
        "transformer_core",
        "decoder",
    ]

    embedding_type_mapping = {
        "default": embeddings.PointNetEmbedding,
        "pointnetv2": embeddings.PointNetEmbeddingV2,
        "embedding_instead_pointnet": embeddings.EmbeddingInsteadPointNet,
    }

    transformer_type_mapping = {
        "default": lambda model_config: transformer_cores.TransformerEncoder(model_config, encoder_type="default"),
        "diff_transformer": lambda model_config: transformer_cores.TransformerEncoder(
            model_config, encoder_type="diff_attention"
        ),
        "get_attn_weights": lambda model_config: transformer_cores.TransformerEncoder(
            model_config, encoder_type="get_attn_weights"
        ),
    }

    decoder_type_mapping = {
        "default": decoders.LinearDecoder,
        "attention": decoders.AttentionDecoder,
        "multi_target_linear": decoders.MultiTargetLinearDecoder,
        "reducing": decoders.ReducingLinearDecoder,
        "pointnet": decoders.PointNetLinearDecoder,
        "ssl": decoders.SslDecoder,
        "ssl_double_cls": decoders.SslDecoderDoubleCls,
    }


class FullTransformer(AbstractModel):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()

        self.embedding = AbstractModel.embedding_type_mapping[model_config.embedding_type](model_config=model_config)
        self.transformer_core = AbstractModel.transformer_type_mapping[model_config.core_type](
            model_config=model_config
        )
        self.decoder = AbstractModel.decoder_type_mapping[model_config.decoder_type](model_config=model_config)

        if model_config.final_layer_type == "identity":
            self.final_layer = torch.nn.Identity()
        elif model_config.final_layer_type == "tanh":
            self.final_layer = torch.nn.Tanh()

        self.initialize_weights(model_config)

    def forward(
        self, x: torch.Tensor, values: torch.Tensor, embedding_input: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embedding(x, values)
        x = self.transformer_core(x)
        x = self.decoder(x)

        if isinstance(x, tuple):
            x = self.final_layer(x[0]), x[1]
        else:
            x = self.final_layer(x), None

        return x

    def initialize_weights(self, model_config: ModelConfig) -> None:
        # Initialization based on : https://github.com/layer6ai-labs/T-Fixup
        for name, module in self.named_modules():
            if hasattr(module, "weight") and module.weight.dim() > 1:
                if name == "decoder.output_head_3":
                    if model_config.final_layer_type == "tanh":
                        nn.init.xavier_uniform_(module.weight.data, gain=nn.init.calculate_gain("tanh"))
                    else:
                        nn.init.xavier_uniform_(module.weight.data)
                else:  # Assuming other layers are using GeLU
                    nn.init.normal_(module.weight.data, 0, 0.02)  # He initialization with slight variance

            if hasattr(module, "bias"):
                try:
                    nn.init.zeros_(module.bias.data)
                except Exception:
                    logger.error(module)
                    logger.error(traceback.format_exc())

            if isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight.data, 1)
                nn.init.constant_(module.bias.data, 0)
