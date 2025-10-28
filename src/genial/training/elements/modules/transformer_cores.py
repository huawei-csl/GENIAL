# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

from loguru import logger
from types import SimpleNamespace
from typing import Optional

import matplotlib.pyplot as plt

import torch
from torch import nn

from genial.training.elements.configs import ModelConfig

try:
    import genial.training.elements.modules.ext.diff_transformer.multihead_diffattn as multihead_diffattn
except ImportError:

    class MultiheadDiffAttn:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def MultiheadDiffAttnFixedLambda(self, *args, **kwargs):
            pass

    multihead_diffattn = MultiheadDiffAttn


class TransformerEncoderLayerDiff(nn.TransformerEncoderLayer):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        max_sequence_length = kwargs.pop("max_sequence_length")

        super().__init__(*args, **kwargs)

        diff_attn_kwargs = {
            "args": SimpleNamespace(
                **{
                    "model_parallel_size": 2,
                    "decoder_kv_attention_heads": None,
                }
            ),
            "embed_dim": kwargs.get("d_model"),
            "depth": 10,  # Will be ignore by the MultiheadDiffAttnFixedLambda implementation # TODO: that should be the depth of the layer
            "num_heads": kwargs.get("nhead"),
            "lambda_value": 0.8,  # Only used by the MultiheadDiffAttnFixedLambda implementation # TODO: that should not be fixed
        }

        self.self_attn = multihead_diffattn.MultiheadDiffAttnFixedLambda(**diff_attn_kwargs)

        seqlen_ro = max_sequence_length
        rotary_dim = kwargs.get("nhead")
        cos = torch.ones((seqlen_ro, rotary_dim // 2))
        sin = torch.zeros((seqlen_ro, rotary_dim // 2))
        self.rel_pos = (cos, sin)

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.self_attn(
            x,
            [val.to(x.device) for val in self.rel_pos],
            attn_mask=attn_mask,
        )
        return self.dropout1(x)


class GetWeightsEncoderLayer(nn.TransformerEncoderLayer):
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask[: x.shape[1], : x.shape[1]],
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)


class TransformerEncoder(nn.Module):
    __valid_mask_types__ = ["skewed_subsequent", "custom_io_encodings", "none", None]

    def __init__(self, model_config: ModelConfig, encoder_type="default") -> None:
        super().__init__()

        self.nhead = model_config.nhead
        self.n_cls_token = model_config.n_cls_token

        encoder_layer_kwargs = {
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

        # Configure Sequence Length
        max_sequence_length = model_config.max_sequence_length
        if self.n_cls_token:
            max_sequence_length += self.n_cls_token
        self.max_sequence_length = max_sequence_length

        self.encoder_type = encoder_type
        if encoder_type == "default":
            encoder_layer = nn.TransformerEncoderLayer(**encoder_layer_kwargs)
        elif encoder_type == "get_attn_weights":
            encoder_layer = GetWeightsEncoderLayer(**encoder_layer_kwargs)
        elif encoder_type == "diff_attention":
            encoder_layer_kwargs.update(
                {
                    "max_sequence_length": self.max_sequence_length,
                }
            )
            encoder_layer = TransformerEncoderLayerDiff(
                **encoder_layer_kwargs,
            )  # TODO: Make CLS Token Compatible with Diff Transformer
        else:
            raise NotImplementedError(f"encoder_type specified: {encoder_type} is not Implemented")

        logger.info(f"Transformer model initiated with encoder type: {type(encoder_layer)}")

        # Instantiate the Self Attention (Encoder) Model
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=model_config.num_layers,
        )

        # Configure Mask
        if model_config.mask_type == "skewed_subsequent":
            self.mask = TransformerEncoder._generate_square_subsequent_mask_skewed(self.max_sequence_length)
        elif model_config.mask_type == "custom_io_encodings":
            self.mask = TransformerEncoder._custom_io_encodings_mask(
                model_config, self.max_sequence_length, n_cls_token=self.n_cls_token
            )
        else:
            self.mask = None
        logger.warning(f"Mask: Attention `mask_type` has been set to {model_config.mask_type}.")

    @staticmethod
    def _generate_square_subsequent_mask_skewed(sz):
        """
        Generates a mask where first and second tokens can see each other fronm the start.
        Similar to NLP mask but shifted by one through "time" (token sequence direction).
        """

        mask = (torch.triu(torch.ones(sz - 1, sz - 1)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        # Desired padding (for skewing the mask)
        padding = (1, 0, 0, 1)  # (left, right, top, bottom)
        mask = torch.nn.functional.pad(mask, padding, mode="constant", value=0)

        mask = mask.requires_grad_(False)
        return mask

    @staticmethod
    def _custom_io_encodings_mask(model_config: ModelConfig, max_sequence_length: int, n_cls_token: int = 1):
        """
        Generate a mask where the input and output encodings token cannot see each other.
        But the CLS token can.
        """
        # max_seq_length = <cls_token> + <max_nb_in_values> + <max_nb_out_values>
        # <max_nb_out_values> is defined based on the bitwidth, not the actual number of values
        start_idx_output_tokens = 2**model_config.encoding_width + n_cls_token
        mask = torch.ones((max_sequence_length, max_sequence_length))

        # Make sure input and output tokens cannot see each others
        mask[start_idx_output_tokens:, :start_idx_output_tokens] = 0
        mask[:start_idx_output_tokens, start_idx_output_tokens:] = 0

        # Allow the CLS token to see all over tokens
        mask[0, :] = 1

        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))

        return mask

    def _plot_mask(self):
        if self.mask is not None:
            fig, ax = plt.subplots(1, 1)
            im = ax.imshow(self.mask)
            plt.colorbar(im)
            plt.savefig("debug/mask.png")
            logger.info(f"Mask saved at debug/mask.png")
            plt.close()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask is not None:
            # _mask = self.mask.repeat((x.shape[0]*self.nhead, 1, 1)).to(x.device)
            _mask = self.mask[: x.shape[1], : x.shape[1]].to(x.device)
        else:
            _mask = None

        x = self.transformer_encoder(x, mask=_mask)
        return x
