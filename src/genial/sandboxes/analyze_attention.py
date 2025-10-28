# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

from genial.config.config_dir import ConfigDir

from genial.training.elements.utils import setup_analyzer
from genial.training.elements.configs import DatasetConfig, ModelConfig
from genial.training.elements.models import FullTransformer
from genial.training.elements.lit.models import LitTransformer
from genial.training.elements.datasets import SwactedDesignDatamodule
from genial.training.mains.trainer_v0_to_v3 import EncToScoreTrainer

from pathlib import Path

import torch

from loguru import logger
import matplotlib.pyplot as plt


# Instantiate the dir_config
args_dict = EncToScoreTrainer.parse_args()
dir_config = ConfigDir(is_analysis=True, **args_dict)
device = torch.device(f"cuda:{args_dict.get('device', 0)}" if torch.cuda.is_available() else "cpu")

# Setup some essential parameters
trainer_version_number = args_dict.get("trainer_version_number")
task = ["custom_io_encodings"]

# Setup model configuration from a checkpoint
model_config = ModelConfig(
    dir_config=dir_config,
    trainer_version_number=trainer_version_number,
    task=task,
)
model_config.core_type = "get_attn_weights"

# Setup model
_model = FullTransformer(model_config=model_config)

# Find checkpoint and restore lightning module
checkpoint_metric, load_checkpoint_strategy = EncToScoreTrainer.setup_checkpoint_metric(model_config=model_config)
checkpoint_path, ckpt_epoch = EncToScoreTrainer._find_best_checkpoint(
    dir_config, trainer_version_number, strategy=load_checkpoint_strategy
)
restored_steps = torch.load(checkpoint_path, weights_only=False).get("global_step")
lit_model = LitTransformer.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    meta_model=_model,
    model_config=model_config,
    restored_steps=restored_steps,
    restored_epochs=ckpt_epoch,
)
model = lit_model.transformer
model.to(device)

# Setup Dataset
_analyzer = setup_analyzer(**args_dict)
dataset_config = DatasetConfig(args_dict=args_dict)
datamodule = SwactedDesignDatamodule(dataset_config=dataset_config, analyzer=_analyzer, task=task)
datamodule.setup("validation")

# Get validation dataloader
train_dataloader = datamodule.train_dataloader()


# Setup a pytorch hook that will catch the activation values at the various layers of attention
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        attn_output, attn_weights = output
        self.outputs.append(attn_weights.detach().cpu())

    def clear(self):
        self.outputs = []


def apply_hook_to_model(model: torch.nn.Module, layer_names: list[str]):
    save_output = SaveOutput()
    for name, layer in model.named_modules():
        if "transformer_core.transformer_encoder" in name:
            if name in layer_names:
                layer.register_forward_hook(save_output)
    return save_output


class AttentionRecorder:
    def __init__(self, model, layer_indices):
        """
        Initialize the recorder with a model and the indices of
        the layers you want to monitor.

        Args:
            model: The model containing the transformer layers.
            layer_indices: A list (or single value) of layer indices
                           you want to hook into.
        """
        if isinstance(layer_indices, int):
            layer_indices = [layer_indices]
        self.model = model
        self.layer_indices = layer_indices
        self.recordings = {}  # Will store attn weights keyed by layer index
        self.handles = []

        self.register_hooks()

    def register_hooks(self):
        """
        Register forward hooks for each specified layer index.
        """
        for i in self.layer_indices:
            layer = self.model.transformer_core.transformer_encoder.layers[i].self_attn
            handle = layer.register_forward_hook(self.get_attention_hook(i))
            self.handles.append(handle)

    def get_attention_hook(self, layer_index):
        def hook(module, input, output):
            # output: (attn_output, attn_weights)
            attn_output, attn_weights = output
            # attn_weights = module.state_dict()["in_proj_weight"]
            if attn_weights is not None:
                self.recordings[layer_index] = attn_weights.detach().cpu()
            else:
                pass

        return hook

    def remove_hooks(self):
        """
        Remove all hooks when done to avoid potential memory leaks.
        """
        for h in self.handles:
            h.remove()
        self.handles = []


recorder = AttentionRecorder(model, layer_indices=range(0, model_config.num_layers))

# tgt_layer_names = [f"transformer_core.transformer_encoder.layers.{i}.self_attn.out_proj" for i in range(model_config.num_layers)]
# save_output = apply_hook_to_model(model, tgt_layer_names)
# Run the Model on a Single Sample of the Validation Dataloader
output_dir = Path("output/sandboxes")
samples_iterator = iter(train_dataloader)
for batch in samples_iterator:
    x = batch["encodings"]
    values = batch["values"]
    y_expected = batch["scores"]

    x[0] = x[0].to(device)
    x[1] = x[1].to(device)
    values = values.to(device)
    y_expected = y_expected.to(device)

    y = model(x, values)

    break

for i in range(0, model_config.num_layers):
    matrix = recorder.recordings[i][0]

    # Plot a matrix of weights as a heatmap
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    main_scatter_handles = ax.imshow(matrix)
    ax.set_ylabel(f"seq_length")
    ax.set_xlabel(f"seq_length")
    ax.set_title(f"Attention Weigths Layer {i}")
    cbar = plt.colorbar(main_scatter_handles)

    # Make sure both the x and y axis start at 0
    # ax.set_xlim(0, ax.get_xlim()[1])
    # ax.set_ylim(0, ax.get_ylim()[1])

    plt.legend()
    plt.tight_layout()
    plt.minorticks_on()
    plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
    plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

    figpath = dir_config.analysis_out_dir / f"attention_weights_layer{i}.png"
    plt.savefig(figpath)
    logger.info(f"Saved figure to {figpath}")
    plt.close()


print("Done.")
