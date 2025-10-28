# Training

Training learns a model to score or generate circuit designs. The workflow uses PyTorch Lightning configurations defined in
`model_configs/` and is driven by `trainer_enc_to_score_value.py`.

## Train from scratch

```bash
python src/genial/training/mains/trainer_enc_to_score_value.py \
  --experiment_name <template> \
  --output_dir_name training_run \
  --yml_config_path model_configs/8b_default.yml \
  --trainer_task enc_to_score
```

Adjust `batch_size`, `max_epochs`, `score_type`, and other arguments as required.

## Self-supervised pretraining

```bash
python src/genial/training/mains/trainer_enc_to_score_value.py \
  --experiment_name <template> \
  --output_dir_name pretrain_run \
  --yml_config_path model_configs/8b_default.yml \
  --trainer_task ssl
```

The run directory will contain `trainer_out/model_config.yml`. Copy this file to `model_configs/` and tweak any hyperparameters
needed for fine‑tuning.

## Fine‑tuning from SSL

```bash
python src/genial/training/mains/trainer_enc_to_score_value.py \
  --experiment_name <template> \
  --output_dir_name finetune_run \
  --yml_config_path model_configs/my_finetune.yml \
  --model_checkpoint_path <path_to_ssl_checkpoint> \
  --trainer_task finetune_from_ssl
```

Replace `<template>` and the configuration file with your own choices. Setting `--trainer_task` to other supported values such
as `enc_to_score` or `synthv0_to_synthv3` allows fine‑tuning for different objectives.

## Model configuration

Sample configuration files live in the `model_configs/` directory. Each file defines the architecture, optimizer,
learning‑rate scheduler and training hyperparameters.

