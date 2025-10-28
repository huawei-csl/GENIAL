# Recommender and Network Inversion

The recommender proposes new designs based on model predictions. It can also perform network inversion to generate prototypes that achieve target metrics.

## Launch recommender

```bash
python src/genial/experiment/task_recommender.py \
  --experiment_name <template> \
  --output_dir_name run1 \
  --trainer_version_number 0 \
  --nb_new_designs 20 \
  --keep_percentage 10
```

The recommender scores existing designs, selects candidates, and may call the generator to synthesize new designs.

## Network inversion

Given a trained network, you can generate prototypes that optimize a target score:

```bash
python src/genial/utils/prototype_generator_v2.py \
  --experiment_name <template> \
  --output_dir_name run1 \
  --dst_output_dir_name prototypes \
  --trainer_version_number 0 \
  --nb_gener_prototypes 1000 \
  --batch_size 256 \
  --device 0 \
  --yml_config_path model_configs/8b_default.yml \
  --score_type power \
  --do_prototype_pattern_gen
```

This produces Verilog prototypes and logs describing the optimization process.

## Prototype generation utilities

- `run_full_prototype_generation_loop.sh`: orchestrate recommendation and generation in one command.
- `run_full_prototype_generation_loop_*`: convenience scripts for common experiment settings.
