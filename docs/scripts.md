# Utility Scripts

Several helper scripts are provided under the `scripts/` directory.

## Docker cleanup

`clean_docker.sh` stops all containers started by the launcher and removes associated volumes:

```bash
./scripts/clean_docker.sh
```

Use this after interrupting a run to free disk space and avoid conflicts.

## Compress and transfer results

To archive output directories:

```bash
./scripts/compress_output_folder.sh <folder> <archive.tar.gz>
```
This produces `archive.tar.gz`, which you can move to another machine or store for backup. Add `--uncompress` to extract an archive.

## File compression utilities

`compress_file` and `decompress_file` compress or expand individual Verilog files:

```bash
decompress_file --filepath <relative_path_to_file>
compress_file --filepath <relative_path_to_file>
```

Use these to shrink or expand single design files when working outside the launcher pipeline.

## Check run status

Count how many design folders exist in an output directory:

```bash
check_status --experiment_name <template> --output_dir_name <run>
```
The command prints how many design subfolders exist, giving a quick snapshot of loop progress.

## Liberty to capacitance

Build a JSON mapping each cell to its total input pin capacitance from a
Liberty file (supports plain `.lib` and gzip-compressed `.lib.gz`):

```bash
scripts/liberty_to_capacitance.py <library.lib[.gz]> cell_caps.json
```

The resulting file can be passed to the analyzer when constructing weighted
switching activity models.

Extras:
- Single cell query:
  - `scripts/liberty_to_capacitance.py --cell-name <CELL> <library.lib[.gz]> [out.json]`
- Generic type query (e.g., BUF, NOT, AND, NAND, OR, NOR, XOR, XNOR, MUX, NMUX, AOI3, OAI3, AOI4, OAI4, DFF_PN0):
  - `scripts/liberty_to_capacitance.py --cell-type <TYPE> <library.lib[.gz]> [out.json]`
  - `scripts/liberty_to_capacitance.py --list-type <TYPE> <library.lib[.gz]> [out.json]` (dump all matches)
- Dump a generic mapping like the Yosys "notech" model (best representative per type):
  - `scripts/liberty_to_capacitance.py --dump-generic <library.lib[.gz]> generic_caps.json`

## SwAct calibration

Fit a linear regression between weighted switching activity and measured
power. The input CSV must contain `swact` and `power` columns:

```bash
scripts/swact_calibration.py data.csv
```

The script prints the coefficients `a` and `b` for `power ≈ a * swact + b`.

## Generate design from encoding dict

Build a single design from a given encoding dictionary and place it under the current run’s `generation_out`.

```bash
# Console script
generate_design \
  --experiment_name <template_name> \
  [--output_dir_name <run_name>] \
  --enc_dict_path <path/to/enc.{json|v|v.bz2}>

# Equivalent module form
python -m genial.tools.generate_design_from_encoding_dict \
  --experiment_name <template_name> \
  [--output_dir_name <run_name>] \
  --enc_dict_path <path/to/enc.{json|v|v.bz2}>
```

- Requires an active environment (installed package) and `.env` with `SRC_DIR` and `WORK_DIR` set.
- `--experiment_name` must match a folder under `src/genial/templates_and_launch_scripts/`.
- The output goes to `output/<experiment>/<run>/generation_out/res_<N>/hdl/mydesign_comb.v`.
- Input formats:
  - `.v` or `.v.bz2` containing encodings as comments (automatically extracted), or
  - `.json` with keys `input` and optionally `output`, mapping integer values to bit representations, for example:

```json
{
  "input": { "0": [0,0], "1": [0,1] },
  "output": { "0": [0], "1": [1] }
}
```

If `output` is omitted, the script derives a default output encoding from the experiment configuration.

## Visualize encoding

Render a heatmap of the input or output encoding from a design or JSON file and save it next to the input as `encoding_representation.png`.

```bash
visualize_encoding \
  -f <path/to/design_or_json> \
  [-t input|output] \
  [-b <bitwidth>]
```

- `-t/--encoding_type`: which encoding to visualize (`input` default, or `output`).
- `-b/--bitwidth`: number of bits to display on the axes (default `4`).
- Works with `.v`, `.v.bz2`, or `.json` encodings (same format expectations as above). No experiment setup required.

## Task launcher

Run generation, synthesis, testing, and power extraction.

```bash
task_launcher \
  --experiment_name <template> \
  [--output_dir_name <run>] \
  [--do_gener | --only_gener] \
  [--nb_new_designs N] \
  [--skip_already_synth] [--skip_already_swact] [--skip_already_power] \
  [--restart] [--design_number_list <dn1> <dn2> ...] [--nb_workers K]
```

- `--do_gener`: generate new LUT designs; `--nb_new_designs` overrides count.
- `--only_gener`: only generate LUTs, skip synth/test/power.
- Uses `.env` (`SRC_DIR`, `WORK_DIR`) and experiment templates.

## Task analyzer

Analyze results and produce databases and plots under `analysis_out`.

```bash
task_analyzer \
  --experiment_name <template> --output_dir_name <run> \
  [--cell_cost_model transistor|capacitance|capacitance_calibrated|none] \
  [--technology <tech>] \
  [--rebuild_db | --continue] [--skip_plots] [--fast_plots] \
  [--skip_fullsweep_analysis] [--interactive] \
  [--skip_tests_list <t1> <t2> ...]
```

- `--rebuild_db`: rebuild analysis databases from scratch.
- `--continue`: append new designs only; keeps existing DBs.
- Plots are saved in `analysis_out/plots`.
- `--cell_cost_model`: select transistor, capacitance, capacitance_calibrated or none for weighting.
- `--technology`: choose which library under `resources/libraries` provides the cost model data.

## Find classic encoding

Identify the design number with classic (unpermuted) encoding.

```bash
task_analyzer_find_classic_encoding_design_number \
  --experiment_name <template> --output_dir_name <run>
```

- Logs the design number, writes `classic_encoding.v`, and updates `special_designs.json`.

## Plot swact vs ncells

Generate a scatter plot of switching activity versus number of cells.

```bash
plot_swact_ncells --experiment_name <template> --output_dir_name <run>
```

- Saves figures under `analysis_out/plots` for the run.

## Trainer

Train or evaluate the encoding-to-score model. Logs to `trainer_out/lightning_logs/version_<N>` and writes `training.db.pqt`.

```bash
task_trainer \
  --experiment_name <template> --output_dir_name <run> \
  [--trainer_task enc_to_score] [--device <id>] [--fast_dev_run] \
  [--max_epochs N] [--check_val_every_n_epoch M] \
  [--score_type trans] [--score_rescale_mode minmax] \
  [--trainer_version_number V] [--yml_config_path <config.yml>]
```

- Typical: set `--device`, `--max_epochs`, `--score_*`, and optionally a YAML config.
- For finetune/inference, provide `--trainer_version_number` or `--model_checkpoint_path` as needed.

## Loop runner

End-to-end loop (generate → launch → analyze → train → recommend) using a preset config.

```bash
run_looper \
  --config <debug|mult4b_trans_minmax|mult4b_trans_standardize|mult4b_unsigned_trans_minmax> \
  [--output_dir_name <run>] [--device <id>] [--trainer_version_number V] \
  [--skip_init] [--skip_init_analysis] [--skip_init_launching]
```

- Uses presets defined in `src/genial/encoding_explorer_loop.py`; `debug` is a good starting point.
- `--skip_init*` flags start from existing data; set `--trainer_version_number` when skipping init training.

## Merge output directories

Merge two runs from the same experiment into a new output directory, deduplicating by encoding.

```bash
merge_out_dirs \
  --experiment_name <template> \
  --output_dir_name_0 <runA> --output_dir_name_1 <runB> \
  [--dst_output_dir_name <merged>] [--frac_0 F0] [--frac_1 F1] [--dry_run] [--force]
```

- Writes mapping to `<merged>/merge_info.pqt`; copies `generation_out`, `synth_out`, `test_out`, and analysis files.

## Get complexity

Rebuild LUT from a design or JSON and print the gate-level complexity metrics.

```bash
get_complexity \
  --experiment_name <template> --output_dir_name <run> \
  --design_filepath <path/to/mydesign_comb.{v|v.bz2|json}>
```

- Prints a complexity dictionary to stdout/logs; uses experiment config to infer encoding and bitwidths.
