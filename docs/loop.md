# Loop Execution

The loop pipeline stitches together design generation, synthesis, analysis, training and recommendation into repeated iterations. `full_run_v2.py` reads a YAML configuration describing the experiment and runs the stages sequentially or via SLURM.

## Configuration files

Ready-to-use YAML files live in `scripts/loop_scripts/configs`. Start from `flowy_loop_mixed_default.yaml` for a standard flow or `power_loop.yaml` to optimise for power. Each file specifies the experiment name, number of iterations and hyper‑parameters for training and recommendation. Copy one of these files and adjust values such as `nb_workers` or `n_iter` to create your own setup.

## Run locally

To execute the loop on a single machine:

```bash
python scripts/loop_scripts/full_run_v2.py \
  --config_filepath scripts/loop_scripts/configs/flowy_loop_mixed_default.yaml
```

This command sequentially launches generation, training, recommendation and analysis using local resources. Results are written under the directory defined in the configuration.

## Run with SLURM

On clusters with SLURM, append `--is_slurm` to delegate each stage to the scheduler:

```bash
python scripts/loop_scripts/full_run_v2.py \
  --config_filepath scripts/loop_scripts/configs/flowy_loop_mixed_default.yaml \
  --is_slurm
```

With this flag, the script submits jobs for each stage and collects their logs. Ensure `WORK_DIR` points to a scratch directory visible to the compute nodes.

## Additional options

- `--skip_init_gener`: start from existing generated designs instead of creating new ones.
- `--is_control_run`: disable the recommender for a random generation baseline.
- `--resume_from {auto|proto|merge|none}`: control resume behavior; default `auto` detects where to restart, `proto` resumes from latest prototype generation (skips retraining), `merge` resumes after the latest merge (reanalyzes then continues), `none` starts fresh ignoring existing outputs.

Housekeeping
- Previous merge directories are deleted by default to save disk space. Pass `--KEEP_merge_dirs` (alias: `--keep_merge_dirs`, `--skip_merge_dirs`) to preserve them. The old `--delete_merge_dirs` flag is deprecated.

## Resume Behavior

- Modes (`--resume_from`):
  - `auto` (default): scans the experiment directory for `proto_*` and `merge_*` to infer where to resume.
  - `proto`: force resume from the latest `proto_*` iteration, subject to token check below.
  - `merge`: force resume “after merge” at the latest `merge_*` (reanalyzes that merge and continues).
  - `none`: disables auto-resume entirely; runs from configuration as if starting fresh.

- Auto-detection algorithm:
  - Compute the highest iteration seen for `proto_*` and `merge_*` within the experiment’s output directory.
  - If no `proto_*` directories exist, start fresh (no resume).
  - If latest `merge` iteration equals latest `proto` iteration `N`:
    - Treat iteration `N` as complete. Reanalyze `merge_iterN` to refresh DB/plots, then continue with `N+1`.
  - If latest `proto` iteration `N` is newer than latest `merge`:
    - Attempt resume at “proto” for `N`. See token rule below for exact behavior.

- Prototype completion token:
  - The recommender writes a token file when prototype generation finishes successfully:
    - Path: `<proto_dir>/proto_generation_done.token`
    - Contents: JSON with `timestamp`, `nb_generated`, `status`.
  - On resume at “proto”:
    - If the token exists: proceed with launch → analyze → merge for that iteration.
    - If the token is missing (e.g., recommender failed and the directory is empty/partial):
      - Only prototype generation is re-run (training is NOT re-run), then launch → analyze → merge.

- Initial dataset resume:
  - If generation exists for the initial dataset but synthesis is incomplete, the script resumes the initial launch to complete synthesis (unless `--resume_from none`).
  - If the initial dataset is fully synthesized, the initial launch is skipped by default.

- Interaction with `--skip_restart_launch`:
  - When resuming from “proto”, `--skip_restart_launch` skips launching flows for existing prototypes.
  - Analysis of the `proto_*` directory still runs to refresh the DB and plots, then the iteration proceeds.

### Examples


- Resume automatically from where the last run stopped:
  - `python scripts/loop_scripts/full_run_v2.py --config_filepath scripts/loop_scripts/configs/flowy_loop_default.yaml`

- Force resume from latest prototype stage (skip retraining):
  - `python scripts/loop_scripts/full_run_v2.py --config_filepath scripts/loop_scripts/configs/flowy_loop_default.yaml --resume_from proto`

- Resume after the latest merge (reanalyze and continue):
  - `python scripts/loop_scripts/full_run_v2.py --config_filepath scripts/loop_scripts/configs/flowy_loop_default.yaml --resume_from merge`

- Start fresh ignoring existing outputs:
  - `python scripts/loop_scripts/full_run_v2.py --config_filepath scripts/loop_scripts/configs/flowy_loop_default.yaml --resume_from none`

Resume notes
- If the initial dataset’s generation exists but synthesis is incomplete, the script resumes the initial launch automatically (unless `--resume_from none`).


## Some Useful Configurations
### Standard Flowy with Transistor Selection and Final SwAct Evaluation

- Run the loop:
    - `run_genial_loop --config_filepath $SRC_DIR/scripts/loop_scripts/configs/flowy_loop_standard_emb_transistors_swact.yaml --do_init_gener_n_launch --delete_merge_dirs [--is_slurm]`
- Analyze the results:
    - `analyze_genial_loop --run standard_flowy_transistors_swact`