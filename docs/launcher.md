# Launcher

The launcher drives the flow from design generation through synthesis and testing. It can run tasks sequentially or in parallel and is designed to work on local machines or through SLURM.

## Generate and synthesize designs

Run the launcher directly to create designs and execute the synthesis pipeline:

```bash
python src/genial/experiment/task_launcher.py \
  --experiment_name <template> \
  --output_dir_name run1 \
  --nb_workers 24
```

The script submits jobs for generation, synthesis and testing. It supports retry logic and can dispatch work to multiple nodes when SLURM is available.

## Useful command line options

- `--experiment_name`: Template defining initial designs.
- `--output_dir_name`: Directory name where results are stored.
- `--synth_version`: Synthesis tool version.
- `--nb_workers`: Parallel workers per task.
- `--design_number_list`: Restrict work to a subset of design numbers.

## Restarting runs

The launcher tracks completed steps. To continue after an interruption, rerun the command with the same arguments. Use `--force` to redo already finished tasks or `--skip-tests` to bypass selected verification stages.

## Utilities

- `scripts/clean_docker.sh`: stop containers and remove volumes created by the launcher.
- `check_status --experiment_name <template> --output_dir_name <run>`: inspect progress by counting design folders in an output directory.
