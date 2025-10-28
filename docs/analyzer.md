# Analyzer

The analyzer processes generated designs to extract statistics and produce reports. It can be launched after synthesis or run independently on existing datasets.

## Basic usage

```bash
python src/genial/experiment/task_analyzer.py \
  --experiment_name <template> \
  --output_dir_name run1 \
  --nb_workers 16
```

The analyzer reads design metadata, computes switching activity and other metrics, and stores results in the output directory. When used with SLURM the dispatcher schedules analysis jobs on CPU nodes.

## Analysis modes

- **Launch full**: analyze everything in a new output directory.
- **Launch partial**: analyze a specific range of design numbers.
- **Continue analysis**: resume from the last processed design after an interruption.
- **Bulk mode**: process data generated separately from the loop pipeline.

## Interactive mode

To explore designs manually, run the analyzer in interactive mode:

```bash
python src/genial/experiment/task_analyzer.py \
  --experiment_name <template> \
  --output_dir_name run1 \
  --interactive
```

This opens interactive plotting windows for deeper inspection.
