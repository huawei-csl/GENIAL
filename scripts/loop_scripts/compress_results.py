#!/usr/bin/env python3
import argparse
import re
import tarfile
from pathlib import Path
import numpy as np
import os


def find_pqt_files(root: Path):
    all_files = []
    # Adjust patterns as needed
    pattern = re.compile(r"flowy_run_12chains_3000steps_(proto_iter.*|gen_iter0)$")

    # Depth-1 directories only
    matches = [p for p in root.iterdir() if p.is_dir() and pattern.fullmatch(p.name)]

    all_search_folders = matches

    for iter_dir in all_search_folders:
        analysis_dir = iter_dir / "analysis_out"
        if analysis_dir.is_dir():
            pqt_pattern = re.compile(r".*\.pqt$")
            pqt_files = [p for p in analysis_dir.iterdir() if p.is_file() and pqt_pattern.fullmatch(p.name)]
            all_files.extend(pqt_files)
    print(all_files)

    # Get checkpoint paths
    checkpoints_folder = root / "flowy_run_12chains_3000steps_saved_ckpts"
    for iteration_dirpath in checkpoints_folder.iterdir():
        checkpoint_dirpath = iteration_dirpath / "checkpoints"
        if checkpoint_dirpath.is_dir():
            checkpoint_files = [p for p in checkpoint_dirpath.iterdir() if p.is_file()]
            validation_loss_values = [float(p.name.split("_")[1]) for p in checkpoint_files]
            best_checkpoint = checkpoint_files[np.argmin(validation_loss_values)]
            all_files.append(best_checkpoint)

    yield from all_files


def create_tarball(output_tar: Path, files, root: Path):
    with tarfile.open(output_tar, "w:gz") as tar:
        for f in files:
            print(f"Processing {f}")
            # Store paths relative to the root for clean extraction
            arcname = f.relative_to(root)
            tar.add(f, arcname=arcname)


def main():
    parser = argparse.ArgumentParser(
        description="Collect *.pqt files from '*_iter0/analysis_out' directories into a tarball."
    )
    parser.add_argument(
        "-r",
        "--root",
        type=Path,
        default=Path(f"{os.environ.get('WORK_DIR')}/output/multiplier_4bi_8bo_permuti_flowy"),
        help="Root directory to search (default: current directory)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("pqt_files.tar.gz"),
        help="Output tarball file name (default: pqt_files.tar.gz)",
    )
    args = parser.parse_args()

    files = list(find_pqt_files(args.root))
    if not files:
        print("No .pqt files found in '*_iter0/analysis_out' directories.")
        return

    create_tarball(args.output, files, args.root)
    print(f"Created tarball: {args.output} with {len(files)} files.")


if __name__ == "__main__":
    main()
