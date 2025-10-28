import os
import shutil
from pathlib import Path

from argparse import ArgumentParser


# Configuration
def main():
    args = ArgumentParser()
    args.add_argument("-d", "--dry-run", action="store_true", help="Dry run (no deletion)")
    args.add_argument("-e", "--experiment_name", type=str, help="Experiment name")
    args = args.parse_args()
    ROOT_DIR = Path(f"{os.getenv('WORK_DIR')}") / "output" / args.experiment_name
    assert ROOT_DIR.exists(), f"Root directory {ROOT_DIR} does not exist"

    TARGET_NAMES = {
        "synth_out",
        "power_out",
        "recommender_out",
        "test_out",
        "trainer_out",
        "logs",
        "analysis_out/swact_analysis",
    }  # names of folders to delete
    DRY_RUN = args.dry_run  # set True to preview without deleting

    delete_matching_dirs(ROOT_DIR, TARGET_NAMES, max_depth=1, dry_run=DRY_RUN)


def delete_matching_dirs(root_dir: str, target_names: set[str], max_depth: int = 2, dry_run: bool = False) -> None:
    """
    Delete directories whose base name is in TARGET_NAMES within root_dir,
    searching up to max_depth levels below root_dir (inclusive of root level = 0).

    Example depths:
      depth 0: root_dir itself
      depth 1: immediate children of root_dir
      depth 2: grandchildren of root_dir

    Args:
        root_dir: Root directory to search.
        target_names: Set of directory names to delete.
        max_depth: Maximum search depth relative to root (default 2).
        dry_run: If True, only print what would be deleted.
    """
    root_path = Path(root_dir).resolve()

    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"Root directory does not exist or is not a directory: {root_dir}")

    # Normalize target names for case-sensitive match; adjust if you need case-insensitive.
    target_names = set(target_names)

    # Walk the tree top-down so we can prevent descending into deleted dirs if desired.
    for current_root, dirnames, _filenames in os.walk(root_path, topdown=True):
        current_path = Path(current_root)
        # Compute depth: number of parts beyond the root
        depth = len(current_path.relative_to(root_path).parts)  # 0 for root

        # Stop descending if we've reached max_depth: we still want to process current level,
        # but we shouldn't go deeper, so clear dirnames.
        if depth >= max_depth:
            # We still handle deletions in dirnames at this depth, but prevent deeper traversal
            # by clearing after we process deletions here.
            pass

        # Collect directories to delete at this level
        to_delete = [d for d in dirnames if d in target_names]

        # Perform deletions (or dry-run)
        for d in to_delete:
            target_path = current_path / d
            if target_path.is_dir():
                if dry_run:
                    print(f"[DRY RUN] Would delete: {target_path}")
                else:
                    print(f"Deleting: {target_path}")
                    shutil.rmtree(target_path, ignore_errors=True)
            # Remove from dirnames so os.walk doesn't try to descend into it
            # (important even in dry_run to avoid unnecessary traversal).
            # Use try/except in case duplicates aren't present.
            try:
                dirnames.remove(d)
            except ValueError:
                pass

        # If we've reached max depth, prevent descending into any remaining subdirs
        if depth >= max_depth:
            dirnames.clear()


if __name__ == "__main__":
    main()
