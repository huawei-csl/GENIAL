from pathlib import Path
import re
from datetime import datetime
import os

# Adjust this path if needed
KEEP_COUNT = 1000

# Regex to match the filename and extract the timestamp for logfiles
LOG_PATTERN = re.compile(r"^genial_flowy_\d+_[a-zA-Z0-9]+_(\d{8}_\d{6})\.log$")

# Regex to match the filename  for scripts
SCRIPT_PATTERN = re.compile(r"^tmp.*\.slurm$")


def extract_timestamp(log_path):
    match = LOG_PATTERN.match(log_path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def main_logs(logdirpath: Path):
    # Get all matching files with timestamps
    logs_with_timestamps = []
    for log_file in logdirpath.iterdir():
        timestamp = extract_timestamp(log_file)
        if timestamp:
            logs_with_timestamps.append((timestamp, log_file))

    # Sort by timestamp descending
    logs_with_timestamps.sort(reverse=True)

    # Keep the newest KEEP_COUNT logs
    logs_to_delete = logs_with_timestamps[KEEP_COUNT:]

    for _, log_file in logs_to_delete:
        try:
            log_file.unlink()
            # print(f"Deleted: {log_file}")
        except Exception as e:
            print(f"Failed to delete {log_file}: {e}")


def main_scripts(script_dirpath: Path):
    # Delete all scripts in script_dirpath
    for script in script_dirpath.iterdir():
        # Make sure the filename matches with the script pattern
        match = SCRIPT_PATTERN.match(script.name)
        if match:
            try:
                script.unlink()
                # print(f"Deleted: {script}")
            except Exception as e:
                print(f"Failed to delete {script}: {e}")


if __name__ == "__main__":
    log_dirpath = Path(f"{os.getenv('HOME')}/slurm_logs/genial/sbatch_error")
    assert log_dirpath.exists(), f"Directory {log_dirpath} does not exist"
    main_logs(log_dirpath)

    log_dirpath = Path(f"{os.getenv('HOME')}/slurm_logs/genial/sbatch_info")
    assert log_dirpath.exists(), f"Directory {log_dirpath} does not exist"
    main_logs(log_dirpath)

    scripts_dirpath = Path(f"{os.getenv('HOME')}/tmp_genial_slurm_scripts")
    assert scripts_dirpath.exists(), f"Directory {scripts_dirpath} does not exist"
    main_scripts(scripts_dirpath)
