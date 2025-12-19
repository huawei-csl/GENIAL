from pathlib import Path
import re
import subprocess

TEMPLATE_PATH = Path("scripts/slurm_scripts/template_flowy.slurm")
OUTPUT_PATH = Path("/home/ramaudruz/tmp_genial_slurm_scripts")
OUTPUT_PATH.mkdir(exist_ok=True)

def write_and_submit(design_numbers, job_name):
    text = TEMPLATE_PATH.read_text()

    # Build replacement string
    design_str = " ".join(design_numbers)
    replacement = f"--design_number_list {design_str}"

    # Replace the existing design_number_list arguments
    text = re.sub(
        r"--design_number_list\s+[\d\s]+",
        replacement,
        text,
    )

    slurm_file = OUTPUT_PATH / f"{job_name}.slurm"
    slurm_file.write_text(text)

    # Submit job
    subprocess.run(["sbatch", str(slurm_file)], check=True)



# Example usage
write_and_submit(
    design_numbers=[
        "00000000000002",
        "00000000000003",
        "00000000000004",
    ],
    job_name="multiplier_chunk_1"
)
