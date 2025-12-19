from pathlib import Path
import re
import subprocess
import os

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


des_nums = [f.split('_')[-1] for f in os.listdir('/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/uniform_initial_dir/generation_out')]
des_nums.sort()

for i in range(0, len(des_nums), 2):
    write_and_submit(design_numbers=[des_nums[i], des_nums[i + 1]], job_name=f'flowy_run_{i}')
    if i > 10:
        break
