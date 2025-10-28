import subprocess
import tempfile
import time
from pathlib import Path
import os

from string import Template


class SlurmGPUFinder:
    JOB_SCRIPT_TEMPLATE = Template("""#!/bin/bash
#SBATCH --job-name=check_free_gpus_${node}
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=${node}
#SBATCH --gres=gpu:1
#SBATCH --partition=${partition}
#SBATCH --output=${out_file}

echo "Checking free GPUs on node: $$(hostname)"
free_gpus=()
num_gpus=$$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
for gpu in $$(seq 0 $$((num_gpus-1))); do
    # List all running PIDs on this GPU
    pids=$$(nvidia-smi -i $$gpu --query-compute-apps=pid --format=csv,noheader)
    if [ -z "$$pids" ]; then
        free_gpus+=($$gpu)
    fi
done
echo "FREE_GPUS: $${free_gpus[@]}"
""")

    def __init__(self, partition="VNL_GPU"):
        self.partition = partition

    def get_gpu_nodes(self):
        # Get all nodes, filter by partition and GRES=gpu
        nodes_info = subprocess.check_output(["scontrol", "show", "nodes"], encoding="utf-8")
        # Split output into node blocks
        node_blocks = nodes_info.strip().split("\n\n")
        nodes = []
        for block in node_blocks:
            if f"{self.partition}" not in block:
                # print(f"Skipping node: {block.splitlines()[0]} because of partition")
                continue
            if "Gres=gpu:" not in block:
                # print(f"Skipping node: {block.splitlines()[0]} because not gpu available")
                continue
            name_line = [line for line in block.split("\n") if line.startswith("NodeName=")]
            if not name_line:
                continue
            name = name_line[0].split("NodeName=")[1].split()[0]
            # Find Gres=gpu:x (x is number of GPUs)
            for field in block.split():
                if field.startswith("Gres=gpu:"):
                    try:
                        n_gpus = int(field.split(":")[2])
                    except IndexError:
                        n_gpus = 1  # fallback: assume at least 1 GPU
                    nodes.append((name, n_gpus))
                    break
        # print(nodes)
        return nodes

    def submit_check_job(self, node, tmp_dir):
        # print(tmp_dir)
        # Put logs in a directory on $HOME (shared filesystem)
        out_file = Path.home() / "slurm_logs" / "gpu_checks" / f"check_{node}.out"
        # err_file = Path.home() / "slurm_logs" / f"check_{node}.error.out"
        out_file.parent.mkdir(exist_ok=True)
        script_content = self.JOB_SCRIPT_TEMPLATE.substitute(node=node, partition=self.partition, out_file=out_file)
        script_path = tmp_dir / f"job_{node}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)
        # Make the script executable
        os.chmod(script_path, 0o755)
        sbatch_out = subprocess.check_output(["sbatch", str(script_path)], encoding="utf-8")
        job_id = int([x for x in sbatch_out.split() if x.isdigit()][0])
        return job_id, out_file

    def wait_job(self, job_id):
        # Wait for job to finish
        try:
            squeue = subprocess.check_output(["squeue", "-j", str(job_id), "-h"], encoding="utf-8")
            if not squeue.strip():
                return 0
        except Exception:
            return 0
        return 1

    def parse_output(self, out_file):
        try:
            with open(out_file) as f:
                lines = f.readlines()
            free_gpu_line = [line for line in lines if line.startswith("FREE_GPUS:")]
            if free_gpu_line:
                gpu_str = free_gpu_line[0].strip().replace("FREE_GPUS:", "").strip()
                if gpu_str:
                    free_gpus = [int(x) for x in gpu_str.split()]
                else:
                    free_gpus = []
                return free_gpus
        except Exception:
            pass
        return []

    def find_node_with_free_gpu(self, timeout: int = 45):
        start_time = time.time()
        jobs = []
        result = {}
        with tempfile.TemporaryDirectory() as tmpd:
            tmp_dir = Path(tmpd)
            nodes = self.get_gpu_nodes()
            if not nodes:
                print("No GPU nodes found in partition.")
                return result
            print(f"Checking free GPUs on nodes: {[n for n, _ in nodes]}")

            for node, _ in nodes:
                job_id, out_file = self.submit_check_job(node, tmp_dir)
                jobs.append((node, job_id, out_file))
            # Wait for all jobs and collect outputs
            while jobs:
                for idx, (node, job_id, out_file) in enumerate(jobs):
                    if self.wait_job(job_id) == 0:
                        free_gpus = self.parse_output(out_file)
                        if len(free_gpus) > 0:
                            result[node] = free_gpus
                        jobs.pop(idx)
                time.sleep(2)
                if time.time() > start_time + timeout:
                    print("Timeout Reached.")
                    break
            if result:
                return result
            else:
                print("No free GPUs found. Waiting 60 seconds before retrying...")
                time.sleep(60)
                return self.find_node_with_free_gpu()


if __name__ == "__main__":
    finder = SlurmGPUFinder(partition="VNL_GPU")
    free_gpus_by_node = finder.find_node_with_free_gpu()
    for node, free_gpus in free_gpus_by_node.items():
        print(f"Node: {node}, Free GPUs: {free_gpus}")
