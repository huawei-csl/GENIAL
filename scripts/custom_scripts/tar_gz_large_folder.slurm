#!/bin/bash
#SBATCH --job-name=tar_large_dir
#SBATCH --output=tar_large_dir.out
#SBATCH --error=tar_large_dir.err
#SBATCH --time=06:00:00        # adjust based on dir size
#SBATCH --cpus-per-task=16      # for parallel gzip
#SBATCH --mem=8G               # adjust if needed
#SBATCH --nodelist=aisrv03
#SBATCH --partition=AI-CPU

module load pigz   # if your cluster has it
cd /scratch/mbouvier

# Using parallel gzip to speed up compression
tar -I pigz -cvf proj.tar.gz proj
