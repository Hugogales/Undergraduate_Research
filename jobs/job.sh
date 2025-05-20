#!/bin/bash
#SBATCH --job-name=RL_HUGO
#SBATCH --output=output_HUGO_v101_6.log
#SBATCH --error=error_HUGO_v101_6.log
#SBATCH --partition=teaching                 # dgx, dgxh100, teaching
#SBATCH --gpus=4
#SBATCH --account=undergrad_research
#SBATCH --time=7-00:00:00  # Set the time limit to one week (7 days)
#SBATCH --cpus-per-task=10


# Load necessary modules if needed
# module load singularity

# Directly run the Python script inside the container
cd ..
cd Game
# Install required packages
singularity exec --nv -B /data:/data /data/containers/msoe-tf2x.sif pip install trueskill
# Run the main script
singularity exec --nv -B /data:/data /data/containers/msoe-tf2x.sif python src/main.py

# run the job with sbatch job.sh
# srun --pty --partition=teaching --gres=gpu:0 --cpus-per-task=1 --account=undergrad_research --time=7-00:00:00 singularity shell --nv -B /data:/data /data/containers/msoe-tf2x.sif