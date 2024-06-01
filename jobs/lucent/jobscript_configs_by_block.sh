#!/bin/sh
#SBATCH --job-name=VIT-Explainability-Configs-DLCV-7715464
#SBATCH --output=/scratch/vihps/vihps01/stdouts/%j.out
#SBATCH --error=/scratch/vihps/vihps01/stderrs/%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --mail-user=allertmika@gmail.com
#SBATCH --mail-type=NONE

conda init
conda activate /scratch/vihps/vihps01/vit-interpretability-thesis/env

srun python3 /scratch/vihps/vihps01/vit-interpretability-thesis/code/jobs/lucent/generate_configs_by_block.py --output-dir /scratch/vihps/vihps01/vit-interpretability-thesis/configs-by-block