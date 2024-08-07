#!/bin/sh
#SBATCH --job-name=VIT-Explainability-DLCV-7715464
#SBATCH --output=/scratch/vihps/vihps01/stdouts/%j.out
#SBATCH --error=/scratch/vihps/vihps01/stderrs/%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --time=288:00:00
#SBATCH --mail-user=allertmika@gmail.com
#SBATCH --mail-type=NONE

eval "$(conda shell.bash hook)"

conda init
conda activate /scratch/vihps/vihps01/vit-interpretability-thesis/env

export MIOPEN_USER_DB_PATH=/scratch/vihps/vihps01/.config/miopen_$SLURM_PROCID/
srun python3 /scratch/vihps/vihps01/vit-interpretability-thesis/code/jobs/lucent/run_val_comp.py