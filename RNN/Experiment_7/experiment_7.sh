#!/usr/bin/env bash
#SBATCH -J 'exp_7'
#SBATCH -o slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-1125
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00



module purge
module load cudatoolkit/12.0
module load cudnn/cuda-11.x/8.2.0
module load anaconda3/2022.5

conda activate ibl_lca
python -u experiment_7.py