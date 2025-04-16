#!/usr/bin/env bash
#SBATCH -J 'exp_2'
#SBATCH -o slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-25
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=mig


module purge
module load cudatoolkit/12.6
module load anaconda3/2022.5

conda activate ibl_lca
python -u experiment_2.py