#!/usr/bin/env bash
#SBATCH -J 'densityae'
#SBATCH --output=Output/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1



module purge
module load anaconda3/2022.5

conda activate ibl_bwm
python -u densityAE.py