#!/usr/bin/env bash
#SBATCH -J 'berylanalysis'
#SBATCH --output=Output/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=16G


module purge
module load anaconda3/2022.5

conda activate ibl_bwm
python -u berylanalysis.py