#!/usr/bin/env bash
#SBATCH -J 'bwm_analysis'
#SBATCH --output=Output/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-200
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G



module purge
module load anaconda3/2022.5

conda activate ibl_bwm
python -u bwm_analysis.py