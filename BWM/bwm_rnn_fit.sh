#!/usr/bin/env bash
#SBATCH -J 'bwm_analysis'
#SBATCH --output=Output/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=1-120


module purge
module load anaconda3/2022.5

conda activate ibl_bwm
python -u bwm_rnn_fit.py