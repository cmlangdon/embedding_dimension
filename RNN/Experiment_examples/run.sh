#!/usr/bin/env bash
#SBATCH -J 'examples'
#SBATCH -o slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-40
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G


module purge
module load cudatoolkit/12.0
module load cudnn/cuda-11.x/8.2.0
module load anaconda3/2022.5

conda activate ibl_lca
python -u run.py