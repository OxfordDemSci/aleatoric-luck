#!/bin/bash
#SBATCH -J aleatoric
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

module purge
module load Python/3.11

cd /gpfs3/users/mills/tej036/aleatoric-luck

source ~/venvs/aleatoric-luck/bin/activate

python src/sample_size.py

echo "All done!"