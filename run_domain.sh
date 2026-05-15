#!/bin/bash
#SBATCH -J aleatoric
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -t 02:00:00
#SBATCH -p short

module purge
module load Python/3.11

cd /gpfs3/users/mills/tej036/aleatoric-luck

source ~/venvs/aleatoric-luck/bin/activate

python src/domain_wise.py

echo "All done!"