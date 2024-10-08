#!/bin/sh
#SBATCH -c 2
#SBATCH -t 0-20:00
#SBATCH -p dl
#SBATCH --mem=10G
#SBATCH -o logs/log%j.out
#SBATCH -e logs/log%j.err
#SBATCH --gres=gpu:2

python training.py