#!/bin/sh
#SBATCH -c 2
#SBATCH -t 0-08:00
#SBATCH -p dl
#SBATCH --mem=10G
#SBATCH -o log.out
#SBATCH -e log.err
#SBATCH --gres=gpu:0

python faissing.py