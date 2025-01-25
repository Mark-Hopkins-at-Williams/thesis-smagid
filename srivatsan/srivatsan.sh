#!/bin/sh
#SBATCH -c 1
#SBATCH -t 4-00:00
#SBATCH -p dl
#SBATCH --mem=10G
#SBATCH -o log.out
#SBATCH -e log.err
#SBATCH --gres=gpu:1

python main.py val Capitals64 --save-every 25 --eval-every 25 --blanks 10 --z-size 6 -e 1000