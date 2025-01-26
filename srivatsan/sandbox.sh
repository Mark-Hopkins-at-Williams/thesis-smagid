#!/bin/sh
#SBATCH -c 1
#SBATCH -t 0-02:00
#SBATCH -p dl
#SBATCH --mem=10G
#SBATCH -o log.out
#SBATCH -e log.err
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=3
python sandbox.py val Capitals64 --save-every 25 --eval-every 25 --blanks 10 --z-size 6 -e 1000