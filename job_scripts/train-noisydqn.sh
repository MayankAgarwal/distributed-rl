#!/bin/bash
#
#SBATCH --job-name=ndqn
#SBATCH -o noisy_dqn_%j.txt  # output file
#SBATCH -e noisy_dqn_%j.err   # File to which STDERR will be written
##SBATCH --partition=m40-short # Partition to submit to
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH --time=1-0:00:00       # Runtime in D-HH:MM
#SBATCH --mem=40000
##SBATCH --exclude=node[050-099]


python ../noisydqn-breakout.py
echo done learning!
exit