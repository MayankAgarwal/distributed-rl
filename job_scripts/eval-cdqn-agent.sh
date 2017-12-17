#!/bin/bash
#
#SBATCH --job-name=eval_cdqn
#SBATCH -o eval_cdqn_%j.txt  # output file
#SBATCH -e eval_cdqn_%j.err   # File to which STDERR will be written
##SBATCH --partition=m40-short # Partition to submit to
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --time=1-0:00:00       # Runtime in D-HH:MM
#SBATCH --mem=40000
##SBATCH --exclude=node[050-099]


MODELS_PATH="results/categorical_dqn-breakout-upd"
RESULT_FOLDER="results/[eval] cdqn"

python "/home/mayankagarwa/RL/code/vis-play-test-games-cdqn.py" "$MODELS_PATH" "$RESULT_FOLDER"
echo done learning!
exit