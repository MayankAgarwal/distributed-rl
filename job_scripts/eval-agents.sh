#!/bin/bash
#
#SBATCH --job-name=eval_dqn
#SBATCH -o eval_dqn_%j.txt  # output file
#SBATCH -e eval_dqn_%j.err   # File to which STDERR will be written
##SBATCH --partition=m40-short # Partition to submit to
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --time=1-0:00:00       # Runtime in D-HH:MM
#SBATCH --mem=40000
##SBATCH --exclude=node[050-099]


MODEL_TYPE="dqn"
MODELS_PATH="results/dqn-breakout-full"
RESULT_FOLDER="results/[T] dqn_play"

python "/home/mayankagarwa/RL/code/vis-play-test-games.py" "$MODEL_TYPE" "$MODELS_PATH" "$RESULT_FOLDER"
echo done learning!
exit