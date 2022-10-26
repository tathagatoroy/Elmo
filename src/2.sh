#!/bin/bash
#SBATCH -A research
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=NLP
#SBATCH --output=elmo.log


# Discord notifs on start and end
source notify

# Fail on error
set -e

cd /home2/tathagato/Elmo/src
source /home2/tathagato/miniconda3/bin/activate habitat

python3 main_review.py
#python3 main_review.py
