#!/bin/bash
#BATCH -A research
#SBATCH -n 30
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=train Language
#SBATCH --output=elmo.log

echo "starting"
mv /home2/tathagato/Elmo/src

source /home2/tathagato/miniconda3/bin/activate habitat

python3 main.py 
#python3 main_review.py 

