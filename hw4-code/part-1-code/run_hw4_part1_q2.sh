#!/bin/bash
#SBATCH --job-name=hw4_part1_q2
#SBATCH --output=logs/bert_q2_%j.out
#SBATCH --error=logs/bert_q2_%j.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4


# Evaluate on transformed test set
echo "Evaluating on transformed test set..."
python3 main.py --eval_transformed

echo "Evaluation completed!"
echo "Output file 'out_transformed.txt' has been created"
