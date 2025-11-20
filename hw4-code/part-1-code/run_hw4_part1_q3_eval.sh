#!/bin/bash
#SBATCH --job-name=hw4_q3_eval
#SBATCH --output=logs/q3_eval_%j.out
#SBATCH --error=logs/q3_eval_%j.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

echo "Evaluating augmented model on ORIGINAL test data..."
python3 main.py --eval --model_dir out_augmented

echo ""
echo "Evaluating augmented model on TRANSFORMED test data..."
python3 main.py --eval_transformed --model_dir out_augmented

echo ""
echo "Evaluation completed!"
echo "Output files created:"
echo "  - out_augmented_original.txt"
echo "  - out_augmented_transformed.txt"