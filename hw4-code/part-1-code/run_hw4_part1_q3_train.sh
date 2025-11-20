#!/bin/bash
#SBATCH --job-name=hw4_q3_train
#SBATCH --output=logs/q3_train_%j.out
#SBATCH --error=logs/q3_train_%j.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4


echo "Starting augmented data training..."
echo "This will train on 25,000 original + 5,000 transformed examples"

# Train with augmented data and evaluate on transformed test set
python3 main.py --train_augmented --eval_transformed

echo "Training completed!"
echo "Model saved to ./out_augmented/"