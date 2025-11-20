#!/bin/bash
#SBATCH --job-name=hw4_part1_q1_full
#SBATCH --output=logs/bert_q1_full_%j.out
#SBATCH --error=logs/bert_q1_full_%j.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

# Check GPU availability
echo "Checking GPU..."
python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

# Run FULL training 
echo "Starting FULL training..."
python3 main.py --train --eval

echo "Job completed!"
echo "Output file 'out_original.txt' has been created in the current directory"