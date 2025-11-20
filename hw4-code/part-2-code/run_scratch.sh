#!/bin/bash
#SBATCH --job-name=t5_scratch
#SBATCH --output=logs/t5_scr_%j.out
#SBATCH --error=logs/t5_scr_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Run training - FROM SCRATCH (for extra credit)
echo "Starting T5 Training from Scratch..."
python train_t5.py \
    --experiment_name scr_experiment \
    --batch_size 16 \
    --test_batch_size 32 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --max_n_epochs 20 \
    --patience_epochs 5 \
    --num_warmup_epochs 2 \
    --scheduler_type cosine \
    --optimizer_type AdamW \
    --max_length 128 \
    --num_beams 4

echo ""
echo "Job completed!"