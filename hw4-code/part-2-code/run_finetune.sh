#!/bin/bash
#SBATCH --job-name=t5_finetune
#SBATCH --output=logs/t5_ft_%j.out
#SBATCH --error=logs/t5_ft_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4


# Run training - FINE-TUNING
echo "Starting T5 Fine-tuning..."
python train_t5.py \
    --finetune \
    --experiment_name ft_experiment \
    --batch_size 16 \
    --test_batch_size 32 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --max_n_epochs 30 \
    --patience_epochs 3 \
    --num_warmup_epochs 1 \
    --scheduler_type cosine \
    --optimizer_type AdamW \
    --max_length 300 \
    --num_beams 4

echo ""
echo "Job completed!"