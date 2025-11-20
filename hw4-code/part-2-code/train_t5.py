import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import (
    initialize_model, 
    initialize_optimizer_and_scheduler, 
    save_model, 
    load_model_from_checkpoint, 
    setup_wandb
)
from transformers import T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0


def get_args():
    # Arguments for training. You may choose to change or extend these as you see fit.

    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', 
                       help="Whether to finetune T5 or train from scratch")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", 
                       choices=["AdamW"],
                       help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--scheduler_type', type=str, default="cosine", 
                       choices=["none", "cosine", "linear"],
                       help="Whether to use a LR scheduler and what type")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                       help="How many epochs to warm up the learning rate")
    parser.add_argument('--max_n_epochs', type=int, default=10,
                       help="How many epochs to train the model")
    parser.add_argument('--patience_epochs', type=int, default=3,
                       help="Early stopping patience")

    parser.add_argument('--use_wandb', action='store_true',
                       help="If set, use wandb for experiment tracking")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                       help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=32)
    
    # Generation hyperparameters
    parser.add_argument('--max_length', type=int, default=128,
                       help="Maximum length for generated SQL queries")
    parser.add_argument('--num_beams', type=int, default=4,
                       help="Number of beams for beam search")

    args = parser.parse_args()
    return args


def train(args, model, train_loader, dev_loader, optimizer, scheduler):

    # Main training loop with validation and early stopping.
    # Args:
        # args: training arguments
        # model: T5 model
        # train_loader: training data loader
        # dev_loader: development data loader
        # optimizer: optimizer
        # scheduler: learning rate scheduler

    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', 
        f't5_{model_type}_experiments', 
        args.experiment_name
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    
    # Paths for evaluation
    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/ground_truth_dev.pkl'
    model_sql_path = f'results/t5_{model_type}_{args.experiment_name}_dev.sql'
    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_dev.pkl'
    
    for epoch in range(args.max_n_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.max_n_epochs}")
        print(f"{'='*60}")
        
        # Training
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Average train loss: {tr_loss:.4f}")

        # Evaluation
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader,
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path
        )
        
        print(f"\nDev Results:")
        print(f"  Loss: {eval_loss:.4f}")
        print(f"  Record F1: {record_f1:.4f}")
        print(f"  Record EM: {record_em:.4f}")
        print(f"  SQL EM: {sql_em:.4f}")
        print(f"  Error Rate: {error_rate*100}%")

        # Log to wandb
        if args.use_wandb:
            result_dict = {
                'train/loss': tr_loss,
                'dev/loss': eval_loss,
                'dev/record_f1': record_f1,
                'dev/record_em': record_em,
                'dev/sql_em': sql_em,
                'dev/error_rate': error_rate,
                'epoch': epoch
            }
            wandb.log(result_dict, step=epoch)

        # Check for improvement
        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            print(f"✓ New best Record F1: {best_f1:.4f}")
        else:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epoch(s)")

        # Save checkpoints
        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        # Early stopping
        if epochs_since_improvement >= args.patience_epochs:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break
    
    print(f"\nTraining completed! Best Record F1: {best_f1:.4f}")


def train_epoch(args, model, train_loader, optimizer, scheduler):
    # Train for one epoch.
    # Args:
        # args: training arguments
        # model: T5 model
        # train_loader: training data loader
        # optimizer: optimizer
        # scheduler: learning rate scheduler 
    # Returns:
        # Average loss for the epoch

    model.train()
    total_loss = 0
    total_tokens = 0

    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        encoder_input, encoder_mask, decoder_input, decoder_targets, _ = batch
        
        # Move to device
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # T5 expects labels (not decoder_input_ids) for training
        # It will automatically shift them internally
        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=decoder_targets,  # T5 handles shifting internally
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        # Track loss
        with torch.no_grad():
            # Count non-padding tokens
            non_pad = decoder_targets != PAD_IDX
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss

        
def eval_epoch(args, model, dev_loader, gt_sql_path, model_sql_path, 
               gt_record_path, model_record_path):
    
    # Evaluate the model on the development set. 
    # Args:
        # args: arguments
        # model: T5 model
        # dev_loader: development data loader
        # gt_sql_path: path to ground truth SQL queries
        # model_sql_path: path to save model-generated SQL queries
        # gt_record_path: path to ground truth database records
        # model_record_path: path to save model-generated database records   
    # Returns:
        # Tuple of (eval_loss, record_f1, record_em, sql_em, error_rate)
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    generated_queries = []
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    progress_bar = tqdm(dev_loader, desc = "Evaluating")
    
    with torch.no_grad():
        for batch in progress_bar:
            encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder = batch
            
            # Move to device
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            
            # Compute loss
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                labels=decoder_targets,
            )
            
            loss = outputs.loss
            
            # Track loss
            non_pad = decoder_targets != PAD_IDX
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Generate SQL queries using beam search
            generated_ids = model.generate(
                input_ids = encoder_input,
                attention_mask = encoder_mask,
                max_length = args.max_length,
                num_beams = args.num_beams,
                early_stopping = True,
            )
            
            # Decode generated queries
            batch_queries = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            generated_queries.extend(batch_queries)
    
    # Compute average loss
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    
    # Save queries and compute metrics
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    
    # Compute metrics
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_path, model_sql_path,
        gt_record_path, model_record_path
    )
    
    # Compute error rate
    error_rate = sum(1 for msg in error_msgs if msg != "") / len(error_msgs)
    
    return avg_loss, record_f1, record_em, sql_em, error_rate

        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):

    # Perform inference on the test set.
    
    # Args:
        # args: arguments
        # model: T5 model
        # test_loader: test data loader
        # model_sql_path: path to save model-generated SQL queries
        # model_record_path: path to save model-generated database records

    model.eval()
    generated_queries = []
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    print("\nGenerating SQL queries for test set...")
    progress_bar = tqdm(test_loader, desc="Test Inference")
    
    with torch.no_grad():
        for batch in progress_bar:
            encoder_input, encoder_mask, initial_decoder = batch
            
            # Move to device
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate SQL queries
            generated_ids = model.generate(
                input_ids = encoder_input,
                attention_mask = encoder_mask,
                max_length = args.max_length,
                num_beams = args.num_beams,
                early_stopping = True,
            )
            
            # Decode generated queries
            batch_queries = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            generated_queries.extend(batch_queries)
    
    # Save queries and records
    print(f"\nSaving {len(generated_queries)} generated queries...")
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    print(f"Saved to {model_sql_path} and {model_record_path}")


def main():
    # Get arguments
    args = get_args()
    print("\n" + "=" * 60)
    print("T5 Text-to-SQL Training")
    print("=" * 60)
    print(f"Mode: {'Fine-tuning' if args.finetune else 'Training from scratch'}")
    print(f"Device: {DEVICE}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_n_epochs}")
    print("=" * 60 + "\n")
    
    # Setup wandb if requested
    if args.use_wandb:
        setup_wandb(args)

    # Load data
    print("Loading data...")
    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size, args.test_batch_size
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    # Initialize model
    model = initialize_model(args)
    
    # Initialize optimizer and scheduler
    optimizer, scheduler = initialize_optimizer_and_scheduler(
        args, model, len(train_loader)
    )
    print(f"\nOptimizer: {args.optimizer_type}")
    print(f"Scheduler: {args.scheduler_type}\n")

    # Train
    print("Starting training...")
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Load best model for evaluation
    print("\nLoading best model for final evaluation...")
    model = load_model_from_checkpoint(args, best = True)
    model.eval()
    
    # Final evaluation on dev set
    model_type = 'ft' if args.finetune else 'scr'
    experiment_name = args.experiment_name
    
    print("\n" + "=" * 60)
    print("Final Evaluation on Dev Set")
    print("=" * 60)
    
    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/ground_truth_dev.pkl'
    model_sql_path = f'results/t5_{model_type}_{experiment_name}_dev.sql'
    model_record_path = f'records/t5_{model_type}_{experiment_name}_dev.pkl'
    
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader,
        gt_sql_path, model_sql_path,
        gt_record_path, model_record_path
    )
    
    print(f"\nDev Set Results:")
    print(f"  Loss: {dev_loss:.4f}")
    print(f"  Record F1: {dev_record_f1:.4f}")
    print(f"  Record EM: {dev_record_em:.4f}")
    print(f"  SQL EM: {dev_sql_em:.4f}")
    print(f"  Error Rate: {dev_error_rate*100:.2f}%")

    # Test set inference
    print("\n" + "=" * 60)
    print("Test Set Inference")
    print("=" * 60)
    
    model_sql_path = f'results/t5_{model_type}_{experiment_name}_test.sql'
    model_record_path = f'records/t5_{model_type}_{experiment_name}_test.pkl'
    
    test_inference(args, model, test_loader, model_sql_path, model_record_path)
    
    print("\n" + "=" * 60)
    print("Training and evaluation completed!")
    print("=" * 60)
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()