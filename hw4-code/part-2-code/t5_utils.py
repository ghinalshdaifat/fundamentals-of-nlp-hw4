import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    # pass


    # Initialize Weights & Biases for experiment tracking.
    # Args:
        # args: arguments containing experiment configuration

    model_type = 'ft' if args.finetune else 'scr'
    wandb.init(
        project="nlp-hw4-text-to-sql",
        name=f"t5_{model_type}_{args.experiment_name}",
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": args.max_n_epochs,
            "optimizer": args.optimizer_type,
            "scheduler": args.scheduler_type,
            "weight_decay": args.weight_decay,
            "finetune": args.finetune,
        }
    )

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    # pass
    model_checkpoint = 'google-t5/t5-small'

    if args.finetune:
        # Fine-tuning: load pretrained weights
        print(f"Loading pretrained T5 model from {model_checkpoint}...")
        model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
        print("Pretrained model loaded successfully!")
    
    else: 
        # Training from scratch: initialize with random weights
        print(f"Initializing T5 model from scratch using {model_checkpoint} config...")
        config = T5Config.from_pretrained(model_checkpoint)
        model = T5ForConditionalGeneration(config)
        print("Model initialized from scratch!")

    # Move model to device 
    model = model.to(DEVICE)

    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model

def mkdir(dirpath):
    # Create directory if doesn't exist
    # Args:
        # dirpath: path to directory

    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    # Args: 
        # checpoint_dir: directory to save checkpoint
        # model: model to save
        # best: whether this is the best model so far
    
    mkdir(checkpoint_dir)

    if best: 
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        print(f"Saving best model to {checkpoint_path}")
    else: 
        checkpoint_path = os.path.join(checkpoint_dir, 'last_model.pt')
        print(f"Saving last model to {checkpoint_path}")

    # Save model state dictionary 
    torch.save({
        'model_state_dict': model.state_dict(), 
    }, checkpoint_path)
    # pass

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    # Args: 
        # args: arguments containing checkpoint directory and finetune flag
        # best: whether to load the best model (True) or last model (False)
    # Returns: 
        # loaded model
    
    # Initialize model architecture
    model = initialize_model(args)

    # Load checkpoint
    if best: 
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else: 
        checkpoint_path = os.path.join(args.checkpoint_dir, 'last_model.pt')

    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location = DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
    # pass

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    # Initialize optimizer and learning rate scheduler
    # Args: 
        # args: arguments containing optimizer / scheduler configuration
        # model: model to optimize
        # epoch_length: number of batches per epoch
    # Returns: 
        # Tuple of (optimizer, scheduler)

    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    # Initilize AdamW optimizer with weight decay for non-bias / LayerNorm parameters
    # Args: 
        # args: arguments containing learning rate and weight decay 
        # model: model to optimize
    # Returns: 
        # Optimizer
    
    # Get parameter names that should have weight decay
    # decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    # Split parameters into two groups: with and without weight decay
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() 
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() 
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr = args.learning_rate, 
            eps=1e-8, 
            betas=(0.9, 0.999)
        )
    else:
        # pass
        raise NotImplementedError(f"Optimizer {args.optimizer_type} not implemented")

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    # Initialize learning rate scheduler
    # Args: 
        # args: arguments containing scheduler confid
        # optimizer: optimizer to schedule
        # epoch_lenght: number of batches per epoch
    # Returns: 
        # Scheduler or None

    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer,
        num_warmup_steps,
        num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps, 
            num_training_steps)
    else:
        raise NotImplementedError(f"Scheduler {args.scheduler_type} not implemented")

def get_parameter_names(model, forbidden_layer_types):
    # Get names of all parameters in the model, excluding specified layer types
    # Args: 
        # model: PyTorch model
        # forbidden_layer_types: tuple of layer types to exclude
    # Returns: 
        # List of parameter names
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

