"""
Diagnostic script to check if all required files and directories exist.
Run this first to make sure your environment is set up correctly.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory(dirpath, description):
    """Check if a directory exists and print status."""
    exists = os.path.isdir(dirpath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {dirpath}")
    return exists

def main():
    print("="*60)
    print("Environment Diagnostic Check")
    print("="*60)
    
    print(f"\nCurrent directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    all_good = True
    
    # Check directories
    print("\n" + "="*60)
    print("Checking Directories")
    print("="*60)
    
    dirs_to_check = [
        ('data', 'Data directory'),
        ('records', 'Records directory'),
        ('results', 'Results directory'),
        ('checkpoints', 'Checkpoints directory'),
        ('logs', 'Logs directory'),
    ]
    
    for dirname, description in dirs_to_check:
        if not check_directory(dirname, description):
            all_good = False
            print(f"  → Run: mkdir -p {dirname}")
    
    # Check required data files
    print("\n" + "="*60)
    print("Checking Data Files")
    print("="*60)
    
    data_files = [
        ('data/train.nl', 'Training natural language queries'),
        ('data/train.sql', 'Training SQL queries'),
        ('data/dev.nl', 'Dev natural language queries'),
        ('data/dev.sql', 'Dev SQL queries'),
        ('data/test.nl', 'Test natural language queries'),
    ]
    
    for filepath, description in data_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Check record files
    print("\n" + "="*60)
    print("Checking Record Files")
    print("="*60)
    
    record_files = [
        ('records/ground_truth_dev.pkl', 'Dev ground truth records'),
    ]
    
    for filepath, description in record_files:
        check_file_exists(filepath, description)
    
    # Check implementation files
    print("\n" + "="*60)
    print("Checking Implementation Files")
    print("="*60)
    
    impl_files = [
        ('load_data.py', 'Data loading implementation'),
        ('t5_utils.py', 'T5 utilities implementation'),
        ('train_t5.py', 'Training script'),
    ]
    
    for filepath, description in impl_files:
        if not check_file_exists(filepath, description):
            all_good = False
            print(f"  → Make sure you renamed {filepath.replace('.py', '_complete.py')} to {filepath}")
    
    # Check helper scripts
    print("\n" + "="*60)
    print("Checking Helper Scripts")
    print("="*60)
    
    helper_files = [
        ('compute_statistics.py', 'Statistics computation script'),
        ('test_implementation.py', 'Test script'),
        ('run_finetune.sh', 'Fine-tuning SLURM script'),
    ]
    
    for filepath, description in helper_files:
        check_file_exists(filepath, description)
    
    # Check Python packages
    print("\n" + "="*60)
    print("Checking Python Packages")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch not installed")
        all_good = False
    
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
        all_good = False
    
    try:
        import nltk
        print(f"✓ NLTK: {nltk.__version__}")
    except ImportError:
        print("✗ NLTK not installed")
        all_good = False
    
    # File content check
    if os.path.exists('data/train.nl'):
        print("\n" + "="*60)
        print("Sample Data Check")
        print("="*60)
        
        with open('data/train.nl', 'r') as f:
            lines = f.readlines()
            print(f"✓ train.nl has {len(lines)} lines")
            if len(lines) > 0:
                print(f"  First line: {lines[0].strip()[:60]}...")
        
        if os.path.exists('data/train.sql'):
            with open('data/train.sql', 'r') as f:
                lines = f.readlines()
                print(f"✓ train.sql has {len(lines)} lines")
                if len(lines) > 0:
                    print(f"  First line: {lines[0].strip()[:60]}...")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if all_good:
        print("✓ All critical checks passed!")
        print("  You can proceed with running test_implementation.py")
    else:
        print("✗ Some issues found. Please fix them before proceeding.")
        print("\nCommon fixes:")
        print("  1. Make sure you're in the part-2-code directory")
        print("  2. Create missing directories: mkdir -p data records results checkpoints logs")
        print("  3. Verify data files exist in data/ folder")
        print("  4. Rename *_complete.py files to remove '_complete'")
        print("  5. Install requirements: pip install -r requirements.txt")
    
    print("="*60)

if __name__ == "__main__":
    main()