"""
Utility functions
"""

import random
import numpy as np
import torch
from config.default_config import Config


def set_seed(seed: int = 7951):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def print_experiment_summary(config: Config, label_distribution):
    """Print experiment summary"""
    print("=" * 60)
    print("EXPERIMENT SETUP")
    print("=" * 60)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Dataset: Diseases Articles (from CSV)")
    print(f"Task: Binary Classification (Medical Articles)")
    print(f"Training samples: {len(label_distribution['train'])}")
    print(f"Validation samples: {len(label_distribution['validation'])}")
    print(f"Test samples: {len(label_distribution['test'])}")
    print(f"Label distribution - Train: {dict(label_distribution['train'])}")
    print(f"Label distribution - Validation: {dict(label_distribution['validation'])}")
    print(f"Label distribution - Test: {dict(label_distribution['test'])}")
    print(f"Text column: {config.TEXT_COLUMN}")
    print(f"Random Seed: {config.SEED}")
    print("=" * 60)


def setup_environment():
    """Setup environment variables and check GPU availability"""
    import os
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Using CPU for training")
    
    # Set environment variable for tokenizers parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
