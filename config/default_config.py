"""
Configuration file for the medical tweet classification experiment
"""

class Config:
    # Model settings
    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 2
    
    # Dataset settings
    DATASET_NAME = "cardiffnlp/tweet_topic_single"
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    
    # Training settings
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500
    
    # LoRA settings
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    
    # Paths
    OUTPUT_DIR = "./outputs"
    LOGGING_DIR = "./logs"
    
    # Experiment settings
    SEED = 42
    TRAIN_SIZE = 0.8  # For train/val split if needed
    VAL_SIZE = 0.2
