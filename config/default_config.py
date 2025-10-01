"""
Configuration file for the medical tweet classification experiment
"""

class Config:
    # Model settings
    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 2
    
    # Dataset settings - Using a medical dataset
    DATASET_NAME = "medical_dialog"
    DATASET_CONFIG = "en"
    TEXT_COLUMN = "description"
    LABEL_COLUMN = "gender"  # for binary classification
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
    SEED = 7951
    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.2
