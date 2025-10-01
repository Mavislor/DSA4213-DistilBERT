"""
Configuration file for the medical text classification experiment
"""

class Config:
    # Model settings
    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 2
    
    # Dataset settings 
    DATASET_PATH = "./data/diseases_articles.csv"
    KAGGLE_DATASET = "shyshcuk/diseases-articles"
    TEXT_COLUMN = "text"
    LABEL_COLUMN = "label"
    MAX_LENGTH = 256
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
    DATA_DIR = "./data"
    
    # Experiment settings
    SEED = 7951
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
