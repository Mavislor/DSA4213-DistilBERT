"""
Data loading and preprocessing utilities
"""

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import pandas as pd
from config.default_config import Config


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_and_preprocess_data(self):
        dataset = load_dataset(self.config.DATASET_NAME)
        
        # Convert to binary classification: medical_health vs other topics
        dataset = dataset.map(self._convert_to_medical_binary)
        dataset = dataset.remove_columns(['label'])
        dataset = dataset.rename_column('binary_label', 'label')
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        return tokenized_dataset
    
    def _convert_to_medical_binary(self, example):
        # Convert multi-class to binary classification
        # Label 4 corresponds to 'medical_health' in the original dataset
        example['binary_label'] = 1 if example['label'] == 4 else 0
        return example
    
    def _tokenize_function(self, examples):
        # Tokenize the text examples
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=self.config.MAX_LENGTH,
        )
    
    def get_label_distribution(self, dataset):
        # Get distribution of labels in dataset
        train_labels = dataset['train']['label']
        val_labels = dataset['validation']['label']
        
        train_counts = pd.Series(train_labels).value_counts().sort_index()
        val_counts = pd.Series(val_labels).value_counts().sort_index()
        
        return {
            'train': train_counts,
            'validation': val_counts
        }
