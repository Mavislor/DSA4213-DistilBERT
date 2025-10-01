"""
Data loading and preprocessing utilities for Diseases Articles dataset
"""

import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
import os
from config.default_config import Config


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_and_preprocess_data(self):
        """Load and preprocess the Diseases Articles dataset from Kaggle"""
        print("Loading Diseases Articles dataset from Kaggle...")
        
        if not os.path.exists(self.config.DATASET_PATH):
            self._download_dataset()
        
        df = pd.read_csv(self.config.DATASET_PATH)
        print(f"Dataset loaded with {len(df)} samples")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        
        # Preprocess the dataset
        dataset = self._preprocess_data(df)
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=[self.config.TEXT_COLUMN]
        )
        
        return tokenized_dataset
    
    def _download_dataset(self):
        """Download the Diseases Articles dataset from Kaggle"""
        import subprocess
        import os
        import zipfile
        
        print("Downloading Diseases Articles dataset from Kaggle...")
        
        # Create data directory
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        
        try:
            # Download using kaggle API
            subprocess.run([
                "kaggle", "datasets", "download", 
                self.config.KAGGLE_DATASET,
                "-p", self.config.DATA_DIR
            ], check=True, capture_output=True)
            
            zip_path = os.path.join(self.config.DATA_DIR, "diseases-articles.zip")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.DATA_DIR)
            
            extracted_files = os.listdir(self.config.DATA_DIR)
            csv_files = [f for f in extracted_files if f.endswith('.csv')]
            
            if csv_files:
                # Use the first CSV file found
                original_csv_path = os.path.join(self.config.DATA_DIR, csv_files[0])
                os.rename(original_csv_path, self.config.DATASET_PATH)
                print(f"Dataset downloaded and saved as {self.config.DATASET_PATH}")
            else:
                raise FileNotFoundError("No CSV file found in downloaded dataset")
            
            print("Dataset downloaded and extracted successfully!")
            
        except Exception as e:
            print(f"Error downloading from Kaggle: {e}")
            print("Creating synthetic medical dataset as fallback...")
            self._create_synthetic_dataset()
    
    def _create_synthetic_dataset(self):
        """Create a synthetic medical dataset if download fails"""
        print("Creating synthetic medical dataset...")
        
        synthetic_data = {
            'text': [
                # Medical articles about diseases
                "Alzheimer's disease is a progressive neurodegenerative disorder that affects memory and cognitive function.",
                "Diabetes mellitus is characterized by high blood sugar levels resulting from insulin deficiency or resistance.",
                "Coronary artery disease occurs when the blood vessels supplying the heart become narrowed or blocked.",
                "Asthma is a chronic inflammatory disease of the airways causing wheezing and shortness of breath.",
                "Rheumatoid arthritis is an autoimmune disorder that primarily affects the joints causing pain and swelling.",
                "Parkinson's disease is a movement disorder characterized by tremors, rigidity, and bradykinesia.",
                "Multiple sclerosis is an autoimmune disease that affects the central nervous system.",
                "Hypertension, or high blood pressure, is a major risk factor for heart disease and stroke.",
                "Osteoporosis is a condition characterized by decreased bone density and increased fracture risk.",
                "Chronic kidney disease involves gradual loss of kidney function over time.",
                
                # Non-medical texts
                "The history of ancient civilizations reveals fascinating cultural developments.",
                "Modern technology has revolutionized communication and information access.",
                "Climate change impacts weather patterns and ecosystems worldwide.",
                "Renewable energy sources like solar and wind power are becoming more efficient.",
                "Artificial intelligence is transforming various industries and daily life.",
                "Space exploration continues to reveal mysteries of the universe.",
                "Sustainable agriculture practices help protect the environment.",
                "Digital marketing strategies evolve with changing consumer behavior.",
                "Urban planning addresses challenges of growing city populations.",
                "Literary analysis explores themes and techniques in great works of fiction."
            ],
            'label': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 1=medical, 0=non-medical
        }
        
        df = pd.DataFrame(synthetic_data)
        df.to_csv(self.config.DATASET_PATH, index=False)
        print("Synthetic dataset created!")
    
    def _preprocess_data(self, df):
        print("Preprocessing medical data...")
    
        print(f"Original dataset shape: {df.shape}")
        print(f"Column names: {df.columns.tolist()}")
        
        df = df.dropna(subset=['Sentence'])
        df = df[df['Sentence'].str.strip() != '']
        df = df.reset_index(drop=True)
        
        # Use 'Sentence' column as text
        self.config.TEXT_COLUMN = 'Sentence'
        print(f"Using column 'Sentence' as text input")
     
        def create_binary_label(row):
            sentence = str(row['Sentence']).lower()
            disease_name = str(row['Disease Name']).lower() if pd.notna(row['Disease Name']) else ""
        
            # If disease name is available and appears in sentence, label as 1 (specific disease)
            # Otherwise label as 0 (general medical text)
            if disease_name and disease_name in sentence:
                return 1 
            else:
                return 0 
    
        # Apply the labeling strategy
        df['label'] = df.apply(create_binary_label, axis=1)
    
        print(f"Processed dataset shape: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
    
        # If data is imbalanced, this code is to balance it
        label_counts = df['label'].value_counts()
        if len(label_counts) > 1:
            min_count = label_counts.min()
            balanced_dfs = []
            for label in label_counts.index:
                label_df = df[df['label'] == label]
                if len(label_df) > min_count:
                    label_df = label_df.sample(min_count, random_state=self.config.SEED)
                balanced_dfs.append(label_df)
        
            df = pd.concat(balanced_dfs, ignore_index=True)
            print(f"Balanced dataset shape: {df.shape}")
            print(f"Balanced label distribution:\n{df['label'].value_counts()}")
    
        # Split the data
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(
            df, 
            test_size=self.config.VAL_SIZE + self.config.TEST_SIZE, 
            random_state=self.config.SEED,
            stratify=df['label']
        )
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=self.config.TEST_SIZE/(self.config.VAL_SIZE + self.config.TEST_SIZE), 
            random_state=self.config.SEED,
            stratify=temp_df['label']
        )
    
        # Create Hugging Face dataset
        dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
            'validation': Dataset.from_pandas(val_df.reset_index(drop=True)),
            'test': Dataset.from_pandas(test_df.reset_index(drop=True))
        })
    
        print(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        print(f"Label distribution - Train: {train_df['label'].value_counts().to_dict()}")
        print(f"Label distribution - Val: {val_df['label'].value_counts().to_dict()}")
        print(f"Label distribution - Test: {test_df['label'].value_counts().to_dict()}")
    
        return dataset
    
    def _tokenize_function(self, examples):
        """Tokenize the text examples"""
        return self.tokenizer(
            examples[self.config.TEXT_COLUMN],
            truncation=True,
            padding=False,
            max_length=self.config.MAX_LENGTH,
        )
    
    def get_label_distribution(self, dataset):
        """Get distribution of labels in dataset"""
        train_labels = dataset['train']['label']
        val_labels = dataset['validation']['label']
        test_labels = dataset['test']['label']
        
        train_counts = pd.Series(train_labels).value_counts().sort_index()
        val_counts = pd.Series(val_labels).value_counts().sort_index()
        test_counts = pd.Series(test_labels).value_counts().sort_index()
        
        return {
            'train': train_counts,
            'validation': val_counts,
            'test': test_counts
        }
