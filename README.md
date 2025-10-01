# DSA4213-DistilBERT
https://www.kaggle.com/datasets/mmvvsss/medical-text-distilbert-dsa4213?select=README.md

## Dataset Setup

This project uses the **Diseases Articles** dataset from Kaggle:

- **Dataset**: [Diseases Articles](https://www.kaggle.com/datasets/shyshcuk/diseases-articles)
- **Description**: Collection of articles about various diseases
- **Task**: Binary classification (Medical articles vs Non-medical texts)

### Setup Instructions

#### Option 1: From data file 
Simply run this script:

```bash
./scripts/run.sh
```
The script will:
- Skip the download step (since the file already exists)
- Load your CSV file from data/diseases_articles.csv
- Preprocess the data - automatically detect the structure and create binary labels
- Train both models - Full fine-tuning and LoRA
- Generate results - Compare both strategies

#### Option 2: Synthetic Data
If the download fails, the code will automatically create a synthetic medical dataset for testing.
Run the script:

```bash
./scripts/run.sh
```

## Repository Structure
 - requirements.txt
 - data
    - diseases_articles.csv
 - config
    - default_config.py
 - src
   - __init__.py
   - data_loader.py
   - model_setup.py
   - trainer.py
   - utils.py
 - scripts
   - run.sh
 - run_experiment.py
 - README.md


