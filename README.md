# DSA4213-DistilBERT
[Kaggle Page](https://www.kaggle.com/datasets/mmvvsss/medical-text-distilbert-dsa4213?select=README.md)

## Dataset Setup

This project uses the **Diseases Articles** dataset from Kaggle:

- **Dataset**: [Diseases Articles](https://www.kaggle.com/datasets/shyshcuk/diseases-articles)
- **Description**: Collection of articles about various diseases
- **Task**: Binary classification (Medical articles vs Non-medical texts)

### Setup Instructions

#### From data file 
Simply run this script:

```bash
./scripts/run.sh
```
The script will:
- Load your CSV file from data/diseases_articles.csv
- Preprocess the data - automatically detect the structure and create binary labels
- Train both models - Full fine-tuning and LoRA
- Generate results - Compare both strategies


## Repository Structure
 - requirements.txt
 - data/
    - diseases_articles.csv
 - config/
    - default_config.py
 - src/
   - init.py
   - data_loader.py
   - model_setup.py
   - trainer.py
   - utils.py
 - scripts/
   - run.sh
 - run_experiment.py
 - README.md




