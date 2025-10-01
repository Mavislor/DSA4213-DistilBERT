# DSA4231-DistilBERT

## Dataset Setup

This project uses the **Diseases Articles** dataset from Kaggle:

- **Dataset**: [Diseases Articles](https://www.kaggle.com/datasets/shyshcuk/diseases-articles)
- **Description**: Collection of articles about various diseases
- **Task**: Binary classification (Medical articles vs Non-medical texts)

### Setup Instructions

#### Option 1: Automatic Download 
The script will automatically download the dataset using the Kaggle API:

```bash
# Install kaggle API
pip install kaggle

# Set up Kaggle API credentials
# 1. Go to https://www.kaggle.com/settings and create API token
# 2. Download kaggle.json
# 3. Place it in ~/.kaggle/ directory

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### Option 2: Synthetic Data
If the download fails, the code will automatically create a synthetic medical dataset for testing.
Run the script:

```bash
./scripts/run.sh
```

## Repository Structure
 - requirements.txt
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


