# DSA4231-DistilBERT
DistilBERT for Medical Sentiment Analysis

# Medical Tweet Classification with Transformers

This repository contains code for comparing different fine-tuning strategies (Full Fine-tuning vs LoRA) for adapting pretrained Transformer models to domain-specific text classification tasks.

## Project Overview

- **Task**: Binary text classification (Medical vs Non-Medical tweets)
- **Model**: DistilBERT-base-uncased
- **Dataset**: CardiffNLP Tweet Topic Single (adapted for medical classification)
- **Fine-tuning Strategies**:
  - Full Fine-tuning
  - LoRA (Low-Rank Adaptation)

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



## Quick Start

### Using the provided script
```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

## Dataset

The project uses the [CardiffNLP Tweet Topic Single]([url](https://huggingface.co/datasets/cardiffnlp/tweet_topic_single)) dataset from Hugging Face. The original multi-class topic classification is adapted to binary classification:

Label 1: Medical/Health tweets

Label 0: All other topics
