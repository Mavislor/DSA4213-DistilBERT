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
 - README.md
 - requirements.txt
 - run_experiment.py
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
