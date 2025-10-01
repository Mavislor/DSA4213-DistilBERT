#!/bin/bash

# Medical Tweet Classification Experiment
# Run this script to reproduce the experiments

echo "Setting up Medical Tweet Classification Experiment..."

# Create virtual environment (optional)
# python -m venv med_tweet_env
# source med_tweet_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the experiment
echo "Starting experiment..."
python run_experiment.py

echo "Experiment completed! Check the outputs/ directory for results."
