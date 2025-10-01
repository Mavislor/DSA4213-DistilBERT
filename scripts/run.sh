#!/bin/bash

# Medical Tweet Classification Experiment
# Run this script to reproduce the experiments

echo "Setting up Medical Tweet Classification Experiment..."

# Use python3 directly 
echo "Python version:"
python3 --version

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install -r requirements.txt

# Run the experiment
echo "Starting experiment..."
python3 run_experiment.py

echo "Experiment completed! Check the outputs/ directory for results."
