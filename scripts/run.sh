#!/bin/bash

echo "Setting up Medical Tweet Classification Experiment..."

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install -r requirements.txt

# Run the experiment
echo "Starting experiment..."
python3 run_experiment.py

echo "Experiment completed! Check the outputs/ directory for results."
