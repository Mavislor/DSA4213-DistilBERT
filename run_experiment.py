"""
Main entry point for the medical tweet classification experiment
"""

import os
from config.default_config import Config
from src.data_loader import DataLoader
from src.model_setup import ModelSetup
from src.trainer import MedicalTweetTrainer
from src.utils import set_seed, print_experiment_summary, setup_environment


def main():
    config = Config()
    
    setup_environment()
    set_seed(config.SEED)
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOGGING_DIR, exist_ok=True)
    
    data_loader = DataLoader(config)
    tokenized_datasets = data_loader.load_and_preprocess_data()
    
    label_distribution = data_loader.get_label_distribution(tokenized_datasets)
    print_experiment_summary(config, label_distribution)
    
    model_setup = ModelSetup(config)
    trainer = MedicalTweetTrainer(config)
    
    results = {}
    
    # Strategy 1: Full Fine-tuning
    print("\n" + "="*50)
    print("STRATEGY 1: FULL FINE-TUNING")
    print("="*50)
    
    full_model = model_setup.setup_full_finetuning()
    full_trainer, full_metrics = trainer.train_model(
        full_model, 
        tokenized_datasets, 
        data_loader.tokenizer,
        "full_finetuning"
    )
    results["full_finetuning"] = full_metrics
    
    # Strategy 2: LoRA Fine-tuning
    print("\n" + "="*50)
    print("STRATEGY 2: LoRA FINE-TUNING")
    print("="*50)
    
    lora_model = model_setup.setup_lora_finetuning()
    lora_trainer, lora_metrics = trainer.train_model(
        lora_model,
        tokenized_datasets,
        data_loader.tokenizer,
        "lora_finetuning"
    )
    results["lora_finetuning"] = lora_metrics
    
    # Print comparison
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    for strategy, metrics in results.items():
        print(f"\n{strategy.upper()}:")
        for metric, value in metrics.items():
            if metric.startswith("eval_"):
                print(f"  {metric[5:]}: {value:.4f}")
    
    import json
    with open(f"{config.OUTPUT_DIR}/experiment_results.json", "w") as f:
        # Convert numpy values to Python types for JSON serialization
        json_results = {}
        for strategy, metrics in results.items():
            json_results[strategy] = {k: float(v) for k, v in metrics.items()}
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {config.OUTPUT_DIR}/experiment_results.json")


if __name__ == "__main__":
    main()
