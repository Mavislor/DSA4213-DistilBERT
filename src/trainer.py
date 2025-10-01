"""
Training and evaluation utilities
"""

import evaluate
import numpy as np
from transformers import (
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from config.default_config import Config


class MedicalTweetTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.metric = evaluate.load("accuracy")
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = self.metric.compute(predictions=predictions, references=labels)
        
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(labels, predictions, average='weighted')
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        
        return {
            "accuracy": accuracy["accuracy"],
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    
    def get_training_arguments(self, strategy_name):
        """Get training arguments for different strategies"""
        try:
            return TrainingArguments(
                output_dir=f"{self.config.OUTPUT_DIR}/{strategy_name}",
                overwrite_output_dir=True,
                learning_rate=self.config.LEARNING_RATE,
                per_device_train_batch_size=self.config.BATCH_SIZE,
                per_device_eval_batch_size=self.config.BATCH_SIZE,
                num_train_epochs=self.config.NUM_EPOCHS,
                weight_decay=self.config.WEIGHT_DECAY,
                warmup_steps=self.config.WARMUP_STEPS,
                logging_dir=f"{self.config.LOGGING_DIR}/{strategy_name}",
                logging_steps=100,
                eval_strategy="epoch",  
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
                greater_is_better=True,
                seed=self.config.SEED,
                report_to="tensorboard",
            )
        except TypeError:
            return TrainingArguments(
                output_dir=f"{self.config.OUTPUT_DIR}/{strategy_name}",
                overwrite_output_dir=True,
                learning_rate=self.config.LEARNING_RATE,
                per_device_train_batch_size=self.config.BATCH_SIZE,
                per_device_eval_batch_size=self.config.BATCH_SIZE,
                num_train_epochs=self.config.NUM_EPOCHS,
                weight_decay=self.config.WEIGHT_DECAY,
                warmup_steps=self.config.WARMUP_STEPS,
                logging_dir=f"{self.config.LOGGING_DIR}/{strategy_name}",
                logging_steps=100,
                evaluation_strategy="epoch",  
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
                greater_is_better=True,
                seed=self.config.SEED,
                report_to="tensorboard",
            )
    
    def train_model(self, model, tokenized_datasets, tokenizer, strategy_name):
        """Train the model with the given strategy"""
        print(f"Training with {strategy_name} strategy...")
        
        training_args = self.get_training_arguments(strategy_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
        # Evaluate on test set
        test_metrics = trainer.evaluate(tokenized_datasets["test"])
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        
        return trainer, test_metrics
