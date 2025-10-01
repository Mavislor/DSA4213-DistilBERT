"""
Model setup for both full fine-tuning and LoRA
"""

from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
import torch
from config.default_config import Config


class ModelSetup:
    def __init__(self, config: Config):
        self.config = config
    
    def setup_full_finetuning(self):
        """Setup model for full fine-tuning"""
        print("Setting up model for full fine-tuning...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.MODEL_NAME,
            num_labels=self.config.NUM_LABELS,
            id2label={0: "Non-Medical", 1: "Medical"},
            label2id={"Non-Medical": 0, "Medical": 1}
        )
        return model
    
    def setup_lora_finetuning(self):
        """Setup model for LoRA fine-tuning"""
        print("Setting up model for LoRA fine-tuning...")
        
        # First load the base model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.MODEL_NAME,
            num_labels=self.config.NUM_LABELS,
            id2label={0: "Non-Medical", 1: "Medical"},
            label2id={"Non-Medical": 0, "Medical": 1},
            torch_dtype=torch.float32
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            lora_dropout=self.config.LORA_DROPOUT,
            target_modules=["q_lin", "v_lin", "k_lin", "out_lin"]  # DistilBERT specific
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
