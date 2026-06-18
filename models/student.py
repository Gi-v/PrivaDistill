import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from config import DistillConfig

def load_student(cfg: DistillConfig, device: str = "cpu"):
    # Load the base model in standard precision
    model = AutoModelForCausalLM.from_pretrained(
        cfg.student_model_name, 
        torch_dtype=torch.float32
    )
    
    # Configure LoRA to target only the attention layers
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["c_attn"] # Specific to GPT-2 style architectures
    )
    
    # Wrap the model and move it to the target device
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    
    return peft_model.to(device)