"""Load a frozen, highly optimized teacher model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_teacher(model_name: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Memory Optimization: Load the Teacher in FP16 if on GPU
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()

    # Freeze all parameters — the teacher never trains
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)

    # Speed Optimization: Compile the PyTorch graph for faster forward passes (PyTorch 2.0+)
    if hasattr(torch, "compile") and sys.platform != "win32":
        try:
            print("Compiling Teacher model graph for faster inference...")
            model = torch.compile(model)
        except Exception as e:
            print(f"Graph compilation skipped: {e}")

    return model, tokenizer