"""Export, Optimize, and Quantize the trained student model to ONNX."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from config import DistillConfig
from models.student import load_student
from opacus.validators import ModuleValidator

# ONNX Runtime optimization libraries
from onnxruntime.transformers import optimizer
from onnxruntime.quantization import quantize_dynamic, QuantType

def export(weights_path: str, onnx_path: str):
    cfg = DistillConfig()
    
    # 1. Load the base model + LoRA architecture
    model = load_student(cfg, device="cpu")
    model = ModuleValidator.fix(model)
    
    # 2. Load the trained DP weights
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # 3. Get the original unwrapped module to disable DP hooks
    raw_student = getattr(model, '_module', model)

    dummy_input = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))

    # Define paths for the intermediate and final quantized models
    raw_onnx_path = onnx_path.replace(".onnx", "_raw.onnx")
    optimized_onnx_path = onnx_path.replace(".onnx", "_opt.onnx")

    print("Phase 1: Exporting raw PyTorch graph to ONNX...")
    torch.onnx.export(
        raw_student,
        dummy_input,
        raw_onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"}, "logits": {0: "batch", 1: "seq_len"}},
        opset_version=17,
    )

    print("Phase 2: Fusing graph nodes (GeLU, LayerNorm, Attention)...")
    # Optimize the model specifically for GPT-2 architecture patterns
    opt_model = optimizer.optimize_model(
        raw_onnx_path,
        model_type='gpt2',
        num_heads=cfg.student_num_heads,
        hidden_size=cfg.student_hidden_size
    )
    opt_model.save_model_to_file(optimized_onnx_path)

    print("Phase 3: Applying Dynamic INT8 Quantization...")
    # Quantize the fused graph for massive CPU inference speedups
    quantize_dynamic(
        model_input=optimized_onnx_path,
        model_output=onnx_path,
        weight_type=QuantType.QUInt8
    )

    # Cleanup intermediate files to keep the outputs directory clean
    os.remove(raw_onnx_path)
    os.remove(optimized_onnx_path)

    print(f"Success! Highly optimized, quantized ONNX model saved → {onnx_path}")

if __name__ == "__main__":
    export("outputs/student.pt", "outputs/student.onnx")