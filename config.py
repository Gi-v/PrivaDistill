from dataclasses import dataclass

TEACHER_MODEL_NAME = "gpt2-medium"
STUDENT_MODEL_NAME = "distilgpt2"
HF_TOKEN = ""
OUTPUT_DIR = "outputs/"

@dataclass
class DistillConfig:
    teacher_model_name: str = TEACHER_MODEL_NAME
    student_model_name: str = STUDENT_MODEL_NAME
    student_hidden_size: int = 256
    student_num_layers: int = 4
    student_num_heads: int = 4
    vocab_size: int = 50257

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    temperature: float = 4.0
    alpha: float = 0.7

    dp_epsilon: float = 8.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 1.1

    batch_size: int = 4
    max_seq_len: int = 128
    learning_rate: float = 5e-4
    num_epochs: int = 3

    output_dir: str = OUTPUT_DIR
    onnx_path: str = f"{OUTPUT_DIR}student.onnx"