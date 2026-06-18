"""High-performance on-device inference using ONNX Runtime with Top-K sampling."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

def top_k_sampling(logits, k=50, temperature=0.7):
    """Applies temperature scaling and Top-K filtering to logits for natural generation."""
    # Scale by temperature (higher = more random, lower = more deterministic)
    logits = logits / temperature
    
    # Keep only the top K probabilities, zero out the rest
    indices_to_remove = logits < np.sort(logits)[-k]
    logits[indices_to_remove] = -float('Inf')
    
    # Convert to probabilities using softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # Sample from the filtered distribution
    return np.random.choice(len(probs), p=probs)

def run_inference(onnx_path: str, text: str, teacher_name: str = "distilgpt2", max_new_tokens: int = 50):
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)

    # Enable advanced ONNX Runtime optimizations
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        onnx_path, 
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )
    
    generated_tokens = []
    
    # Autoregressive generation loop with Top-K Sampling
    for _ in range(max_new_tokens):
        # Run the fused, quantized graph
        logits = session.run(["logits"], {"input_ids": input_ids})[0]
        
        # Extract the logits for the very last token in the sequence
        next_token_logits = logits[0, -1, :]
        
        # Sample the next token rather than taking the strict maximum
        next_token_id = top_k_sampling(next_token_logits, k=40, temperature=0.8)
        generated_tokens.append(next_token_id)
        
        # Stop generating if the model outputs an End-Of-Sentence token
        if next_token_id == tokenizer.eos_token_id:
            break
            
        # Append the new token back into the input sequence
        input_ids = np.append(input_ids, [[next_token_id]], axis=1)

    predicted_text = tokenizer.decode(generated_tokens)
    return predicted_text

if __name__ == "__main__":
    result = run_inference("outputs/student.onnx", "Patient reports persistent")
    print(f"\nFinal Output: {result}")