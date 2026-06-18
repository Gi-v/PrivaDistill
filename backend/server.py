import sys, os, threading, time, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from config import DistillConfig

app = FastAPI(title="PrivaDistill API")

# CORS is now largely unnecessary since the UI and API share an origin, 
# but we keep it locked down for safety.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"], 
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

cfg = DistillConfig()
training_state = {"status": "idle", "epochs_done": 0, "final_epsilon": None}

class AnalyzeRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(cfg.onnx_path)}

@app.get("/config")
def get_config():
    return {
        "epsilon": cfg.dp_epsilon,
        "delta": cfg.dp_delta,
        "noise_multiplier": cfg.dp_noise_multiplier,
        "max_grad_norm": cfg.dp_max_grad_norm,
        "model_name": cfg.teacher_model_name
    }

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Asynchronous generation endpoint to prevent server lockup."""
    from inference.infer_onnx import run_inference
    
    if not os.path.exists(cfg.onnx_path):
        return {"error": "ONNX model not found. Train and export first."}
    
    start_time = time.time()
    try:
        # Offload the blocking ONNX inference to a background thread
        generated_text = await asyncio.to_thread(
            run_inference,
            cfg.onnx_path,
            req.text,
            cfg.teacher_model_name,
            20 # max_new_tokens
        )
        
        inference_ms = (time.time() - start_time) * 1000
        return {"generated_text": generated_text, "inference_ms": round(inference_ms, 2)}
    
    except Exception as e:
        return {"error": str(e)}

def _run_training_thread():
    global training_state
    training_state["status"] = "running"
    try:
        from training.train import main
        main()
        training_state["status"] = "completed"
        training_state["final_epsilon"] = cfg.dp_epsilon
    except Exception as e:
        training_state["status"] = f"error: {str(e)}"

@app.post("/train")
def train():
    if training_state["status"] == "running":
        return {"status": "already running"}
    thread = threading.Thread(target=_run_training_thread)
    thread.start()
    return {"status": "started"}

@app.post("/export")
def export_model():
    from export.export_onnx import export
    pt_path = os.path.join(cfg.output_dir, "student.pt")
    if not os.path.exists(pt_path):
        return {"error": "PyTorch model not found. Train first."}
    export(pt_path, cfg.onnx_path)
    return {"status": "success", "onnx_path": cfg.onnx_path}

@app.post("/quantize")
def quantize():
    import torch
    if torch.cuda.is_available():
        from quantize.quantize_bnb import quantize_to_4bit
        path = os.path.join(cfg.output_dir, "student_4bit")
        quantize_to_4bit(cfg.teacher_model_name, path)
        return {"status": "success", "saved_path": path}
    return {"status": "skipped", "message": "CPU only"}

# CRITICAL: Mount the static directory LAST so it doesn't intercept API routes
public_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public")
app.mount("/", StaticFiles(directory=public_dir, html=True), name="public")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)