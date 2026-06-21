import sys, os, threading, time, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
import uvicorn

from config import DistillConfig

app = FastAPI(title="PrivaDistill API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

cfg = DistillConfig()

# Shared state — written by background threads, read by /status
training_state = {
    "status": "idle",       # idle | running | completed | error
    "progress": 0,          # 0-100
    "epoch": 0,
    "epochs_total": cfg.num_epochs,
    "loss": None,
    "epsilon": None,
    "message": "",
}

# ── Request models ────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        if len(v) > 4096:
            raise ValueError("text exceeds 4096 character limit")
        return v

# ── Health / Config ───────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": os.path.exists(cfg.onnx_path),
        "training_status": training_state["status"],
    }

@app.get("/config")
def get_config():
    return {
        "model_name": cfg.teacher_model_name,
        "epsilon": cfg.dp_epsilon,
        "delta": cfg.dp_delta,
        "noise_multiplier": cfg.dp_noise_multiplier,
        "max_grad_norm": cfg.dp_max_grad_norm,
        "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size,
    }

# ── Training status (polled by frontend) ──────────────────────────
@app.get("/status")
def get_status():
    return training_state

# ── Inference ─────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if not os.path.exists(cfg.onnx_path):
        raise HTTPException(status_code=404, detail="ONNX model not found. Train and export first.")

    from inference.infer_onnx import run_inference
    start = time.time()
    try:
        generated_text = await asyncio.to_thread(
            run_inference, cfg.onnx_path, req.text, cfg.teacher_model_name, 20
        )
        return {
            "generated_text": generated_text,
            "inference_ms": round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Training ──────────────────────────────────────────────────────
def _training_thread():
    global training_state
    training_state.update({"status": "running", "progress": 0, "epoch": 0, "message": "Starting…"})
    try:
        from training.train import main as run_training
        run_training(progress_callback=_training_progress)
        training_state.update({"status": "completed", "progress": 100, "message": "Training complete."})
    except Exception as e:
        training_state.update({"status": "error", "message": str(e)})

def _training_progress(epoch: int, total: int, loss: float, epsilon: float):
    training_state.update({
        "epoch": epoch,
        "epochs_total": total,
        "loss": round(loss, 4),
        "epsilon": round(epsilon, 4),
        "progress": round((epoch / total) * 100),
        "message": f"Epoch {epoch}/{total} — loss {loss:.4f} — ε {epsilon:.2f}",
    })

@app.post("/train")
def train():
    if training_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Training already in progress.")
    threading.Thread(target=_training_thread, daemon=True).start()
    return {"status": "started"}

# ── Export ────────────────────────────────────────────────────────
@app.post("/export")
async def export_model():
    pt_path = os.path.join(cfg.output_dir, "student.pt")
    if not os.path.exists(pt_path):
        raise HTTPException(status_code=404, detail="PyTorch checkpoint not found. Train first.")
    try:
        from export.export_onnx import export
        await asyncio.to_thread(export, pt_path, cfg.onnx_path)
        return {"status": "success", "onnx_path": cfg.onnx_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Quantize ──────────────────────────────────────────────────────
@app.post("/quantize")
async def quantize():
    import torch
    if not torch.cuda.is_available():
        return {"status": "skipped", "message": "GPU not available — quantization runs on CUDA only."}
    try:
        from quantize.quantize_bnb import quantize_to_4bit
        path = os.path.join(cfg.output_dir, "student_4bit")
        await asyncio.to_thread(quantize_to_4bit, cfg.teacher_model_name, path)
        return {"status": "success", "saved_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Global error handler ──────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": str(exc)})

# Mount static LAST
public_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public")
app.mount("/", StaticFiles(directory=public_dir, html=True), name="public")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)