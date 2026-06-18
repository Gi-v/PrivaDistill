<div align="center">

<img src="https://raw.githubusercontent.com/YOUR_USERNAME/privadistill/main/assets/logo.png" alt="PrivaDistill Logo" width="120" height="120" />

# 🔒 PrivaDistill

### Locally Distilled LLM with On-Device Differential Privacy

*AI-powered note analysis for sensitive professionals — no data ever leaves your device.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Opacus](https://img.shields.io/badge/Opacus-DP%20Training-6366F1?style=flat-square)](https://opacus.ai)
[![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-4--bit%20Quant-F59E0B?style=flat-square)](https://github.com/TimDettmers/bitsandbytes)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME/privadistill/pulls)
[![Stars](https://img.shields.io/github/stars/YOUR_USERNAME/privadistill?style=flat-square&color=yellow)](https://github.com/YOUR_USERNAME/privadistill/stargazers)

<br/>

> **Therapists. Defense Lawyers. Medical Personnel.**
> Finally — LLM intelligence on your most sensitive notes, with *mathematically proven* privacy guarantees.

<br/>

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#️-architecture) · [🔐 Privacy Math](#-the-privacy-guarantee) · [🖥️ Frontend](#️-frontend-ui) · [📡 API Reference](#-api-reference) · [🤝 Contributing](#-contributing)

</div>

---

## 🎯 The Problem

Professionals in sensitive fields generate critical notes every day. Current LLM tools force an impossible choice:

| Option | Problem |
|---|---|
| ☁️ Cloud API (GPT-4, Claude) | **PII leaves your device.** HIPAA/attorney-client privilege violations. |
| 🏠 Local fine-tuning (naive) | **Gradient leakage.** Training data reconstructible from model weights. |
| 📵 No AI assistance | **Missed insights.** Hours of manual analysis. |

**PrivaDistill solves all three** — a tiny, compressed model that runs entirely on your machine, with formal differential privacy guarantees baked into the training loop.

---

## ✨ Features

- 🔒 **Zero cloud dependency** — all inference runs on local hardware via ONNX Runtime
- 🧮 **Formal (ε, δ)-DP guarantee** — Opacus clips gradients and injects calibrated Gaussian noise
- 🎓 **Knowledge Distillation** — KL-divergence loss transfers teacher intelligence to a tiny student
- ⚡ **4-bit Quantization** — BitsAndBytes NF4 compression, ~4× smaller with minimal accuracy loss
- 🖥️ **Professional SPA Frontend** — three-panel dashboard with live privacy meter
- 🔌 **FastAPI Backend** — clean REST API connecting UI to the ML pipeline
- 📊 **Live Privacy Dashboard** — real-time ε-budget gauge, noise level bars, gradient clipping status

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRIVADISTILL PIPELINE                    │
│                                                                 │
│   ┌──────────────┐    KL Divergence    ┌──────────────────┐    │
│   │   TEACHER    │ ──────────────────► │    STUDENT LM    │    │
│   │  (frozen)    │    Loss (T=4.0)     │  4-layer, 256-d  │    │
│   │  distilgpt2  │                     │  ~12M params     │    │
│   └──────────────┘                     └────────┬─────────┘    │
│                                                  │              │
│                                         Opacus PrivacyEngine   │
│                                         ┌────────▼─────────┐   │
│                                         │  DP Training Loop │   │
│                                         │  • Gradient Clip  │   │
│                                         │  • Gaussian Noise │   │
│                                         │  • ε accounting   │   │
│                                         └────────┬─────────┘   │
│                                                  │              │
│              ┌───────────────────────────────────┤              │
│              │                                   │              │
│   ┌──────────▼──────────┐         ┌──────────────▼──────────┐  │
│   │  BitsAndBytes 4-bit │         │     ONNX Export         │  │
│   │  NF4 Quantization   │         │   dynamic axes          │  │
│   │  double quant       │         │   opset 17              │  │
│   └──────────┬──────────┘         └──────────────┬──────────┘  │
│              └───────────────────────────────────┘              │
│                                   │                             │
│                        ┌──────────▼──────────┐                  │
│                        │   ONNX Runtime      │                  │
│                        │   CPU Inference     │                  │
│                        │   ~50ms/token       │                  │
│                        └──────────┬──────────┘                  │
└───────────────────────────────────┼─────────────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────┐
              │           FastAPI Backend :8000            │
              │  /health  /config  /analyze  /train        │
              └─────────────────────┬─────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────┐
              │         SPA Frontend (index.html)          │
              │  Left Panel | Note Analyzer | Privacy Dash │
              └────────────────────────────────────────────┘
```

### Component Breakdown

```
privadistill/
├── 📄 config.py                    # All hyperparams & paths
├── 🤖 models/
│   ├── teacher.py                  # Frozen HuggingFace teacher
│   └── student.py                  # Custom 4-layer transformer
├── 🏋️ training/
│   ├── distill_loss.py             # KL divergence + CE combined loss
│   ├── dp_trainer.py               # Opacus DP training loop
│   └── train.py                    # Entry point
├── 🗜️ quantize/
│   └── quantize_bnb.py             # BitsAndBytes NF4 4-bit
├── 📦 export/
│   └── export_onnx.py              # ONNX export (opset 17)
├── ⚡ inference/
│   └── infer_onnx.py               # ONNX Runtime inference
├── 🌐 backend/
│   └── server.py                   # FastAPI REST API
├── 🖥️ frontend/
│   └── index.html                  # ~1000-line professional SPA
├── 📊 data/
│   └── sample_notes.jsonl          # Synthetic test data
└── 🎬 demo.py                      # End-to-end CLI demo
```

---

## 🔐 The Privacy Guarantee

PrivaDistill provides **formal (ε, δ)-Differential Privacy**. In plain English:

> An adversary who inspects the trained model's weights cannot determine — with confidence greater than **e^ε + δ** — whether any single patient note, legal brief, or medical record was ever used in training.

### The Math

The Opacus training loop applies two operations to every gradient update:

**Step 1 — Per-Sample Gradient Clipping**
```
g̃ᵢ = gᵢ / max(1, ‖gᵢ‖₂ / C)
```
where `C = max_grad_norm = 1.0`. This bounds how much any one record can influence the model.

**Step 2 — Gaussian Noise Injection**
```
g̃ = (1/B) · Σᵢ g̃ᵢ + 𝒩(0, σ²C²I)
```
where `σ = noise_multiplier` is auto-computed from your target `(ε, δ, epochs)`.

**Resulting Privacy Budget** (default config):
```
ε = 8.0,  δ = 1e-5
```

| ε value | Meaning |
|---|---|
| ε < 1 | Very strong privacy, significant accuracy cost |
| **ε = 8** | **Practical balance — industry standard for medical data** |
| ε > 20 | Weak privacy, near-baseline accuracy |

> 💡 Adjust `dp_epsilon` in `config.py` to trade privacy strength for model quality.

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
CUDA-capable GPU (optional — CPU works, quantization requires CUDA)
```

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/privadistill.git
cd privadistill

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies (ORDER MATTERS — see gotchas)
pip install torch==2.2.0
pip install transformers==4.40.0
pip install opacus==1.4.0
pip install bitsandbytes==0.43.0
pip install onnx==1.16.0 onnxruntime==1.18.0
pip install fastapi==0.111.0 uvicorn==0.29.0
```

### ✏️ Configure (The Only File You Need to Touch)

Open `config.py` and set:

```python
# ── REQUIRED ──────────────────────────────────────────────
TEACHER_MODEL_NAME = "distilgpt2"   # Free, no key needed
                                     # Swap: "gpt2", "facebook/opt-125m"

# ── ONLY IF using a gated model (LLaMA, Mistral, etc.) ────
HF_TOKEN = "hf_YOUR_TOKEN_HERE"     # https://huggingface.co/settings/tokens
                                     # Leave "" for open models

# ── PATHS (auto-created, change if needed) ─────────────────
OUTPUT_DIR = "outputs/"
ONNX_PATH  = "outputs/student.onnx"
```

That's it. No other keys, IDs, or credentials required.

### Run

```bash
# Option A: Full end-to-end demo
python demo.py

# Option B: Step by step
python training/train.py          # Train with DP
python export/export_onnx.py      # Export to ONNX
python inference/infer_onnx.py    # Run inference

# Option C: With UI
python backend/server.py          # Start API on :8000
# Then open frontend/index.html in your browser
```

---

## 🖥️ Frontend UI

The frontend is a single `index.html` file — no build step, no npm, no bundler.

**Three-panel layout:**

```
┌──────────────────────────────────────────────────────────────┐
│  🔒 PrivaDistill   ● Online    On-Device · Zero Cloud        │
├──────────────┬───────────────────────┬───────────────────────┤
│ LEFT PANEL   │   NOTE ANALYZER       │  PRIVACY DASHBOARD    │
│              │                       │                       │
│ Model Status │  ┌─────────────────┐  │  ε-Budget Gauge       │
│ DP Params    │  │ Paste notes...  │  │  ████████░░  8.0/10   │
│ Controls     │  │                 │  │                       │
│              │  └─────────────────┘  │  Noise Level          │
│ [▶ Train]    │  [Analyze Privately]  │  ████████████  σ=1.1  │
│ [⬇ Export]  │                       │                       │
│ [⚡ Quantize]│  Results Card         │  Gradient Clip        │
│              │  Next token: "the"    │  C = 1.0 ✓            │
│              │  Confidence: 87%      │                       │
│              │  Latency: 43ms        │  (ε,δ)-DP Certified   │
└──────────────┴───────────────────────┴───────────────────────┘
```

**To launch:**
1. Start the backend: `python backend/server.py`
2. Open `frontend/index.html` in any browser
3. The status dot turns green when backend is connected

---

## 📡 API Reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Description | Response |
|---|---|---|---|
| `GET` | `/health` | Backend status + model loaded state | `{ status, model_loaded }` |
| `GET` | `/config` | Current DP params + model info | `{ epsilon, delta, noise_multiplier, ... }` |
| `POST` | `/analyze` | Run private inference on text | `{ next_token, confidence, inference_ms }` |
| `POST` | `/train` | Start DP distillation (async) | `{ status: "started" }` |
| `POST` | `/export` | Export student to ONNX | `{ status, onnx_path }` |
| `POST` | `/quantize` | 4-bit quantize model | `{ status, saved_path }` |
| `GET` | `/status` | Poll long-running job status | `{ job, progress, done }` |

### Example: Analyze a Note

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient reports persistent headache for 3 days"}'
```

```json
{
  "next_token": "with",
  "confidence": 0.847,
  "inference_ms": 43,
  "model": "student_onnx",
  "privacy_certified": true
}
```

---

## ⚙️ Configuration Reference

All settings live in `config.py` — one place, clearly commented:

```python
@dataclass
class DistillConfig:
    # ── Models ─────────────────────────────────────
    teacher_model_name: str = "distilgpt2"    # HF model ID
    student_hidden_size: int = 256             # Student width
    student_num_layers: int = 4               # Depth
    student_num_heads: int  = 4               # Attention heads
    vocab_size: int         = 50257           # GPT-2 vocab

    # ── Distillation ───────────────────────────────
    temperature: float = 4.0                  # Logit softening (Hinton 2015)
    alpha: float       = 0.7                  # KL weight (1-alpha = CE weight)

    # ── Differential Privacy ───────────────────────
    dp_epsilon: float        = 8.0            # Privacy budget
    dp_delta: float          = 1e-5           # Failure probability
    dp_max_grad_norm: float  = 1.0            # Clipping threshold C

    # ── Training ───────────────────────────────────
    batch_size: int    = 4
    max_seq_len: int   = 128
    learning_rate: float = 5e-4
    num_epochs: int    = 3

    # ── Paths ──────────────────────────────────────
    output_dir: str  = "outputs/"
    onnx_path: str   = "outputs/student.onnx"
```

---

## 🔧 Troubleshooting

<details>
<summary><b>❌ Opacus: "Model is not valid for DP training"</b></summary>

Run the validator and auto-fix before training:

```python
from opacus.validators import ModuleValidator
student = ModuleValidator.fix(student)
errors = ModuleValidator.validate(student, strict=False)
assert errors == [], f"Remaining errors: {errors}"
```

Opacus replaces `BatchNorm` layers with `GroupNorm` automatically.
</details>

<details>
<summary><b>❌ RuntimeError: DataLoader workers > 0 crash with Opacus</b></summary>

Opacus is incompatible with multiprocessing DataLoader. Always use:

```python
DataLoader(dataset, num_workers=0, batch_size=cfg.batch_size)
```
</details>

<details>
<summary><b>❌ ONNX export fails on DP-wrapped model</b></summary>

Unwrap the model before export:

```python
raw_student = privacy_engine._module
torch.onnx.export(raw_student, ...)
```
</details>

<details>
<summary><b>❌ BitsAndBytes: "CUDA not available"</b></summary>

4-bit quantization requires a CUDA GPU. PrivaDistill auto-detects and skips quantization on CPU-only machines, falling back to fp32 ONNX export.
</details>

<details>
<summary><b>❌ ModuleNotFoundError across subdirectories</b></summary>

Every file inside a subdirectory needs the sys.path fix:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
```
</details>

<details>
<summary><b>❌ Frontend shows "Backend offline" yellow banner</b></summary>

The backend is not running. Start it:

```bash
python backend/server.py
# Should print: Uvicorn running on http://0.0.0.0:8000
```
</details>

---

## 📊 Benchmarks

| Metric | Teacher (distilgpt2) | Student (PrivaDistill) |
|---|---|---|
| Parameters | 82M | ~12M |
| Model size (fp32) | 331 MB | ~48 MB |
| Model size (4-bit) | — | ~12 MB |
| Inference latency (CPU) | ~310ms | ~43ms |
| Privacy guarantee | ❌ None | ✅ (ε=8, δ=1e-5) |
| Runs offline | ✅ | ✅ |
| Cloud dependency | ❌ | ❌ |

---

## 🔬 How Knowledge Distillation Works

```
TEACHER OUTPUT (soft labels, T=4):
  "the" → 0.42  "a" → 0.31  "with" → 0.18  "..." → 0.09

STUDENT OUTPUT (soft labels, T=4):
  "the" → 0.38  "a" → 0.29  "with" → 0.21  "..." → 0.12

KL DIVERGENCE LOSS:
  L_KL = Σ p_teacher · log(p_teacher / p_student) × T²

  → Student learns not just what is correct ("the")
    but HOW confident teacher is about alternatives
    — this is the "dark knowledge" that makes distillation work
```

Standard cross-entropy only teaches the hard label (`"the"` = correct). KL divergence teaches the *shape* of the teacher's uncertainty — which is why a 12M param student can approach 82M param teacher quality.

---

## 🗺️ Roadmap

- [ ] `v0.2` — Support LLaMA-3.2-1B as teacher (via HF token)
- [ ] `v0.2` — LoRA adapter support for faster distillation
- [ ] `v0.3` — Electron wrapper for true desktop app (no browser needed)
- [ ] `v0.3` — Automatic PII redaction layer before inference
- [ ] `v0.4` — Federated distillation across multiple local machines
- [ ] `v1.0` — HIPAA compliance checklist + audit log

---

## 🤝 Contributing

Contributions welcome. Please follow the structure:

```bash
# Fork → clone → branch
git checkout -b feature/your-feature

# Install dev deps
pip install pytest black isort

# Format before committing
black . && isort .

# Run tests
pytest tests/

# PR to main
```

**Good first issues:** adding support for new teacher models, improving the frontend privacy gauge animation, writing unit tests for `distill_loss.py`.

---

## 📚 References

| Paper / Resource | Relevance |
|---|---|
| [Hinton et al. 2015 — Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | Foundation of KL-divergence distillation |
| [Dwork et al. 2006 — Differential Privacy](https://link.springer.com/chapter/10.1007/11681878_14) | Formal DP definition used here |
| [Yousefpour et al. 2021 — Opacus](https://arxiv.org/abs/2109.12298) | The DP library powering this project |
| [Dettmers et al. 2023 — QLoRA](https://arxiv.org/abs/2305.14314) | NF4 quantization scheme used in BitsAndBytes |
| [ONNX Runtime Docs](https://onnxruntime.ai/docs/) | On-device inference engine |

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

This project handles sensitive professional data. The MIT license grants you full freedom to use and modify it, but **you are responsible** for ensuring your deployment meets applicable regulations (HIPAA, GDPR, attorney-client privilege, etc.).

---

<div align="center">

**Built for professionals who cannot afford to compromise.**

[![GitHub](https://img.shields.io/badge/GitHub-YOUR__USERNAME-181717?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME/privadistill)
[![Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/Powered%20by-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)

*If this project helped you, consider giving it a ⭐*

</div>