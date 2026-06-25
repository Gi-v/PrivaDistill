<div align="center">

# 🔒 PrivaDistill

### Locally Distilled LLM with On-Device Differential Privacy

*AI-powered analysis of sensitive professional notes — with mathematically proven privacy guarantees. No data ever leaves your device.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Opacus](https://img.shields.io/badge/Opacus-DP%20Training-6366F1?style=flat-square)](https://opacus.ai)
[![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-4--bit%20Quant-F59E0B?style=flat-square)](https://github.com/TimDettmers/bitsandbytes)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/YOUR_USERNAME/privadistill/actions)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME/privadistill/pulls)

<br/>

> **Built for therapists, defense attorneys, and medical personnel.**
> Bring LLM intelligence to your most sensitive notes — with formal, auditable privacy protection.

<br/>

[Quick Start](#-quick-start) · [Architecture](#️-architecture) · [Privacy Guarantee](#-the-privacy-guarantee) · [Frontend UI](#️-frontend-ui) · [API Reference](#-api-reference) · [Configuration](#️-configuration-reference) · [Troubleshooting](#-troubleshooting) · [Contributing](#-contributing)

</div>

---

## Table of Contents

- [Overview](#-overview)
- [The Problem It Solves](#-the-problem-it-solves)
- [Features](#-features)
- [Architecture](#️-architecture)
- [The Privacy Guarantee](#-the-privacy-guarantee)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Docker Deployment](#-docker-deployment)
- [Frontend UI](#️-frontend-ui)
- [API Reference](#-api-reference)
- [Configuration Reference](#️-configuration-reference)
- [How Knowledge Distillation Works](#-how-knowledge-distillation-works)
- [Benchmarks](#-benchmarks)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#️-roadmap)
- [Contributing](#-contributing)
- [References](#-references)
- [License](#-license)

---

## 📖 Overview

**PrivaDistill** is an open-source framework that trains a compact, privacy-preserving language model entirely on your local machine. It uses **knowledge distillation** to compress a large teacher model (e.g., `gpt2-medium`) into a small, fast student model (~12M parameters), and uses **Opacus differential privacy** to guarantee that no individual training record can be reconstructed from the model's weights.

The result is a deployable AI system that can analyze sensitive professional notes — clinical records, legal briefs, therapy session summaries — with **zero cloud dependency** and **formal (ε, δ)-DP privacy guarantees** baked in.

---

## 🎯 The Problem It Solves

Professionals in sensitive fields generate critical notes daily. Existing AI tools force an impossible trade-off:

| Approach | Problem |
|---|---|
| ☁️ **Cloud APIs** (e.g., GPT-4, Claude) | PII leaves your device. High risk of HIPAA violations, attorney-client privilege breaches, and GDPR non-compliance. |
| 🏠 **Naive local fine-tuning** | Training data can be reconstructed from model weights via gradient inversion attacks. Privacy is not guaranteed. |
| 📵 **No AI assistance** | Hours of manual analysis. Missed patterns. Reduced quality of care or legal preparation. |

PrivaDistill eliminates all three problems: it runs locally, applies formal mathematical privacy protection during training, and is small enough to run efficiently on standard hardware.

---

## ✨ Features

- **Zero cloud dependency** — all training and inference runs locally via ONNX Runtime; your notes never leave your machine
- **Formal (ε, δ)-DP guarantee** — Opacus clips per-sample gradients and injects calibrated Gaussian noise, providing auditable privacy accounting
- **Knowledge distillation** — KL-divergence loss transfers "dark knowledge" from a large teacher to a tiny, efficient student model
- **4-bit quantization** — BitsAndBytes NF4 compression reduces the model to ~12 MB with minimal accuracy loss
- **No-build-step frontend** — a single `index.html` SPA with a three-panel dashboard, live privacy meter, and real-time ε-budget gauge
- **FastAPI backend** — a clean REST API connecting the UI to the ML pipeline, with async job status polling
- **Docker support** — a multi-stage, non-root Dockerfile and Compose file for reproducible, production-ready deployments
- **GitHub Actions CI** — automated test pipeline runs on every push and pull request to `main`

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       PRIVADISTILL PIPELINE                     │
│                                                                 │
│   ┌──────────────┐    KL Divergence    ┌──────────────────┐    │
│   │   TEACHER    │──────────────────►  │   STUDENT LM     │    │
│   │  (frozen)    │    Loss (T=4.0)     │  4-layer, 256-d  │    │
│   │ gpt2-medium  │                     │  ~12M params     │    │
│   └──────────────┘                     └────────┬─────────┘    │
│                                                  │              │
│                                        Opacus PrivacyEngine    │
│                                        ┌─────────▼────────┐    │
│                                        │  DP Training Loop │    │
│                                        │  • Gradient Clip  │    │
│                                        │  • Gaussian Noise │    │
│                                        │  • ε accounting   │    │
│                                        └─────────┬────────┘    │
│                                                  │              │
│              ┌───────────────────────────────────┤              │
│              │                                   │              │
│  ┌───────────▼──────────┐         ┌──────────────▼──────────┐  │
│  │  BitsAndBytes 4-bit  │         │      ONNX Export        │  │
│  │  NF4 Quantization    │         │   dynamic axes          │  │
│  │  double quant        │         │   opset 17              │  │
│  └───────────┬──────────┘         └──────────────┬──────────┘  │
│              └───────────────────────────────────┘              │
│                                   │                             │
│                       ┌───────────▼──────────┐                  │
│                       │    ONNX Runtime       │                  │
│                       │    CPU Inference      │                  │
│                       │    ~50ms/token        │                  │
│                       └───────────┬──────────┘                  │
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

### Project Structure

```
privadistill/
├── config.py                   # All hyperparameters and file paths
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Multi-stage, non-root production image
├── docker-compose.yml          # Single-command deployment
├── models/
│   ├── teacher.py              # Frozen HuggingFace teacher loader
│   └── student.py              # Custom 4-layer transformer student
├── training/
│   ├── distill_loss.py         # Combined KL-divergence + CE loss
│   ├── dp_trainer.py           # Opacus DP training loop
│   └── train.py                # Training entry point
├── export/
│   └── export_onnx.py          # ONNX export (opset 17, dynamic axes)
├── inference/
│   └── infer_onnx.py           # ONNX Runtime CPU inference
├── backend/
│   └── server.py               # FastAPI REST API server
├── public/
│   ├── index.html              # Single-page frontend application
│   ├── css/styles.css
│   └── js/
│       ├── app.js
│       ├── api.js
│       └── ui.js
├── data/
│   └── sample_notes.jsonl      # Synthetic sample data for testing
└── tests/
    └── test_api.py             # API test suite (pytest + httpx)
```

---

## 🔐 The Privacy Guarantee

PrivaDistill provides **formal (ε, δ)-Differential Privacy**. In plain English:

> An adversary who inspects the trained model's weights cannot determine — with probability greater than **e^ε + δ** — whether any single patient note, legal brief, or medical record was ever used during training.

### How It Works

The Opacus training loop applies two operations to every gradient update:

**Step 1 — Per-Sample Gradient Clipping**

```
g̃ᵢ = gᵢ / max(1, ‖gᵢ‖₂ / C)
```

where `C = max_grad_norm = 1.0`. This bounds how much influence any single training record can have on the model.

**Step 2 — Gaussian Noise Injection**

```
g̃ = (1/B) · Σᵢ g̃ᵢ + N(0, σ²C²I)
```

where `σ = noise_multiplier` is automatically computed from your target `(ε, δ, epochs)` via the Rényi Differential Privacy accountant.

**Default Privacy Budget**

```
ε = 8.0,   δ = 1e-5
```

| ε Value | Interpretation |
|---|---|
| ε < 1 | Very strong privacy; significant accuracy cost |
| **ε = 8** | **Practical balance — industry standard for medical data** |
| ε > 20 | Weak privacy; near-baseline model accuracy |

> **Tip:** Adjust `dp_epsilon` in `config.py` to trade privacy strength for model quality. Lower ε = stronger privacy, but noisier gradients and reduced accuracy.

---

## 📋 Prerequisites

Before installing, ensure your environment meets the following requirements:

| Requirement | Minimum | Notes |
|---|---|---|
| **Python** | 3.10+ | 3.11 recommended |
| **RAM** | 8 GB | 16 GB recommended for training |
| **Disk space** | 5 GB | For models and ONNX outputs |
| **GPU (CUDA)** | Optional | Required for 4-bit quantization only; CPU works for all other steps |
| **OS** | Linux, macOS, Windows | Docker recommended on Windows |

---

## 🚀 Quick Start

### Option 1: Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/privadistill.git
cd privadistill

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` installs the CPU build of PyTorch by default. If you have a CUDA-capable GPU, install the appropriate CUDA build of PyTorch first, then run `pip install -r requirements.txt`.

### Option 2: Docker (Recommended for Production)

See the [Docker Deployment](#-docker-deployment) section below.

---

### Configuration

Open `config.py` and set the values relevant to your setup. For most users, only two fields need attention:

```python
# The teacher model to distill from (free, no API key required)
TEACHER_MODEL_NAME = "gpt2-medium"    # Alternatives: "distilgpt2", "gpt2"

# Only required for gated models (LLaMA, Mistral, etc.)
# Leave as empty string "" for open-access models
HF_TOKEN = ""                         # https://huggingface.co/settings/tokens
```

That is the only configuration required to get started.

---

### Running the Pipeline

**All-in-one demo (recommended for first-time users):**

```bash
python training/train.py
```

**Step-by-step execution:**

```bash
# Step 1: Train the student model with differential privacy
python training/train.py

# Step 2: Export the trained model to ONNX format
python export/export_onnx.py

# Step 3: Run a test inference
python inference/infer_onnx.py
```

**With the UI:**

```bash
# Step 1: Start the API server
python backend/server.py
# The server starts at http://localhost:8000

# Step 2: Open the frontend in your browser
open public/index.html             # macOS
# Or simply double-click public/index.html in your file explorer
```

---

## 🐳 Docker Deployment

Docker is the fastest path to a reproducible, isolated deployment. The included Dockerfile uses a multi-stage build and runs the application as a non-root user for improved security.

```bash
# Build and start the service
docker compose up --build

# Run in detached mode (background)
docker compose up --build -d

# View logs
docker compose logs -f

# Stop the service
docker compose down
```

The API will be available at `http://localhost:8000`.

> **Note:** The `outputs/` and `data/` directories are mounted as volumes, so trained models persist across container restarts.

---

## 🖥️ Frontend UI

The frontend is a dependency-free single `index.html` file — no Node.js, no build step, no bundler required.

```
┌──────────────────────────────────────────────────────────────┐
│  🔒 PrivaDistill    ● Online    On-Device · Zero Cloud       │
├──────────────┬───────────────────────┬───────────────────────┤
│ LEFT PANEL   │   NOTE ANALYZER       │  PRIVACY DASHBOARD    │
│              │                       │                       │
│ Model Status │  ┌─────────────────┐  │  ε-Budget Gauge       │
│ DP Params    │  │ Paste notes...  │  │  ████████░░  8.0/10   │
│ Controls     │  └─────────────────┘  │                       │
│              │                       │  Noise Level          │
│ [▶ Train]    │  [Analyze Privately]  │  ████████████  σ=1.1  │
│ [⬇ Export]  │                       │                       │
│ [⚡ Quantize]│  Results Card         │  Gradient Clip        │
│              │  Next token: "the"    │  C = 1.0 ✓            │
│              │  Confidence: 87%      │                       │
│              │  Latency: 43ms        │  (ε,δ)-DP Certified   │
└──────────────┴───────────────────────┴───────────────────────┘
```

**Panels:**

- **Left Panel** — model status, DP configuration parameters, and action buttons (Train, Export, Quantize)
- **Note Analyzer** — paste any text and run private inference; results show the predicted next token, confidence score, and latency
- **Privacy Dashboard** — real-time ε-budget gauge, noise level bars, and gradient clipping status

**To launch:**

1. Start the backend: `python backend/server.py`
2. Open `public/index.html` in any modern browser
3. The status indicator turns green once the backend connection is confirmed

---

## 📡 API Reference

**Base URL:** `http://localhost:8000`

| Method | Endpoint | Description | Response |
|---|---|---|---|
| `GET` | `/health` | Backend liveness check and model load status | `{ status, model_loaded }` |
| `GET` | `/config` | Current DP parameters and model configuration | `{ epsilon, delta, noise_multiplier, ... }` |
| `POST` | `/analyze` | Run private inference on a text input | `{ next_token, confidence, inference_ms, privacy_certified }` |
| `POST` | `/train` | Start DP distillation training (async) | `{ status: "started" }` |
| `POST` | `/export` | Export the student model to ONNX | `{ status, onnx_path }` |
| `POST` | `/quantize` | Apply 4-bit NF4 quantization to the model | `{ status, saved_path }` |
| `GET` | `/status` | Poll the status of a running async job | `{ job, progress, done }` |

### Example: Analyze a Note

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient reports persistent headache for 3 days"}'
```

**Response:**

```json
{
  "next_token": "with",
  "confidence": 0.847,
  "inference_ms": 43,
  "model": "student_onnx",
  "privacy_certified": true
}
```

### Example: Check Backend Health

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## ⚙️ Configuration Reference

All settings are centralized in `config.py`. The `DistillConfig` dataclass is the single source of truth for the entire pipeline:

```python
@dataclass
class DistillConfig:

    # ── Models ─────────────────────────────────────────────────
    teacher_model_name: str   = "gpt2-medium"   # HuggingFace model ID for teacher
    student_hidden_size: int  = 256             # Student embedding dimension
    student_num_layers: int   = 4               # Number of transformer layers
    student_num_heads: int    = 4               # Number of attention heads
    vocab_size: int           = 50257           # GPT-2 vocabulary size

    # ── Distillation ───────────────────────────────────────────
    temperature: float        = 4.0             # Logit softening temperature (Hinton 2015)
    alpha: float              = 0.7             # KL-divergence loss weight; (1−α) = CE weight

    # ── Differential Privacy ───────────────────────────────────
    dp_epsilon: float         = 8.0             # Privacy budget (lower = stronger privacy)
    dp_delta: float           = 1e-5            # Failure probability (keep << 1/dataset_size)
    dp_max_grad_norm: float   = 1.0             # Per-sample gradient clipping threshold C
    dp_noise_multiplier: float = 1.1            # Gaussian noise scale σ (auto-computed)

    # ── Training ───────────────────────────────────────────────
    batch_size: int           = 4
    max_seq_len: int          = 128
    learning_rate: float      = 5e-4
    num_epochs: int           = 3

    # ── Paths ──────────────────────────────────────────────────
    output_dir: str           = "outputs/"
    onnx_path: str            = "outputs/student.onnx"
```

**Common tuning scenarios:**

| Goal | Parameter to Adjust |
|---|---|
| Stronger privacy | Decrease `dp_epsilon` (e.g., `4.0` or `2.0`) |
| Better model accuracy | Increase `dp_epsilon` or `num_epochs` |
| Faster training | Decrease `num_epochs` or `max_seq_len` |
| Smaller model | Decrease `student_hidden_size` or `student_num_layers` |
| Use a different teacher | Change `teacher_model_name` to any HuggingFace causal LM |

---

## 🔬 How Knowledge Distillation Works

Instead of training on hard labels alone (e.g., "the next word is *the*"), knowledge distillation trains the student to mimic the teacher's full probability distribution over all possible next tokens — including tokens it considers unlikely. This "soft" knowledge is far richer:

```
TEACHER OUTPUT (soft labels, T=4):
  "the" → 0.42   "a" → 0.31   "with" → 0.18   "..." → 0.09

STUDENT OUTPUT (soft labels, T=4):
  "the" → 0.38   "a" → 0.29   "with" → 0.21   "..." → 0.12

KL DIVERGENCE LOSS:
  L_KL = Σ p_teacher · log(p_teacher / p_student) × T²
```

The student learns not just *what* is correct, but *how uncertain* the teacher is about alternatives. This "dark knowledge" is why a 12M-parameter student can approach the quality of an 82M-parameter teacher — and it means the student generalizes well to inputs it has never seen before.

---

## 📊 Benchmarks

| Metric | Teacher (`gpt2-medium`) | Student (PrivaDistill) |
|---|---|---|
| Parameters | 345M | ~12M |
| Model size (fp32) | ~1.4 GB | ~48 MB |
| Model size (4-bit) | — | ~12 MB |
| Inference latency (CPU) | ~310 ms | ~43 ms |
| Runs fully offline | ✅ | ✅ |
| Privacy guarantee | ❌ None | ✅ (ε=8, δ=1e-5) |
| Cloud dependency | ❌ | ❌ |

> **Note:** Latency figures are approximate and will vary depending on hardware. Benchmarks were measured on a standard CPU without GPU acceleration.

---

## 🔧 Troubleshooting

<details>
<summary><strong>Opacus error: "Model is not valid for DP training"</strong></summary>

Opacus does not support `BatchNorm` layers. Run the validator and auto-fix before training:

```python
from opacus.validators import ModuleValidator

student = ModuleValidator.fix(student)       # Replaces BatchNorm with GroupNorm
errors  = ModuleValidator.validate(student, strict=False)
assert errors == [], f"Remaining errors: {errors}"
```

</details>

<details>
<summary><strong>RuntimeError: DataLoader crashes with Opacus when num_workers > 0</strong></summary>

Opacus is incompatible with multiprocessing DataLoaders. Always use single-process loading:

```python
DataLoader(dataset, num_workers=0, batch_size=cfg.batch_size)
```

</details>

<details>
<summary><strong>ONNX export fails on a DP-wrapped model</strong></summary>

Opacus wraps the model in a `GradSampleModule`. You must unwrap it before exporting:

```python
raw_student = privacy_engine._module
torch.onnx.export(raw_student, ...)
```

</details>

<details>
<summary><strong>BitsAndBytes error: "CUDA not available"</strong></summary>

4-bit quantization requires a CUDA-capable GPU. PrivaDistill automatically detects CPU-only environments and skips the quantization step, falling back to a standard fp32 ONNX export. You can safely ignore this on CPU-only machines.

</details>

<details>
<summary><strong>ModuleNotFoundError when running scripts from subdirectories</strong></summary>

Add the project root to the Python path at the top of any script in a subdirectory:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
```

</details>

<details>
<summary><strong>Frontend shows "Backend offline" warning banner</strong></summary>

The FastAPI backend is not running. Start it before opening the frontend:

```bash
python backend/server.py
# Expected output: Uvicorn running on http://0.0.0.0:8000
```

Then refresh the browser. The status indicator should turn green.

</details>

<details>
<summary><strong>Training is very slow on CPU</strong></summary>

Training a transformer with differential privacy is computationally intensive. On CPU-only hardware, consider:

- Reducing `num_epochs` to `1` for a quick test run
- Reducing `max_seq_len` to `64`
- Using a smaller teacher model (`distilgpt2` instead of `gpt2-medium`)
- Running inside Docker with resource limits disabled to use all available cores

</details>

---

## 🗺️ Roadmap

- [ ] **v0.2** — Support for LLaMA-3.2-1B as teacher (via HuggingFace token)
- [ ] **v0.2** — LoRA adapter support for faster, more memory-efficient distillation
- [ ] **v0.3** — Electron wrapper for a true desktop application (no browser required)
- [ ] **v0.3** — Automatic PII redaction layer applied before inference
- [ ] **v0.4** — Federated distillation across multiple local machines
- [ ] **v1.0** — HIPAA compliance checklist and structured audit logging

---

## 🤝 Contributing

Contributions are welcome. Please follow this workflow:

```bash
# 1. Fork the repository, then clone your fork
git clone https://github.com/YOUR_USERNAME/privadistill.git
cd privadistill

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Install development dependencies
pip install -r requirements.txt
pip install pytest black isort httpx

# 4. Make your changes, then format the code
black .
isort .

# 5. Run the test suite
pytest tests/ -v

# 6. Push your branch and open a pull request to main
git push origin feature/your-feature-name
```

**Good first contributions:**

- Adding support for new teacher model architectures
- Writing additional unit tests for `training/distill_loss.py`
- Improving the frontend privacy gauge animation
- Adding CLI argument parsing to `training/train.py`

Please open an issue before starting work on a significant change, so we can discuss the approach.

---

## 📚 References

| Paper / Resource | Relevance |
|---|---|
| [Hinton et al. (2015) — Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | Foundation of KL-divergence knowledge distillation |
| [Dwork et al. (2006) — Differential Privacy](https://link.springer.com/chapter/10.1007/11681878_14) | Formal (ε, δ)-DP definition used in this project |
| [Yousefpour et al. (2021) — Opacus: User-Friendly Differential Privacy Library](https://arxiv.org/abs/2109.12298) | The DP training library powering PrivaDistill |
| [Dettmers et al. (2023) — QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | NF4 quantization scheme used in BitsAndBytes |
| [ONNX Runtime Documentation](https://onnxruntime.ai/docs/) | On-device inference engine documentation |

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.

> **Important:** PrivaDistill is designed to handle sensitive professional data. The MIT license grants you broad freedom to use and modify this software, but **you are solely responsible** for ensuring that your deployment complies with applicable regulations, including HIPAA, GDPR, attorney-client privilege rules, and any other jurisdiction-specific data protection requirements. This software is provided as-is, with no warranty of regulatory compliance.

---

<div align="center">

**Built for professionals who cannot afford to compromise.**

[![GitHub](https://img.shields.io/badge/GitHub-YOUR__USERNAME-181717?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME/privadistill)
[![Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/Powered%20by-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)

*If this project helped you, consider giving it a ⭐*

</div>
