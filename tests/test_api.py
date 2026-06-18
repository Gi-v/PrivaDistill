"""Automated Unit Tests for the PrivaDistill API"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from backend.server import app

# Initialize the test client
client = TestClient(app)

def test_health_check():
    """Verify the health endpoint responds and correctly identifies model state."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data

def test_config_endpoint():
    """Verify the privacy configuration is accurately exposed to the frontend."""
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    
    # Check for critical Differential Privacy parameters
    assert "epsilon" in data
    assert "delta" in data
    assert "noise_multiplier" in data
    assert "max_grad_norm" in data

def test_analyze_guardrail():
    """Verify the API correctly handles inference requests when no model exists."""
    response = client.post("/analyze", json={"text": "Patient reports persistent headache."})
    assert response.status_code == 200
    data = response.json()
    
    # Since the CI pipeline doesn't run a 3-hour training loop, 
    # it should gracefully return our predefined error state.
    assert "error" in data
    assert "ONNX model not found" in data["error"]

def test_public_directory_mount():
    """Verify the unified stack is serving the frontend UI on the root path."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PrivaDistill" in response.text