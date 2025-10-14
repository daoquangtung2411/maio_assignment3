import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from scripts.api_v0_2 import app

client = TestClient(app)

def test_health_check():
    """
    
    Test health check endpoint
    
    """

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "v0.1"

def test_predict(monkeypatch):

    sample_input = {
        "age": 0.02,
        "sex": -0.044,
        "bmi": 0.06,
        "bp": -0.03,
        "s1": -0.02,
        "s2": 0.03,
        "s3": -0.02,
        "s4": 0.02,
        "s5": 0.02,"s6": -0.001
        }
    
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
