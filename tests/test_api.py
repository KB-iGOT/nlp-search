import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app

client = TestClient(app)

# Mock response from Vertex AI
def mock_generate_content(*args, **kwargs):
    mock_response = MagicMock()
    # The app expects response.text to be valid JSON string
    mock_response.text = '{"keywords": [{"keyword": "test keyword", "priority": 1, "synonyms": ["test syn"]}]}'
    return [mock_response] 

@patch("vertexai.generative_models.GenerativeModel.generate_content")
def test_search_valid(mock_generate, monkeypatch):
    """Test a valid search request."""
    # Mock the LLM call
    mock_generate.side_effect = mock_generate_content
    
    response = client.post(
        "/nlp/search",
        json={"query": "test query", "synonyms": True}
    )
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "keywords" in data["data"]
    assert data["data"]["keywords"][0]["keyword"] == "test keyword"

def test_search_invalid_length():
    """Test validation for query too long."""
    long_query = "a" * 501
    response = client.post(
        "/nlp/search",
        json={"query": long_query, "synonyms": False}
    )
    assert response.status_code == 422
    assert "Query too long" in response.text

def test_search_injection_chars():
    """Test validation for injection characters."""
    bad_query = "valid query {{ invalid"
    response = client.post(
        "/nlp/search",
        json={"query": bad_query, "synonyms": False}
    )
    assert response.status_code == 422
    assert "Query contains invalid characters" in response.text

@patch("vertexai.generative_models.GenerativeModel.generate_content")
def test_vapt_payload_safe(mock_generate):
    """Test that code injection payload is treated as text (mocked behavior simulation)."""
    # Here we are testing that the request passes validation and reaches the LLM service.
    # The actual LLM behavior (not executing code) was verified manually with curl.
    # But we want to ensure this payload doesn't crash our app or validation logic.
    
    mock_generate.side_effect = mock_generate_content
    
    payload = "\")\"\"testzzz\".replace(\"z\",\"a\")\"\""
    response = client.post(
        "/nlp/search",
        json={"query": payload, "synonyms": False}
    )
    assert response.status_code == 200
    # Ensuring it didn't trigger validation error or 500 error
