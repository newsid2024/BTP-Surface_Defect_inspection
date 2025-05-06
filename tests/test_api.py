import os
import pytest
import io
import json
from api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test the health check endpoint"""
    response = client.get('/health')
    data = json.loads(response.data)
    assert response.status_code == 200
    assert 'status' in data
    assert data['status'] == 'healthy'

def test_predict_no_file(client):
    """Test predict endpoint with no file"""
    response = client.post('/predict')
    data = json.loads(response.data)
    assert response.status_code == 400
    assert 'error' in data
    assert data['error'] == 'No file part'

def test_predict_empty_file(client):
    """Test predict endpoint with empty file"""
    response = client.post('/predict', data={
        'file': (io.BytesIO(), '')
    })
    data = json.loads(response.data)
    assert response.status_code == 400
    assert 'error' in data
    assert data['error'] == 'No selected file'

def test_predict_invalid_filetype(client):
    """Test predict endpoint with invalid file type"""
    response = client.post('/predict', data={
        'file': (io.BytesIO(b'test data'), 'test.txt')
    }, content_type='multipart/form-data')
    data = json.loads(response.data)
    assert response.status_code == 400
    assert 'error' in data
    assert data['error'] == 'Invalid file type'

def test_batch_predict_no_file(client):
    """Test batch predict endpoint with no file"""
    response = client.post('/batch_predict')
    data = json.loads(response.data)
    assert response.status_code == 400
    assert 'error' in data
    assert data['error'] == 'No files part'

def test_batch_predict_empty_file(client):
    """Test batch predict endpoint with empty file"""
    response = client.post('/batch_predict', data={
        'files': (io.BytesIO(), '')
    })
    data = json.loads(response.data)
    assert response.status_code == 400
    assert 'error' in data
    assert data['error'] == 'No selected files' 