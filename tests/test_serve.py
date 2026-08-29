"""Tests for HTTP serving."""

import pytest
import json
import tempfile
import threading
import time
import numpy as np
from http.server import HTTPServer
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from tiny_llm.tokenizer import CharTokenizer
from tiny_llm.model import TinyTransformer
from tiny_llm.serve import ModelServer, RequestHandler


@pytest.fixture
def test_checkpoint(tmp_path):
    """Create a test checkpoint."""
    corpus = "abcdefghijklmnopqrstuvwxyz "
    tokenizer = CharTokenizer.from_text(corpus)
    
    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=1,
        max_seq_len=16,
    )
    
    checkpoint = {
        "model_config": {
            "vocab_size": tokenizer.vocab_size,
            "d_model": 32,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": model.d_ff,
            "max_seq_len": 16,
        },
        "vocab": tokenizer.vocab,
        "token_emb": model.token_emb,
        "pos_emb": model.pos_emb,
        "ln_f_g": model.ln_f_g,
        "ln_f_b": model.ln_f_b,
        "lm_head": model.lm_head,
    }
    
    for i, block in enumerate(model.blocks):
        checkpoint[f"block{i}_attn_Wq"] = block.attn.W_q
        checkpoint[f"block{i}_attn_Wk"] = block.attn.W_k
        checkpoint[f"block{i}_attn_Wv"] = block.attn.W_v
        checkpoint[f"block{i}_attn_Wo"] = block.attn.W_o
        checkpoint[f"block{i}_ff_W1"] = block.ff.W1
        checkpoint[f"block{i}_ff_b1"] = block.ff.b1
        checkpoint[f"block{i}_ff_W2"] = block.ff.W2
        checkpoint[f"block{i}_ff_b2"] = block.ff.b2
        checkpoint[f"block{i}_ln1_g"] = block.ln1_g
        checkpoint[f"block{i}_ln1_b"] = block.ln1_b
        checkpoint[f"block{i}_ln2_g"] = block.ln2_g
        checkpoint[f"block{i}_ln2_b"] = block.ln2_b
    
    checkpoint_path = tmp_path / "test_checkpoint.npz"
    np.savez(checkpoint_path, **checkpoint)
    
    return str(checkpoint_path)


def test_model_server_loads_checkpoint(test_checkpoint):
    """Test ModelServer can load checkpoint."""
    server = ModelServer(test_checkpoint)
    
    assert server.model is not None
    assert server.tokenizer is not None


def test_model_server_generate(test_checkpoint):
    """Test ModelServer can generate text."""
    server = ModelServer(test_checkpoint)
    
    result = server.generate("hello", max_tokens=10)
    
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.timeout(30)
def test_http_server_health_endpoint(test_checkpoint):
    """Test HTTP server /health endpoint."""
    server_instance = ModelServer(test_checkpoint)
    RequestHandler.server_instance = server_instance
    
    server = HTTPServer(("127.0.0.1", 0), RequestHandler)
    port = server.server_port
    
    def run_server():
        server.handle_request()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    time.sleep(0.1)
    
    try:
        response = urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
        data = json.loads(response.read())
        
        assert response.status == 200
        assert data["status"] == "healthy"
    finally:
        server.server_close()


@pytest.mark.timeout(30)
def test_http_server_generate_endpoint(test_checkpoint):
    """Test HTTP server /generate endpoint."""
    server_instance = ModelServer(test_checkpoint)
    RequestHandler.server_instance = server_instance
    
    server = HTTPServer(("127.0.0.1", 0), RequestHandler)
    port = server.server_port
    
    def run_server():
        for _ in range(2):
            server.handle_request()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    time.sleep(0.1)
    
    try:
        payload = json.dumps({"prompt": "hello", "max_tokens": 5}).encode()
        request = Request(
            f"http://127.0.0.1:{port}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        
        response = urlopen(request, timeout=10)
        data = json.loads(response.read())
        
        assert response.status == 200
        assert "text" in data
        assert isinstance(data["text"], str)
        assert len(data["text"]) > 0
    finally:
        server.server_close()


def test_model_server_empty_prompt(test_checkpoint):
    """Test server handles empty prompt."""
    server = ModelServer(test_checkpoint)
    result = server.generate("", max_tokens=10)
    
    assert result == ""
