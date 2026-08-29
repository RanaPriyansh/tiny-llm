"""Tests for model module."""

import pytest
import numpy as np
from tiny_llm.model import TinyTransformer, MultiHeadAttention


def test_model_forward_shape():
    """Test model forward pass produces correct output shape."""
    vocab_size = 50
    batch_size = 2
    seq_len = 16
    d_model = 32
    
    model = TinyTransformer(vocab_size=vocab_size, d_model=d_model, n_heads=2, n_layers=1)
    
    idx = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    logits = model.forward(idx)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_model_generate():
    """Test model can generate tokens."""
    vocab_size = 30
    model = TinyTransformer(vocab_size=vocab_size, d_model=32, n_heads=2, n_layers=1)
    
    start_idx = np.array([[0, 1, 2]])
    generated = model.generate(start_idx, max_new_tokens=5, seed=42)
    
    assert generated.shape[0] == 1
    assert generated.shape[1] == 8


def test_model_parameter_count():
    """Test model parameter counting."""
    model = TinyTransformer(vocab_size=50, d_model=32, n_heads=2, n_layers=1)
    num_params = model.get_num_params()
    
    assert num_params > 0
    assert isinstance(num_params, int)


def test_attention_causal_mask():
    """Test attention uses causal masking correctly."""
    d_model = 16
    n_heads = 2
    max_seq_len = 8
    
    attn = MultiHeadAttention(d_model, n_heads, max_seq_len)
    
    mask = attn.causal_mask
    
    assert mask.shape == (max_seq_len, max_seq_len)
    assert np.all(np.tril(mask) == mask)


def test_attention_forward_shape():
    """Test attention forward pass produces correct shape."""
    d_model = 16
    n_heads = 2
    batch_size = 2
    seq_len = 4
    
    attn = MultiHeadAttention(d_model, n_heads, max_seq_len=8)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    out = attn.forward(x)
    
    assert out.shape == (batch_size, seq_len, d_model)


def test_model_different_temperatures():
    """Test model generation with different temperatures."""
    vocab_size = 30
    model = TinyTransformer(vocab_size=vocab_size, d_model=32, n_heads=2, n_layers=1)
    
    start_idx = np.array([[0, 1, 2]])
    
    gen_low = model.generate(start_idx.copy(), max_new_tokens=10, temperature=0.1, seed=42)
    gen_high = model.generate(start_idx.copy(), max_new_tokens=10, temperature=2.0, seed=43)
    
    assert gen_low.shape == gen_high.shape
