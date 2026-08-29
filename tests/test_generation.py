"""Tests for text generation."""

import pytest
import numpy as np
from tiny_llm.model import TinyTransformer
from tiny_llm.tokenizer import CharTokenizer


def test_generation_returns_text():
    """Test that generation returns non-empty text."""
    corpus = "hello world"
    tokenizer = CharTokenizer.from_text(corpus)
    
    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=1,
    )
    
    prompt = "hello"
    prompt_ids = tokenizer.encode(prompt)
    input_ids = np.array([prompt_ids])
    
    generated_ids = model.generate(input_ids, max_new_tokens=10, seed=42)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    assert isinstance(generated_text, str)
    assert len(generated_text) > len(prompt)


def test_generation_deterministic_with_seed():
    """Test generation is deterministic with same seed."""
    tokenizer = CharTokenizer.from_text("abc")
    model = TinyTransformer(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=2, n_layers=1)
    
    start_idx = np.array([[0]])
    
    gen1 = model.generate(start_idx.copy(), max_new_tokens=5, seed=42)
    gen2 = model.generate(start_idx.copy(), max_new_tokens=5, seed=42)
    
    assert np.array_equal(gen1, gen2)


def test_generation_produces_valid_tokens():
    """Test generation produces valid token indices."""
    vocab_size = 20
    model = TinyTransformer(vocab_size=vocab_size, d_model=32, n_heads=2, n_layers=1)
    
    start_idx = np.array([[0, 1]])
    generated = model.generate(start_idx, max_new_tokens=10, seed=42)
    
    assert np.all(generated >= 0)
    assert np.all(generated < vocab_size)
