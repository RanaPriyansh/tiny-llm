"""Tests for tokenizer module."""

import pytest
from tiny_llm.tokenizer import CharTokenizer


def test_tokenizer_encode_decode():
    """Test tokenizer can encode and decode text."""
    text = "Hello, world!"
    tokenizer = CharTokenizer.from_text(text)
    
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    assert decoded == text


def test_tokenizer_vocab_size():
    """Test tokenizer correctly counts vocabulary."""
    text = "aabbcc"
    tokenizer = CharTokenizer.from_text(text)
    
    assert tokenizer.vocab_size == 3


def test_tokenizer_from_text():
    """Test tokenizer creation from text corpus."""
    text = "The quick brown fox"
    tokenizer = CharTokenizer.from_text(text)
    
    assert tokenizer.vocab_size > 0
    assert "T" in tokenizer.char_to_idx
    assert "q" in tokenizer.char_to_idx


def test_tokenizer_roundtrip():
    """Test encoding and decoding is lossless."""
    corpus = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 !?,."
    tokenizer = CharTokenizer.from_text(corpus)
    
    test_text = "Hello123"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    assert decoded == test_text


def test_tokenizer_encode_returns_list():
    """Test encode returns list of integers."""
    tokenizer = CharTokenizer.from_text("abc")
    result = tokenizer.encode("abc")
    
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


def test_tokenizer_decode_returns_string():
    """Test decode returns string."""
    tokenizer = CharTokenizer.from_text("abc")
    result = tokenizer.decode([0, 1, 2])
    
    assert isinstance(result, str)
