"""Tests for training module."""

import pytest
import numpy as np
from tiny_llm.model import TinyTransformer
from tiny_llm.trainer import Trainer, AdamW
from tiny_llm.tokenizer import CharTokenizer


def test_trainer_initialization():
    """Test trainer can be initialized."""
    model = TinyTransformer(vocab_size=30, d_model=32, n_heads=2, n_layers=1)
    trainer = Trainer(model)
    
    assert trainer.model is model
    assert trainer.optimizer is not None


def test_train_step_runs():
    """Test single training step executes."""
    model = TinyTransformer(vocab_size=30, d_model=32, n_heads=2, n_layers=1)
    trainer = Trainer(model, learning_rate=1e-3)
    
    batch_size = 4
    seq_len = 8
    x = np.random.randint(0, 30, size=(batch_size, seq_len))
    y = np.random.randint(0, 30, size=(batch_size, seq_len))
    
    loss = trainer.train_step(x, y)
    
    assert isinstance(loss, (float, np.floating))
    assert loss > 0


@pytest.mark.timeout(30)
def test_training_reduces_loss():
    """Test that training reduces loss on toy data."""
    np.random.seed(42)
    
    corpus = "abcdefghij" * 10
    tokenizer = CharTokenizer.from_text(corpus)
    data = np.array(tokenizer.encode(corpus))
    
    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=1,
        max_seq_len=16,
    )
    trainer = Trainer(model, learning_rate=1e-2)
    
    def get_batch(seq_len=8):
        ix = np.random.randint(0, len(data) - seq_len, size=2)
        x = np.stack([data[i:i+seq_len] for i in ix])
        y = np.stack([data[i+1:i+seq_len+1] for i in ix])
        return x, y
    
    losses = []
    for _ in range(20):
        x, y = get_batch()
        loss = trainer.train_step(x, y)
        losses.append(loss)
    
    initial_loss = np.mean(losses[:5])
    final_loss = np.mean(losses[-5:])
    
    assert final_loss < initial_loss


def test_adamw_optimizer():
    """Test AdamW optimizer step."""
    params = {
        "w": np.array([1.0, 2.0, 3.0]),
        "b": np.array([0.0]),
    }
    
    optimizer = AdamW(params, lr=0.01)
    
    grads = {
        "w": np.array([0.1, 0.2, 0.3]),
        "b": np.array([0.5]),
    }
    
    original_w = params["w"].copy()
    optimizer.step(grads)
    
    assert not np.allclose(params["w"], original_w)
