"""Tiny transformer model implementation."""

import numpy as np
from typing import Optional


class MultiHeadAttention:
    """Multi-head causal self-attention."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        """Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            max_seq_len: Maximum sequence length for causal mask
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
        
        self.causal_mask = np.tril(np.ones((max_seq_len, max_seq_len)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = np.where(mask[None, None, :, :] == 0, -1e9, scores)
        
        attn = self._softmax(scores, axis=-1)
        
        out = attn @ V
        
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        out = out @ self.W_o
        
        return out

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class FeedForward:
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int):
        """Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension
        """
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)
        out = h @ self.W2 + self.b2
        return out


class TransformerBlock:
    """Single transformer block with attention and feed-forward."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int):
        """Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
        """
        self.attn = MultiHeadAttention(d_model, n_heads, max_seq_len)
        self.ff = FeedForward(d_model, d_ff)
        
        self.ln1_g = np.ones(d_model)
        self.ln1_b = np.zeros(d_model)
        self.ln2_g = np.ones(d_model)
        self.ln2_b = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.attn.forward(self._layer_norm(x, self.ln1_g, self.ln1_b))
        x = x + self.ff.forward(self._layer_norm(x, self.ln2_g, self.ln2_b))
        return x

    def _layer_norm(self, x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / np.sqrt(var + eps) + b


class TinyTransformer:
    """Tiny transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 128,
    ):
        """Initialize tiny transformer.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_emb = np.random.randn(max_seq_len, d_model) * 0.02
        
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len)
            for _ in range(n_layers)
        ]
        
        self.ln_f_g = np.ones(d_model)
        self.ln_f_b = np.zeros(d_model)
        
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, idx: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            idx: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = idx.shape
        
        tok_emb = self.token_emb[idx]
        pos_emb = self.pos_emb[:seq_len]
        
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block.forward(x)
        
        x = self._layer_norm(x, self.ln_f_g, self.ln_f_b)
        
        logits = x @ self.lm_head
        
        return logits

    def _layer_norm(self, x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / np.sqrt(var + eps) + b

    def generate(
        self,
        idx: np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate tokens autoregressively.
        
        Args:
            idx: Starting token indices of shape (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            seed: Random seed for reproducibility
            
        Returns:
            Generated token indices of shape (batch_size, seq_len + max_new_tokens)
        """
        if seed is not None:
            np.random.seed(seed)
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            
            logits = self.forward(idx_cond)
            
            logits = logits[:, -1, :] / temperature
            
            probs = self._softmax(logits, axis=-1)
            
            idx_next = np.array([np.random.choice(self.vocab_size, p=probs[i]) for i in range(idx.shape[0])])
            idx_next = idx_next[:, None]
            
            idx = np.concatenate([idx, idx_next], axis=1)
        
        return idx

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        count = 0
        count += self.token_emb.size
        count += self.pos_emb.size
        
        for block in self.blocks:
            count += block.attn.W_q.size + block.attn.W_k.size
            count += block.attn.W_v.size + block.attn.W_o.size
            count += block.ff.W1.size + block.ff.b1.size
            count += block.ff.W2.size + block.ff.b2.size
            count += block.ln1_g.size + block.ln1_b.size
            count += block.ln2_g.size + block.ln2_b.size
        
        count += self.ln_f_g.size + self.ln_f_b.size
        count += self.lm_head.size
        
        return count
