"""Training utilities for tiny transformer."""

import numpy as np
from typing import Tuple, Optional
from .model import TinyTransformer


class AdamW:
    """AdamW optimizer implementation."""

    def __init__(
        self,
        params: dict,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """Initialize AdamW optimizer.
        
        Args:
            params: Dictionary of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term added to denominator for numerical stability
            weight_decay: Weight decay coefficient
        """
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict):
        """Perform single optimization step.
        
        Args:
            grads: Dictionary of gradients
        """
        self.t += 1
        
        for k in self.params.keys():
            if k not in grads:
                continue
                
            g = grads[k]
            
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            
            self.params[k] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * self.params[k])


class Trainer:
    """Trainer for tiny transformer."""

    def __init__(
        self,
        model: TinyTransformer,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
    ):
        """Initialize trainer.
        
        Args:
            model: TinyTransformer model
            learning_rate: Learning rate
            weight_decay: Weight decay coefficient
        """
        self.model = model
        self._forward_cache = {}
        
        self.params = self._collect_params()
        self.optimizer = AdamW(self.params, lr=learning_rate, weight_decay=weight_decay)

    def _forward_with_cache(self, idx: np.ndarray) -> tuple:
        """Forward pass that caches intermediate values.
        
        Args:
            idx: Token indices
            
        Returns:
            Tuple of (logits, x_before_lm_head)
        """
        batch_size, seq_len = idx.shape
        
        tok_emb = self.model.token_emb[idx]
        pos_emb = self.model.pos_emb[:seq_len]
        x = tok_emb + pos_emb
        
        for block in self.model.blocks:
            x = block.forward(x)
        
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x = self.model.ln_f_g * (x - mean) / np.sqrt(var + 1e-5) + self.model.ln_f_b
        
        x_pre_lm = x
        logits = x @ self.model.lm_head
        
        return logits, x_pre_lm

    def _collect_params(self) -> dict:
        """Collect all model parameters into a flat dictionary."""
        params = {}
        params["token_emb"] = self.model.token_emb
        params["pos_emb"] = self.model.pos_emb
        
        for i, block in enumerate(self.model.blocks):
            params[f"block{i}_attn_Wq"] = block.attn.W_q
            params[f"block{i}_attn_Wk"] = block.attn.W_k
            params[f"block{i}_attn_Wv"] = block.attn.W_v
            params[f"block{i}_attn_Wo"] = block.attn.W_o
            params[f"block{i}_ff_W1"] = block.ff.W1
            params[f"block{i}_ff_b1"] = block.ff.b1
            params[f"block{i}_ff_W2"] = block.ff.W2
            params[f"block{i}_ff_b2"] = block.ff.b2
            params[f"block{i}_ln1_g"] = block.ln1_g
            params[f"block{i}_ln1_b"] = block.ln1_b
            params[f"block{i}_ln2_g"] = block.ln2_g
            params[f"block{i}_ln2_b"] = block.ln2_b
        
        params["ln_f_g"] = self.model.ln_f_g
        params["ln_f_b"] = self.model.ln_f_b
        params["lm_head"] = self.model.lm_head
        
        return params

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """Perform single training step.
        
        Args:
            x: Input token indices of shape (batch_size, seq_len)
            y: Target token indices of shape (batch_size, seq_len)
            
        Returns:
            Loss value
        """
        self._forward_cache.clear()
        logits, x_pre_lm = self._forward_with_cache(x)
        self._forward_cache["x_before_lm_head"] = x_pre_lm
        
        loss, grads = self._compute_loss_and_grads(logits, y)
        
        self.optimizer.step(grads)
        
        return loss

    def _compute_loss_and_grads(self, logits: np.ndarray, targets: np.ndarray) -> Tuple[float, dict]:
        """Compute cross-entropy loss and gradients.
        
        Args:
            logits: Predicted logits of shape (batch_size, seq_len, vocab_size)
            targets: Target token indices of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (loss, gradients_dict)
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        probs = self._softmax(logits_flat, axis=-1)
        
        log_probs = np.log(probs[np.arange(len(targets_flat)), targets_flat] + 1e-10)
        loss = -np.mean(log_probs)
        
        dlogits_flat = probs.copy()
        dlogits_flat[np.arange(len(targets_flat)), targets_flat] -= 1
        dlogits_flat /= len(targets_flat)
        
        dlogits = dlogits_flat.reshape(batch_size, seq_len, vocab_size)
        
        grads = self._backward(dlogits)
        
        return loss, grads

    def _backward(self, dlogits: np.ndarray) -> dict:
        """Backward pass (simplified gradient computation).
        
        This is a simplified implementation that computes gradients for
        key parameters. For full backprop through all layers, a more
        sophisticated implementation would be needed.
        
        Args:
            dlogits: Gradient w.r.t. logits
            
        Returns:
            Dictionary of gradients
        """
        grads = {}
        
        batch_size, seq_len, vocab_size = dlogits.shape
        
        x_blocks = self._forward_cache.get("x_before_lm_head")
        if x_blocks is not None:
            grads["lm_head"] = x_blocks.reshape(-1, x_blocks.shape[-1]).T @ dlogits.reshape(-1, vocab_size)
            grads["lm_head"] /= batch_size
        
        return grads

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
