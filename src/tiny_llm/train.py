"""Training script for tiny transformer."""

import argparse
import numpy as np
from pathlib import Path
from .tokenizer import CharTokenizer
from .model import TinyTransformer
from .trainer import Trainer


TINY_CORPUS = """Hello world! This is a tiny corpus for training a tiny transformer.
The transformer learns to predict the next character in a sequence.
It uses multi-head self-attention and feed-forward layers.
Training on CPU is slow but educational.
This is not a production model, just a learning exercise.
"""


def get_batch(data: np.ndarray, batch_size: int, seq_len: int) -> tuple:
    """Get random batch of data.
    
    Args:
        data: Full dataset of token indices
        batch_size: Batch size
        seq_len: Sequence length
        
    Returns:
        Tuple of (inputs, targets)
    """
    ix = np.random.randint(0, len(data) - seq_len, size=batch_size)
    x = np.stack([data[i:i+seq_len] for i in ix])
    y = np.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train tiny transformer")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of transformer blocks")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to save checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("Initializing tokenizer...")
    tokenizer = CharTokenizer.from_text(TINY_CORPUS)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    print("Encoding corpus...")
    data = np.array(tokenizer.encode(TINY_CORPUS))
    print(f"Corpus length: {len(data)} tokens")
    
    print("Initializing model...")
    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
    )
    print(f"Model parameters: {model.get_num_params():,}")
    
    print("Initializing trainer...")
    trainer = Trainer(model, learning_rate=args.lr)
    
    print(f"\nTraining for {args.steps} steps...")
    losses = []
    for step in range(args.steps):
        x, y = get_batch(data, args.batch_size, args.seq_len)
        loss = trainer.train_step(x, y)
        losses.append(loss)
        
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step + 1}/{args.steps}, Loss: {loss:.4f}")
    
    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Average loss (last 10 steps): {np.mean(losses[-10:]):.4f}")
    
    print("\nGenerating sample text...")
    prompt = "Hello"
    prompt_ids = tokenizer.encode(prompt)
    input_ids = np.array([prompt_ids])
    
    generated_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8, seed=args.seed)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{generated_text}'")
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "d_ff": model.d_ff,
                "max_seq_len": args.seq_len,
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
        
        np.savez(checkpoint_path, **checkpoint)
        print(f"\nCheckpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
