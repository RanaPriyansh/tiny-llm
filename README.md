# tiny-llm

Train a tiny transformer on CPU and talk to it on localhost.

An educational implementation of a character-level transformer language model that you can train on your laptop and serve over HTTP. No GPU required.

## What This Is

- A working transformer implementation with multi-head causal self-attention
- Character-level tokenization
- Training loop with AdamW optimizer
- HTTP server to query trained models
- Runs entirely on CPU
- ~30K-600K parameters (configurable)

## What This Is Not

- Not a production LLM
- Not GPT-2 or any published model
- Not for GPU/CUDA training
- Not a tutorial fork of another project

See [NOTES.md](NOTES.md) for more details.

## Quick Start

### Clone and Install

```bash
git clone https://github.com/RanaPriyansh/tiny-llm.git
cd tiny-llm

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

All tests should pass. Tests include tokenizer, model architecture, training convergence, generation, and HTTP serving.

### Train a Model

```bash
python -m tiny_llm.train --steps 100 --checkpoint checkpoints/model.npz
```

This trains on a tiny built-in corpus. Training takes a minute or two on CPU. Options:

- `--steps N` - Number of training steps (default: 100)
- `--batch-size N` - Batch size (default: 8)
- `--seq-len N` - Sequence length (default: 32)
- `--d-model N` - Model dimension (default: 64)
- `--n-heads N` - Number of attention heads (default: 4)
- `--n-layers N` - Number of transformer blocks (default: 2)
- `--lr F` - Learning rate (default: 1e-3)
- `--checkpoint PATH` - Path to save checkpoint

### Serve the Model

```bash
python -m tiny_llm.serve --checkpoint checkpoints/model.npz --port 8765
```

The server binds to `127.0.0.1` only (localhost).

### Query the Server

Health check:

```bash
curl http://127.0.0.1:8765/health
```

Generate text:

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50, "temperature": 0.8}'
```

Response:

```json
{"text": "Hello world! This is a tiny..."}
```

## Architecture

- **Tokenizer**: Character-level, built from training corpus vocabulary
- **Model**: Token + positional embeddings, N transformer blocks, language modeling head
- **Transformer Block**: Pre-norm architecture with multi-head causal self-attention and position-wise feed-forward
- **Attention**: Causal masking, scaled dot-product, multi-head
- **Training**: Cross-entropy loss, AdamW optimizer
- **Generation**: Autoregressive sampling with temperature

## Requirements

- Python 3.8+
- NumPy (no PyTorch, TensorFlow, JAX, or other frameworks)
- CPU only

## License

MIT - see [LICENSE](LICENSE)

## Development

Run tests with verbose output:

```bash
pytest -v
```

Run a specific test:

```bash
pytest tests/test_tokenizer.py::test_tokenizer_roundtrip
```

## References

This is an educational implementation inspired by transformer architecture papers and learning resources. It is not derived from any single codebase.
