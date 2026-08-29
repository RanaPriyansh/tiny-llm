# Notes

## What This Project Is

`tiny-llm` is an educational implementation of a transformer language model that:

- Trains a small transformer (30K-600K parameters) on CPU
- Uses character-level tokenization
- Implements multi-head causal self-attention from scratch
- Provides an HTTP server to interact with trained models
- Runs entirely on CPU without GPU dependencies

The goal is to demonstrate a complete end-to-end pipeline: tokenization → training → generation → serving. This is a learning tool for understanding transformer architectures.

## What This Project Is Not

### Not a Production LLM

This is a toy implementation for educational purposes. The model is far too small to produce coherent long-form text. It will overfit on tiny corpora and produce nonsense on anything else.

### Not GPU-Ready

This implementation uses NumPy and runs on CPU only. There is no CUDA, GPU acceleration, or distributed training. Training large models would be impractically slow.

### Not GPT-2 or Any Published Model

This is not a reimplementation of GPT-2, GPT-3, or any other specific published model. The architecture is inspired by the general transformer design but is its own minimal implementation.

### Not a Fork

This is original educational code, not a fork of `rasbt/LLMs-from-scratch`, `karpathy/nanoGPT`, or other projects. It may share conceptual similarities (all transformers do), but the implementation is independent.

## Architecture Details

### Tokenizer

Character-level tokenization: each unique character in the corpus becomes a token. Simple but limited to small vocabularies.

### Model Structure

- Token embeddings: `vocab_size × d_model`
- Position embeddings: `max_seq_len × d_model`
- N transformer blocks:
  - Layer norm
  - Multi-head causal self-attention (with residual)
  - Layer norm
  - Feed-forward network (with residual)
- Final layer norm
- Language modeling head: `d_model × vocab_size`

### Training

- Loss: Cross-entropy on next-token prediction
- Optimizer: AdamW (Adam with decoupled weight decay)
- Data: Random batches from a tokenized corpus

### Serving

HTTP server with:
- `GET /health` - Health check
- `POST /generate` - Generate text from prompt

Server binds to `127.0.0.1` only. No authentication (this is a local demo tool).

## Limitations

### Small Models Only

The model is designed to be trained in seconds/minutes on CPU. Anything beyond ~1M parameters becomes impractically slow.

### Poor Text Quality

Character-level models with this few parameters produce low-quality text. The model will memorize tiny corpora and generate nonsense on larger ones.

### No Advanced Features

This implementation does not include:
- Byte-pair encoding or other modern tokenization
- Flash attention or other optimizations
- Gradient accumulation or mixed precision
- Distributed training
- Model parallelism
- Fine-tuning utilities
- Prompt engineering tools

## Use Cases

This project is suitable for:
- Learning how transformers work end-to-end
- Experimenting with attention mechanisms
- Understanding training loops and generation
- Seeing how to serve a model over HTTP

This project is **not** suitable for:
- Generating production-quality text
- Any real NLP task
- Benchmarking model performance
- Training on large datasets

## Extending This Project

If you want to build on this:

1. **Better tokenization**: Add BPE or WordPiece
2. **More data**: Expand beyond toy corpora
3. **GPU support**: Port to PyTorch/JAX for GPU acceleration
4. **Larger models**: Scale up parameters (requires GPU)
5. **Evaluation**: Add perplexity, benchmarks
6. **Advanced sampling**: Add top-k, top-p, beam search

## Safety Notes

- This server binds to localhost only (`127.0.0.1`)
- No authentication or rate limiting
- Do not expose to the internet
- Do not send sensitive data through this model
- Checkpoints are saved as NumPy `.npz` files (no security scanning)

## License

MIT - see LICENSE file.
