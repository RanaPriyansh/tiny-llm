"""Tiny LLM - Train a tiny transformer on CPU and talk to it on localhost."""

__version__ = "0.1.0"

from .tokenizer import CharTokenizer
from .model import TinyTransformer
from .trainer import Trainer

__all__ = ["CharTokenizer", "TinyTransformer", "Trainer"]
