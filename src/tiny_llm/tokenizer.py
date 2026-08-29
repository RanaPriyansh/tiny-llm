"""Character-level tokenizer."""

from typing import List


class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, vocab: str):
        """Initialize tokenizer with a vocabulary string.
        
        Args:
            vocab: String containing all unique characters in vocabulary
        """
        self.vocab = vocab
        self.char_to_idx = {ch: i for i, ch in enumerate(vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        return "".join([self.idx_to_char[idx] for idx in ids])

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Create tokenizer from a text corpus.
        
        Args:
            text: Text corpus to extract vocabulary from
            
        Returns:
            CharTokenizer instance
        """
        vocab = "".join(sorted(set(text)))
        return cls(vocab)
