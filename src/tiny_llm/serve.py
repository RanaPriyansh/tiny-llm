"""HTTP server for serving trained transformer model."""

import argparse
import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
from .tokenizer import CharTokenizer
from .model import TinyTransformer


class ModelServer:
    """Server for hosting trained model."""

    def __init__(self, checkpoint_path: str):
        """Initialize model server.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        self.checkpoint_path = checkpoint_path
        self.model: Optional[TinyTransformer] = None
        self.tokenizer: Optional[CharTokenizer] = None
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load model checkpoint."""
        checkpoint = np.load(self.checkpoint_path, allow_pickle=True)
        
        config = checkpoint["model_config"].item()
        vocab = str(checkpoint["vocab"])
        
        self.tokenizer = CharTokenizer(vocab)
        
        self.model = TinyTransformer(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_ff=config["d_ff"],
            max_seq_len=config["max_seq_len"],
        )
        
        self.model.token_emb = checkpoint["token_emb"]
        self.model.pos_emb = checkpoint["pos_emb"]
        self.model.ln_f_g = checkpoint["ln_f_g"]
        self.model.ln_f_b = checkpoint["ln_f_b"]
        self.model.lm_head = checkpoint["lm_head"]
        
        for i, block in enumerate(self.model.blocks):
            block.attn.W_q = checkpoint[f"block{i}_attn_Wq"]
            block.attn.W_k = checkpoint[f"block{i}_attn_Wk"]
            block.attn.W_v = checkpoint[f"block{i}_attn_Wv"]
            block.attn.W_o = checkpoint[f"block{i}_attn_Wo"]
            block.ff.W1 = checkpoint[f"block{i}_ff_W1"]
            block.ff.b1 = checkpoint[f"block{i}_ff_b1"]
            block.ff.W2 = checkpoint[f"block{i}_ff_W2"]
            block.ff.b2 = checkpoint[f"block{i}_ff_b2"]
            block.ln1_g = checkpoint[f"block{i}_ln1_g"]
            block.ln1_b = checkpoint[f"block{i}_ln1_b"]
            block.ln2_g = checkpoint[f"block{i}_ln2_g"]
            block.ln2_b = checkpoint[f"block{i}_ln2_b"]

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.8) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if not prompt:
            return ""
        
        prompt_ids = self.tokenizer.encode(prompt)
        input_ids = np.array([prompt_ids])
        
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        
        return self.tokenizer.decode(generated_ids[0].tolist())


class RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler."""

    server_instance: Optional[ModelServer] = None

    def log_message(self, format, *args):
        """Override to suppress request logging."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/generate":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body)
                
                prompt = data.get("prompt", "")
                max_tokens = data.get("max_tokens", 50)
                temperature = data.get("temperature", 0.8)
                
                if not self.server_instance:
                    raise RuntimeError("Server not initialized")
                
                generated = self.server_instance.generate(prompt, max_tokens, temperature)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"text": generated}).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()


def main():
    """Main serve function."""
    parser = argparse.ArgumentParser(description="Serve tiny transformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    server_instance = ModelServer(str(checkpoint_path))
    
    RequestHandler.server_instance = server_instance
    
    server = HTTPServer((args.host, args.port), RequestHandler)
    
    print(f"Server listening on http://{args.host}:{args.port}")
    print("Endpoints:")
    print("  GET  /health    - Health check")
    print("  POST /generate  - Generate text")
    print("\nPress Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()
