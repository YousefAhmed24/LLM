import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dotenv import load_dotenv

load_dotenv()

block_size = int(os.getenv("BLOCK_SIZE", 256))
n_embd = int(os.getenv("N_EMBD", 512))
n_head = int(os.getenv("N_HEAD", 6))
n_layer = int(os.getenv("N_LAYER", 6))
dropout = float(os.getenv("DROPOUT", 0.1))
vocab_size = int(os.getenv("VOCAB_SIZE", 50257))

class CausalSelfAttention(nn.Module):
    """Fused Multi-Head Attention using PyTorch FlashAttention kernels."""
    def __init__(self):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.head_size = n_embd // n_head
        
        # Fused Linear layer for Q, K, V projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        
        # Calculate Query, Key, Value for all heads in a single pass
        qkv = self.c_attn(x)
        q, k, v = qkv.split(n_embd, dim=2)
        
        # Reshape to (B, n_head, T, head_size)
        q = q.view(B, T, n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, n_head, self.head_size).transpose(1, 2)
        
        # PyTorch Native Causal Scaled Dot-Product Attention (FlashAttention)
        y = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            dropout_p=self.dropout if self.training else 0.0
        )
        
        # Re-assemble head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """Feed-Forward Network with 4x expansion."""
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gelu = nn.GELU() # Modern standard activation
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Standard Transformer Block (Pre-LayerNorm)."""
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        # Pre-LN residual stream
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Token and Positional Embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        
        # Transformer Blocks
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Output Linear Head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight Tying: Share token embeddings with output projection matrix
        self.token_embedding_table.weight = self.lm_head.weight

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= block_size, f"Sequence length {T} exceeds block size {block_size}"

        # Combine embeddings
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        # Forward through layers
        x = self.blocks(x)
        x = self.ln_f(x)
        
        if targets is not None:
            # Training: Calculate loss across entire vocabulary
            logits = self.lm_head(x) # (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            # Inference optimization: Only calculate logits for the very last token
            logits = self.lm_head(x[:, [-1], :]) # (B, 1, vocab_size)
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=50):
        """Autoregressive generation with top-k filtering and temperature scaling."""
        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = idx[:, -block_size:]
            
            # Forward pass (only computes logits for the last token)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx