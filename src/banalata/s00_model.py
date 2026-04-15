"""Banalata model architecture and training configurations."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for the Banalata transformer model."""

    vocab_size: int = 5000  # set from tokenizer at runtime
    context_len: int = 512  # max sequence length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.15  # higher than typical — small data needs it
    bias: bool = False  # no bias in linear layers (GPT-2 style)


@dataclass
class TrainConfig:
    """Configuration for the model training process."""

    # Paths
    data_dir: str = '../../data'
    tok_dir: str = 'tokenizer'
    ckpt_dir: str = 'checkpoints'

    # Training dimensions — these are the knobs you actually want to tune.
    # max_iters is DERIVED from epochs after data is loaded; never set it directly.
    epochs: float = 20.0  # total training epochs (float → fractional ok)
    batch_size: int = 32
    context_len: int = 512  # token context window (must match model)
    grad_clip: float = 1.0
    weight_decay: float = 0.1

    # Learning rate schedule: linear warmup then cosine decay.
    # warmup_epochs and lr_decay are also epoch-relative so they scale
    # automatically when you change batch size or dataset size.
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_epochs: float = 1.0  # how many epochs to warm up over

    # Evaluation & checkpointing — expressed in epochs for the same reason.
    eval_every_epochs: float = 1.0  # run validation every N epochs
    save_every_epochs: float = 5.0  # save periodic checkpoint every N epochs
    sample_every_epochs: float = 5.0  # print a generation sample every N epochs
    eval_iters: int = 60  # number of val batches to average over
    patience: int = 5  # early stop after N evals with no improvement

    # Mixed precision — bfloat16 on Ampere+ GPUs (A100/H100), float16 on T4/V100
    dtype: str = 'auto'  # "auto" | "bfloat16" | "float16" | "float32"

    # Sample generation settings
    sample_max_tokens: int = 200


# ---------------------------------------------------------------------------
# Rotary Positional Encoding (RoPE)
# ---------------------------------------------------------------------------


def precompute_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute complex rotation frequencies for RoPE.
    Returns tensor of shape (max_seq_len, head_dim // 2) as complex64.
    """
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, theta)  # (seq, head_dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to query or key tensor.
    x:     (batch, n_heads, seq_len, head_dim)
    freqs: (seq_len, head_dim // 2)  complex
    """
    # Reshape x to complex pairs
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[: x.shape[2]]  # trim to actual seq len
    # Broadcast over batch and heads
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim//2)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).reshape(x.shape).to(x.dtype)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.scale


# ---------------------------------------------------------------------------
# Causal Self-Attention (using F.scaled_dot_product_attention — PyTorch 2.0+)
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Causal self-attention layer with RoPE and FlashAttention."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, 'n_embd must be divisible by n_head'

        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

        # Fused QKV projection — one matrix instead of three saves memory
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (not V)
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # F.scaled_dot_product_attention handles causal masking automatically
        # with is_causal=True. On CUDA with PyTorch 2.0+, this dispatches to
        # FlashAttention-2 when available — no manual mask needed.
        attn_dropout = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=attn_dropout,
            is_causal=True,
        )

        # Reassemble: (B, n_head, T, head_dim) → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


# ---------------------------------------------------------------------------
# Feed-Forward Block (SwiGLU activation — better than GELU for small models)
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(4 * cfg.n_embd * 2 / 3)  # SwiGLU hidden dim formula
        hidden = (hidden + 7) // 8 * 8  # round up to multiple of 8

        # SwiGLU: gate ⊗ SiLU(linear)
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w3 = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class Block(nn.Module):
    """Transformer block with attention and feed-forward layers."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_freqs)  # pre-norm residual
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Banalata Model
# ---------------------------------------------------------------------------


class Banalata(nn.Module):
    """The main Banalata transformer model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(
            dict(
                tok_emb=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                drop=nn.Dropout(cfg.dropout),
                blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                norm_f=RMSNorm(cfg.n_embd),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: lm_head shares weights with token embedding.
        # Widely used since GPT-2 — reduces params and improves perplexity.
        self.transformer.tok_emb.weight = self.lm_head.weight

        # Precompute RoPE frequencies (stored as buffer, not parameter)
        rope_freqs = precompute_rope_freqs(cfg.n_embd // cfg.n_head, cfg.context_len)
        self.register_buffer('rope_freqs', rope_freqs)

        # Init weights
        self.apply(self._init_weights)
        # GPT-2 paper: scale residual projections by 1/sqrt(n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith(('c_proj.weight', 'w3.weight')):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f'Banalata: {n_params / 1e6:.2f}M parameters')

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx:     (B, T) int64 token indices
        targets: (B, T) int64 shifted targets (for training)
        Returns: (logits, loss) where loss is None if targets not provided
        """
        B, T = idx.shape
        assert T <= self.cfg.context_len, (
            f'Sequence length {T} exceeds context_len {self.cfg.context_len}'
        )

        x = self.transformer.drop(self.transformer.tok_emb(idx))  # (B, T, C)

        for block in self.transformer.blocks:
            x = block(x, self.rope_freqs)

        x = self.transformer.norm_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is not None:
            # Standard cross-entropy next-token prediction
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss

        return logits, None

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.85,
        top_p: float = 0.92,
        eow_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with nucleus (top-p) sampling.
        Stops early if <|eow|> token is generated.
        """
        for _ in range(max_new_tokens):
            # Crop context to context_len if necessary
            idx_cond = idx[:, -self.cfg.context_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-p (nucleus) sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens with cumulative prob above top_p threshold
            sorted_probs[cumulative - sorted_probs > top_p] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)  # renorm
            next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))

            idx = torch.cat([idx, next_token], dim=1)

            if eow_id is not None and next_token.item() == eow_id:
                break

        return idx
