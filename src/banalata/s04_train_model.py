"""Step 4: Banalata — Model Definition and Training
====================================================
Decoder-only transformer trained from scratch on Bengali literary text.

Architecture:
  - ~28M parameters (8 layers × 512 embed dim × 8 heads)
  - Causal self-attention using torch.nn.functional.scaled_dot_product_attention
    (PyTorch 2.0+: uses FlashAttention-style kernels automatically on CUDA)
  - RoPE positional encoding (better than learned positions for generalization)
  - RMSNorm instead of LayerNorm (faster, no learned bias)
  - Author-conditioning via special tokens at start of each work

Requirements:
    pip install torch==2.11.0 sentencepiece==0.2.1 numpy

Usage:
    python s04_train_model.py                   # train from scratch
    python s04_train_model.py --resume          # resume from latest checkpoint
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.banalata.s00_model import Banalata, ModelConfig, TrainConfig

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TokenDataset(Dataset):
    """Sliding-window dataset over a flat token array.
    Each item is (context, target) where target = context shifted by 1.
    """

    def __init__(self, token_path: str | Path, context_len: int):
        self.tokens = np.load(str(token_path)).astype(np.int64)
        self.context_len = context_len
        # Number of full windows
        self.n = max(0, len(self.tokens) - context_len)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        chunk = self.tokens[idx : idx + self.context_len + 1]
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


# ---------------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------------


def get_lr(step: int, tcfg: TrainConfig, warmup_iters: int, max_iters: int) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < warmup_iters:
        return tcfg.learning_rate * max(step, 1) / warmup_iters
    if step >= max_iters:
        return tcfg.min_lr
    progress = (step - warmup_iters) / max(max_iters - warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return tcfg.min_lr + coeff * (tcfg.learning_rate - tcfg.min_lr)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss(
    model: Banalata, loaders: dict, eval_iters: int, device: torch.device, ctx
) -> dict:
    """Estimate training and validation loss by averaging over multiple batches."""
    model.eval()
    out = {}
    for split, loader in loaders.items():
        losses = []
        it = iter(loader)
        for _ in range(min(eval_iters, len(loader))):
            try:
                x, y = next(it)
            except StopIteration:
                break
            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses)) if losses else float('nan')
    model.train()
    return out


# ---------------------------------------------------------------------------
# Sample helper
# ---------------------------------------------------------------------------


def sample_text(
    model: Banalata, sp, config: dict, device: torch.device, ctx, prompt_author: str | None = None
) -> str:
    """Generate a sample, optionally conditioned on an author."""
    import random

    author_tokens = config.get('author_tokens', [])
    bow_id = config.get('bow_id')
    eow_id = config.get('eow_id')
    poem_id = config.get('poem_id')
    prose_id = config.get('prose_id')

    # Build prompt: <|bow|><|author:X|>
    if prompt_author and author_tokens:
        aut_tok = random.choice(author_tokens)
        aut_id = sp.piece_to_id(aut_tok)
        # Pick a random type token if both are available
        type_candidates = [t for t in [poem_id, prose_id] if t is not None]
        type_id = random.choice(type_candidates) if type_candidates else None
        prompt_ids = [x for x in [bow_id, aut_id, type_id] if x is not None]
    else:
        prompt_ids = [bow_id] if bow_id else []

    if not prompt_ids:
        # Fallback: random token from vocab
        prompt_ids = [1]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with ctx:
        out = model.generate(idx, max_new_tokens=150, temperature=0.85, top_p=0.92, eow_id=eow_id)
    tokens = out[0].tolist()
    # Decode, skipping special token IDs
    special_ids = set(config.get('special_tokens', {}).values())
    content_ids = [t for t in tokens if t not in special_ids]
    return sp.decode(content_ids)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def loss_by_author(
    model: Banalata,
    sp,
    tok_config: dict,
    data_dir: Path,
    device: torch.device,
    ctx,
    context_len: int = 512,
) -> dict:
    """Compute per-author val loss from val_works.json.
    Requires s01 to have saved data/val_works.json with fields:
      'author', 'type', 'formatted_text'
    Returns dict mapping author -> {'mean': float, 'std': float, 'n': int}
    """
    val_works_path = data_dir / 'val_works.json'
    if not val_works_path.exists():
        print(f'  [loss_by_author] {val_works_path} not found — skipping.')
        print('  Make sure s01 saves val_works.json (see note below).')
        return {}

    val_works = json.loads(val_works_path.read_text(encoding='utf-8'))
    special_ids = set(tok_config.get('special_tokens', {}).values())

    results: dict[str, list[float]] = {}
    model.eval()

    for work in val_works:
        author = work.get('author', '<unknown>')
        text = work.get('formatted_text', '')
        if not text.strip():
            continue

        ids = sp.encode(text)
        # Skip works too short to produce a meaningful loss estimate
        if len(ids) < 16:
            continue

        # Crop to context window — same as training
        ids = ids[: context_len + 1]
        if len(ids) < 2:
            continue

        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)

        with ctx:
            _, loss = model(x, y)

        if author not in results:
            results[author] = []
        results[author].append(loss.item())

    model.train()

    # Summarise
    summary = {
        author: {
            'mean': float(np.mean(losses)),
            'std': float(np.std(losses)),
            'n': len(losses),
        }
        for author, losses in results.items()
    }
    return summary


def print_author_losses(summary: dict):
    """Print a formatted table of per-author validation losses."""
    if not summary:
        return
    print(f'\n{"─" * 65}')
    print('  PER-AUTHOR VAL LOSS')
    print(f'  {"Author":<32} {"Works":>5}  {"Avg Loss":>8}  {"Std":>6}')
    print(f'  {"─" * 32}  {"─" * 5}  {"─" * 8}  {"─" * 6}')
    for author, s in sorted(summary.items(), key=lambda x: -x[1]['mean']):
        flag = '  ← HIGH' if s['mean'] > 4.2 else ''
        print(f'  {author:<32} {s["n"]:>5}  {s["mean"]:>8.4f}  {s["std"]:>6.4f}{flag}')
    print(f'{"─" * 65}\n')


def train(
    resume: bool = False,
    epochs: float | None = None,
    batch_size: int | None = None,
    weight_decay: float | None = None,
    dropout: float | None = None,
    context_len: int | None = None,
    lr: float | None = None,
    n_layer: int | None = None,
    n_head: int | None = None,
    n_embd: int | None = None,
    smoke_frac: float = 1.0,
    smoke: bool = False,
):
    """The main training loop for the Banalata model."""
    tcfg = TrainConfig()

    # Apply CLI overrides to tcfg BEFORE any derived computation
    if epochs is not None:
        tcfg.epochs = epochs
    if batch_size is not None:
        tcfg.batch_size = batch_size
    if weight_decay is not None:
        tcfg.weight_decay = weight_decay
    if context_len is not None:
        tcfg.context_len = context_len
    if lr is not None:
        tcfg.learning_rate = lr

    if dropout is None:
        dropout = ModelConfig.dropout

    # Smoke preset: 2 epochs on 1% of data — fast full-pipeline check
    if smoke:
        tcfg.epochs = 2.0
        tcfg.eval_every_epochs = 1.0
        tcfg.save_every_epochs = 2.0
        tcfg.sample_every_epochs = 2.0
        tcfg.patience = 3
        smoke_frac = min(smoke_frac, 0.01)

    Path(tcfg.ckpt_dir).mkdir(exist_ok=True)

    # ---- Device & dtype setup -----------------------------------------------
    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Device: {device}')

    if tcfg.dtype == 'auto':
        if device.type == 'cuda':
            # bfloat16 on Ampere+ (compute capability 8.0+), else float16
            cc = torch.cuda.get_device_capability()
            dtype_str = 'bfloat16' if cc[0] >= 8 else 'float16'
        else:
            dtype_str = 'float32'
    else:
        dtype_str = tcfg.dtype

    pt_dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[
        dtype_str
    ]
    print(f'Training dtype: {dtype_str}')

    # autocast context
    # On CPU: autocast only supports bfloat16/float16, but CPU training is
    # always float32 anyway — so we use nullcontext (a genuine no-op) instead
    # of torch.amp.autocast, which would emit a UserWarning and do nothing.
    from contextlib import nullcontext

    if device.type == 'cuda':
        ctx = torch.amp.autocast(device_type='cuda', dtype=pt_dtype)
    elif device.type == 'mps':
        ctx = torch.amp.autocast(device_type='mps', dtype=torch.bfloat16)
    else:
        ctx = nullcontext()  # CPU: no autocast, no warning

    # ---- Load tokenizer config -----------------------------------------------
    config_path = Path(tcfg.tok_dir) / 'tokenizer_config.json'
    if not config_path.exists():
        raise FileNotFoundError('tokenizer_config.json not found. Run steps 1-3 first.')
    tok_config = json.loads(config_path.read_text(encoding='utf-8'))

    import sentencepiece as spm_module

    sp = spm_module.SentencePieceProcessor()
    sp.load(tok_config['model_path'])

    # ---- Model config --------------------------------------------------------
    mcfg = ModelConfig(
        vocab_size=tok_config['vocab_size'],
        context_len=tcfg.context_len,  # from CLI / TrainConfig, not hardcoded
        n_layer=(n_layer if n_layer is not None else 8),  # overrideable via CLI
        n_head=(n_head if n_head is not None else 8),
        n_embd=(n_embd if n_embd is not None else 512),
        dropout=dropout,
    )

    # Basic sanity check: embedding dim must be divisible by num heads
    if mcfg.n_embd % mcfg.n_head != 0:
        raise ValueError(f'n_embd ({mcfg.n_embd}) must be divisible by n_head ({mcfg.n_head})')

    # ---- Data ----------------------------------------------------------------
    data_dir = Path(tcfg.data_dir)
    train_ds = TokenDataset(data_dir / 'train_tokens.npy', mcfg.context_len)
    val_ds = TokenDataset(data_dir / 'val_tokens.npy', mcfg.context_len)

    # Smoke-test: slice to a fraction without touching disk files
    if smoke_frac < 1.0:
        n_train = max(tcfg.batch_size * 8, int(len(train_ds.tokens) * smoke_frac))
        n_val = max(tcfg.batch_size * 4, int(len(val_ds.tokens) * smoke_frac))
        train_ds.tokens = train_ds.tokens[:n_train]
        val_ds.tokens = val_ds.tokens[:n_val]
        train_ds.n = max(0, len(train_ds.tokens) - mcfg.context_len)
        val_ds.n = max(0, len(val_ds.tokens) - mcfg.context_len)
        print(
            f'[SMOKE] {smoke_frac * 100:.0f}% of data: '
            f'train={len(train_ds.tokens):,} val={len(val_ds.tokens):,} tokens'
        )

    # ---- Derive all iteration counts from epochs HERE -----------------------
    # iters_per_epoch = how many full batches fit in the training set
    iters_per_epoch = max(1, len(train_ds.tokens) // (mcfg.context_len * tcfg.batch_size))
    max_iters = max(1, round(tcfg.epochs * iters_per_epoch))
    warmup_iters = max(1, round(tcfg.warmup_epochs * iters_per_epoch))
    eval_interval = max(1, round(tcfg.eval_every_epochs * iters_per_epoch))
    save_interval = max(1, round(tcfg.save_every_epochs * iters_per_epoch))
    sample_interval = max(1, round(tcfg.sample_every_epochs * iters_per_epoch))

    print(f'\nTrain tokens : {len(train_ds.tokens):,}  |  Val tokens: {len(val_ds.tokens):,}')
    print(f'Iters/epoch  : {iters_per_epoch:,}')
    print(f'Total iters  : {max_iters:,}  ({tcfg.epochs:.1f} epochs × {iters_per_epoch:,})')
    print(f'Warmup iters : {warmup_iters:,}  ({tcfg.warmup_epochs:.1f} epochs)')
    print(f'Eval every   : {eval_interval:,} iters  ({tcfg.eval_every_epochs:.1f} epochs)')

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    loaders = {'train': train_loader, 'val': val_loader}

    # ---- Model ---------------------------------------------------------------
    model = Banalata(mcfg).to(device)

    # Compile for speed on PyTorch 2.x (optional — skip if causing issues)
    if device.type == 'cuda' and hasattr(torch, 'compile'):
        print('Compiling model with torch.compile()...')
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f'  torch.compile() failed: {e}. Continuing without.')

    # ---- Optimizer -----------------------------------------------------------
    # Separate weight decay from bias / norm params
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': tcfg.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    # Use fused AdamW on CUDA (PyTorch 2.x)
    fused = device.type == 'cuda' and 'fused' in str(torch.optim.AdamW.__init__.__doc__ or '')
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=tcfg.learning_rate,
        betas=(0.9, 0.95),
        fused=fused if fused else False,
    )

    # GradScaler for float16 (not needed for bfloat16)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype_str == 'float16'))

    # ---- Resume from checkpoint ---------------------------------------------
    start_iter = 0
    best_val = float('inf')
    patience_count = 0

    if resume:
        ckpt_files = sorted(Path(tcfg.ckpt_dir).glob('ckpt_iter*.pt'))
        if ckpt_files:
            latest = ckpt_files[-1]
            print(f'Resuming from {latest}')
            ckpt = torch.load(latest, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_iter = ckpt['iter']
            best_val = ckpt.get('best_val', float('inf'))
            print(f'  Resumed at iter {start_iter}, best_val={best_val:.4f}')

    # ---- WandB (optional) ----------------------------------------------------
    # Activated only if wandb is installed AND WANDB_API_KEY is set in env.
    # Never crashes training if wandb is absent or key is missing.
    wandb_run = None
    try:
        import os

        import wandb

        wandb_key = os.environ.get('WANDB_API_KEY', '').strip()
        if wandb_key:
            wandb.login(key=wandb_key, relogin=False)
            wandb_run = wandb.init(
                project='banalata',
                config={
                    **mcfg.__dict__,
                    **tcfg.__dict__,
                    'smoke_frac': smoke_frac,
                    'device': str(device),
                    'dtype': dtype_str,
                    'iters_per_epoch': iters_per_epoch,
                    'max_iters': max_iters,
                    'warmup_iters': warmup_iters,
                },
                resume='allow',
            )
            print(f'WandB run: {wandb_run.url}')
        else:
            print('WandB: WANDB_API_KEY not set — logging to console only.')
    except ImportError:
        print('WandB: not installed — logging to console only. (pip install wandb to enable)')

    # ---- Training loop -------------------------------------------------------
    model.train()
    train_iter = iter(train_loader)
    t0 = time.time()
    iter_start = start_iter

    LOG_WINDOW = 20
    loss_window: list[float] = []
    iter_times: list[float] = [time.time()]

    def _fmt_time(seconds: float) -> str:
        s = int(seconds)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h:
            return f'{h}h {m:02d}m {sec:02d}s'
        if m:
            return f'{m}m {sec:02d}s'
        return f'{sec}s'

    def _eta(step: int) -> str:
        if len(iter_times) < 2:
            return 'estimating...'
        window = iter_times[-min(LOG_WINDOW, len(iter_times)) :]
        elapsed_w = window[-1] - window[0]
        iters_w = len(window) - 1
        if iters_w <= 0:
            return 'estimating...'
        secs_per_iter = elapsed_w / iters_w
        return _fmt_time(secs_per_iter * (max_iters - step - 1))

    def _elapsed() -> str:
        return _fmt_time(time.time() - t0)

    print(f'\n{"=" * 60}')
    print('Starting training')
    print(
        f'  epochs     : {tcfg.epochs:.1f}  ({max_iters:,} iters × '
        f'{tcfg.batch_size} batch × {mcfg.context_len} ctx)'
    )
    print(f'  batch size : {tcfg.batch_size}  |  context : {mcfg.context_len}')
    print(
        f'  tokens/iter: {tcfg.batch_size * mcfg.context_len:,}  |  '
        f'tokens/epoch: {tcfg.batch_size * mcfg.context_len * iters_per_epoch:,}'
    )
    print(
        f'  lr         : {tcfg.learning_rate:.1e} → {tcfg.min_lr:.1e}  '
        f'(warmup {tcfg.warmup_epochs:.1f} ep = {warmup_iters} iters)'
    )
    print(f'{"=" * 60}\n')

    for step in range(start_iter, max_iters):
        iter_t0 = time.time()

        # --- LR update — uses local warmup_iters and max_iters
        lr = get_lr(step, tcfg, warmup_iters, max_iters)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # --- Get batch (cycle through dataset)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # --- Forward + backward
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Track iteration timing and loss
        iter_times.append(time.time())
        if len(iter_times) > LOG_WINDOW + 1:
            iter_times.pop(0)
        raw_loss = loss.item()
        loss_window.append(raw_loss)
        if len(loss_window) > LOG_WINDOW:
            loss_window.pop(0)
        smooth_loss = sum(loss_window) / len(loss_window)

        # ms per iteration (last iter only — for the per-step display)
        ms_per_iter = (iter_times[-1] - iter_times[-2]) * 1000 if len(iter_times) >= 2 else 0.0

        # --- Per-step console log
        log_every = 1 if max_iters <= 50 else 10
        if step % log_every == 0:
            current_epoch = step / iters_per_epoch
            print(
                f'iter {step:5d}/{max_iters} '
                f'(ep {current_epoch:5.2f}/{tcfg.epochs:.1f}) | '
                f'loss {raw_loss:.4f} (smooth {smooth_loss:.4f}) | '
                f'lr {lr:.2e} | '
                f'{ms_per_iter:5.0f}ms/it | '
                f'elapsed {_elapsed()} | '
                f'eta {_eta(step)}'
            )

        # --- WandB per-step
        if wandb_run is not None:
            wandb_run.log(
                {
                    'train/loss_raw': raw_loss,
                    'train/loss_smooth': smooth_loss,
                    'train/lr': lr,
                    'perf/ms_per_iter': ms_per_iter,
                },
                step=step,
            )

        # --- Eval
        if step % eval_interval == 0 and step > 0:
            eval_t0 = time.time()
            losses = estimate_loss(model, loaders, tcfg.eval_iters, device, ctx)
            eval_secs = time.time() - eval_t0
            current_epoch = step / iters_per_epoch

            print(
                f'\n{"─" * 60}\n'
                f'  EVAL  iter {step:5d}/{max_iters}  '
                f'(epoch {current_epoch:.2f}/{tcfg.epochs:.1f})\n'
                f'  train loss : {losses["train"]:.4f}\n'
                f'  val   loss : {losses["val"]:.4f}   '
                f'(best so far: {best_val:.4f})\n'
                f'  elapsed    : {_elapsed()}   |   '
                f'eta: {_eta(step)}   |   '
                f'eval took: {eval_secs:.1f}s'
            )

            if losses['val'] < best_val:
                best_val = losses['val']
                patience_count = 0
                ckpt = {
                    'iter': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val': best_val,
                    'mcfg': mcfg.__dict__,
                    'tcfg': tcfg.__dict__,
                }
                torch.save(ckpt, Path(tcfg.ckpt_dir) / 'ckpt_best.pt')
                print('  ✓ New best — checkpoint saved')
            else:
                patience_count += 1
                print(f'  No improvement  ({patience_count}/{tcfg.patience} patience)')
                if patience_count >= tcfg.patience:
                    print(f'\n{"─" * 60}')
                    print(
                        f'Early stopping at iter {step}. '
                        f'No val improvement for {tcfg.patience} evals.'
                    )
                    break

            print(f'{"─" * 60}\n')

            if wandb_run is not None:
                wandb_run.log(
                    {
                        'eval/train_loss': losses['train'],
                        'eval/val_loss': losses['val'],
                        'eval/best_val': best_val,
                        'epoch': current_epoch,
                    },
                    step=step,
                )

            model.train()

        # --- Periodic checkpoint
        if step % save_interval == 0 and step > 0:
            ckpt = {
                'iter': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val': best_val,
                'mcfg': mcfg.__dict__,
                'tcfg': tcfg.__dict__,
            }
            torch.save(ckpt, Path(tcfg.ckpt_dir) / f'ckpt_iter{step:05d}.pt')
            print(f'  [checkpoint saved: iter {step}]')

        # --- Sample text
        if step % sample_interval == 0 and step > 0:
            print(f'\n{"─" * 60}')
            print(f'  SAMPLE  iter {step}  (temp=0.85, top_p=0.92)')
            print(f'{"─" * 60}')
            text = sample_text(model, sp, tok_config, device, ctx)
            print(text[:500])
            print(f'{"─" * 60}\n')
            if wandb_run is not None:
                wandb_run.log({'samples/text': wandb.Html(f'<pre>{text[:1000]}</pre>')}, step=step)
            model.train()

    # ---- Final eval (always runs — catches the last epoch regardless of
    #      whether the loop ended naturally or via early stopping) -----------
    print(f'\n{"─" * 60}')
    print('  FINAL EVAL')
    eval_t0 = time.time()
    losses = estimate_loss(model, loaders, tcfg.eval_iters, device, ctx)
    eval_secs = time.time() - eval_t0
    final_epoch = step / iters_per_epoch  # type: ignore[possibly-undefined]
    print(
        f'  train loss : {losses["train"]:.4f}\n'
        f'  val   loss : {losses["val"]:.4f}   (best so far: {best_val:.4f})\n'
        f'  total time : {_elapsed()}   |   eval took: {eval_secs:.1f}s'
    )
    if losses['val'] < best_val:
        best_val = losses['val']
        ckpt = {
            'iter': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val': best_val,
            'mcfg': mcfg.__dict__,
            'tcfg': tcfg.__dict__,
        }
        torch.save(ckpt, Path(tcfg.ckpt_dir) / 'ckpt_best.pt')
        print('  ✓ New best at final eval — checkpoint saved')
    print(f'{"─" * 60}')

    # ---- Per-author val loss diagnostic  ← ADD THIS BLOCK
    print('\nRunning per-author val loss diagnostic...')
    author_summary = loss_by_author(
        model,
        sp,
        tok_config,
        data_dir=Path(tcfg.data_dir),
        device=device,
        ctx=ctx,
        context_len=mcfg.context_len,
    )
    print_author_losses(author_summary)

    if wandb_run is not None:
        wandb_run.log(
            {
                'eval/train_loss': losses['train'],
                'eval/val_loss': losses['val'],
                'eval/best_val': best_val,
                'epoch': tcfg.epochs,  # log as full epoch count at end
            },
            step=step,
        )

    total_time = _fmt_time(time.time() - t0)
    print(f'\n{"=" * 60}')
    print('Training complete.')
    print(f'  Best val loss : {best_val:.4f}')
    print(f'  Total time    : {total_time}')
    print(f'  Best ckpt     : {tcfg.ckpt_dir}/ckpt_best.pt')
    print(f'{"=" * 60}')

    if wandb_run is not None:
        wandb_run.summary['best_val_loss'] = best_val
        wandb_run.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Banalata from scratch',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core training knobs — all epoch-based
    parser.add_argument(
        '--epochs',
        type=float,
        default=None,
        help='Total training epochs. '
        'Iterations are derived automatically from '
        'dataset size, batch size, and context length. '
        f'Default: {TrainConfig.epochs}',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size. Increase until GPU memory is ~70-80%%. '
        'Doubling batch size halves iters/epoch but keeps '
        'total token exposure the same. '
        f'Default: {TrainConfig.batch_size}',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=None,
        help=f'Model dropout.Default: {ModelConfig.dropout}',
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=None,
        help=f'AdamW weight decay. Default: {TrainConfig.weight_decay}',
    )
    parser.add_argument(
        '--context-len',
        type=int,
        default=None,
        help='Token context window length. Must be ≤ the value '
        'used when the tokenizer was trained. '
        f'Default: {TrainConfig.context_len}',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help=f'Peak learning rate. Default: {TrainConfig.learning_rate}',
    )

    parser.add_argument(
        '--n-layer',
        type=int,
        default=6,
        help='Number of transformer blocks (n_layer). If provided overrides the default in the script.',
    )
    parser.add_argument(
        '--n-head',
        type=int,
        default=6,
        help='Number of attention heads (n_head). If provided overrides the default in the script.',
    )
    parser.add_argument(
        '--n-embd',
        type=int,
        default=384,
        help='Embedding dimension (n_embd). If provided overrides the default in the script.',
    )

    # Workflow
    parser.add_argument(
        '--resume', action='store_true', help='Resume from latest checkpoint in checkpoints/'
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Quick 2-epoch pipeline check on 1%% of data. '
        'Verifies the full code path on CPU in ~1 min.',
    )
    parser.add_argument(
        '--smoke-frac',
        type=float,
        default=1.0,
        help='Use this fraction of token data (0.0–1.0). '
        'Useful with --smoke or for faster iteration.',
    )

    args = parser.parse_args()

    train(
        resume=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        context_len=args.context_len,
        lr=args.lr,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        smoke_frac=args.smoke_frac,
        smoke=args.smoke,
    )
