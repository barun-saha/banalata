r"""Step 5: Generate Bengali Text
==============================
Load a trained checkpoint and generate text, optionally conditioning
on a specific author or completing a given prompt.

Usage:
    # Generate with a random author
    python s05_generate.py

    # Generate conditioned on a specific author
    python s05_generate.py --author "রবীন্দ্রনাথ ঠাকুর" --type poem

    # Complete a given Bengali prompt
    python s05_generate.py --prompt $'<|bow|><|author:জীবনানন্দ দাশ|><|poem|>\nহাজার বছর ধরে'

    # Interactive mode
    python s05_generate.py --interactive

    # Show all available authors
    python s05_generate.py --list-authors
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

from src.banalata.s00_model import Banalata, ModelConfig

# Model and tokenizer configs
MODULE_PATH = Path(__file__).resolve().parent
CKPT_PATH = MODULE_PATH / 'checkpoints/ckpt_best.pt'
TOK_DIR = MODULE_PATH / 'tokenizer'

# Defaults used when --prompt is given without --author / --type
DEFAULT_AUTHOR = 'জীবনানন্দ দাশ'
DEFAULT_TYPE = 'poem'


def load_model_and_tokenizer(ckpt_path: str, device: torch.device):
    """Load checkpoint, reconstruct model, load tokenizer."""
    tok_config = json.loads((TOK_DIR / 'tokenizer_config.json').read_text(encoding='utf-8'))
    sp = spm.SentencePieceProcessor()
    sp.load(str(MODULE_PATH / tok_config['model_path']))

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    mcfg_dict = ckpt['mcfg']
    mcfg = ModelConfig(**mcfg_dict)

    model = Banalata(mcfg).to(device)
    state = ckpt['model']
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    print(
        f'Loaded checkpoint (iter={ckpt.get("iter", "?")}, '
        f'val_loss={ckpt.get("best_val", "?"):.4f})'
    )
    return model, sp, tok_config


@torch.inference_mode()
def generate(
    model: Banalata,
    sp,
    tok_config: dict,
    device: torch.device,
    author: str | None = None,
    content_type: str | None = None,
    prompt: str | None = None,
    max_tokens: int = 300,
    temperature: float = 0.85,
    top_p: float = 0.92,
    repetition_penalty: float = 1.0,
    n_samples: int = 1,
) -> list[str]:
    """Generate text samples.

    Prompt construction priority:
      1. If `prompt` given: encode it directly (author/type ignored — embed them in the prompt string)
      2. If `author` given: <|bow|><|author:NAME|>[<|poem|> or <|prose|>]
      3. Otherwise: <|bow|> only

    repetition_penalty: divides logits of already-seen tokens before sampling.
      1.0  = no penalty (original behaviour)
      1.2  = light penalty, reduces mild loops
      1.3  = recommended default, handles most repetition
      1.5  = aggressive, may hurt coherence for highly repetitive styles (e.g. Lalan)
    """
    bow_id = tok_config.get('bow_id')
    eow_id = tok_config.get('eow_id')
    special_ids = set(tok_config.get('special_tokens', {}).values())

    results = []

    if device.type == 'cuda':
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    elif device.type == 'mps':
        ctx = torch.amp.autocast(device_type='mps', dtype=torch.bfloat16)
    else:
        ctx = nullcontext()

    for _ in range(n_samples):
        if prompt:
            # --prompt accepts plain Bengali text only.
            # Author and type conditioning come from --author / --type args,
            # falling back to DEFAULT_AUTHOR / DEFAULT_TYPE if not set.
            effective_author = content_type and author  # use explicit if both given
            eff_author = author or DEFAULT_AUTHOR
            eff_type = content_type or DEFAULT_TYPE

            aut_tok = f'<|author:{eff_author}|>'
            aut_id = sp.piece_to_id(aut_tok)
            if aut_id == sp.unk_id():
                available = tok_config.get('author_tokens', [])
                matches = [t for t in available if eff_author in t]
                aut_id = sp.piece_to_id(matches[0]) if matches else None
                if aut_id:
                    print(f'Using author token: {matches[0]}')
                else:
                    print(f"Author '{eff_author}' not found, omitting.")

            type_tok = f'<|{eff_type}|>'
            type_id = sp.piece_to_id(type_tok)
            if type_id == sp.unk_id():
                print(f"Type token '{type_tok}' not found, omitting.")
                type_id = None

            # Build: <|bow|><|author:NAME|><|poem|>\nplain text
            text_ids = sp.encode(prompt, out_type=int)
            prefix_ids = [x for x in [bow_id, aut_id, type_id] if x is not None]
            prompt_ids = prefix_ids + text_ids

        elif author:
            # Author + optional type conditioning
            aut_tok = f'<|author:{author}|>'
            aut_id = sp.piece_to_id(aut_tok)
            if aut_id == sp.unk_id():
                available = tok_config.get('author_tokens', [])
                matches = [t for t in available if author in t]
                if matches:
                    aut_tok = matches[0]
                    aut_id = sp.piece_to_id(aut_tok)
                    print(f'Using author token: {aut_tok}')
                else:
                    print(f"Author '{author}' not found. Using <|bow|> only.")
                    aut_id = None

            type_id = None
            if content_type:
                type_tok = f'<|{content_type}|>'
                type_id = sp.piece_to_id(type_tok)
                if type_id == sp.unk_id():
                    print(f"Type token '{type_tok}' not found, ignoring.")
                    type_id = None

            prompt_ids = [x for x in [bow_id, aut_id, type_id] if x is not None]
            if not prompt_ids:
                prompt_ids = [bow_id] if bow_id else []

        else:
            prompt_ids = [bow_id] if bow_id else []

        if not prompt_ids:
            prompt_ids = [sp.bos_id()]

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with ctx:
            out = _generate_tokens(
                model,
                idx,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                eow_id=eow_id,
                repetition_penalty=repetition_penalty,
            )

        tokens = out[0].tolist()
        content_ids = [t for t in tokens if t not in special_ids]
        text = sp.decode(content_ids)
        results.append(text.strip())

    return results


@torch.inference_mode()
def _generate_tokens(
    model: Banalata,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eow_id: int | None,
    repetition_penalty: float = 1.3,
) -> torch.Tensor:
    """Core autoregressive loop with repetition penalty and nucleus sampling.

    Repetition penalty (from the original "CTRL" paper):
      - For each token already in the sequence, divide its logit by the penalty.
      - Positive logits become smaller (less likely).
      - Negative logits become more negative (even less likely).
      - Applied BEFORE temperature scaling so temperature still controls overall sharpness.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.context_len :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (1, vocab_size)

        # Repetition penalty
        # Collect unique token ids seen so far in the full sequence
        if repetition_penalty != 1.0:
            seen = idx[0].unique()
            # Penalise: divide positive logits, multiply negative logits
            # This preserves sign while reducing magnitude in both directions
            logits[0, seen] = torch.where(
                logits[0, seen] > 0,
                logits[0, seen] / repetition_penalty,
                logits[0, seen] * repetition_penalty,
            )

        # Temperature
        logits = logits / temperature

        # Top-p (nucleus) sampling
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative - sorted_probs > top_p] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))

        idx = torch.cat([idx, next_token], dim=1)

        if eow_id is not None and next_token.item() == eow_id:
            break

    return idx


def list_authors(tok_config: dict):
    """List all available author names for conditioning."""
    tokens = tok_config.get('author_tokens', [])
    print(f'\nAvailable author tokens ({len(tokens)}):')
    for t in sorted(tokens):
        name = t.replace('<|author:', '').replace('|>', '')
        print(f'  --author "{name}"')


def interactive_mode(model, sp, tok_config, device):
    """Start an interactive REPL session for Bengali text generation."""
    print('\n' + '=' * 55)
    print('Banalata — Interactive Mode')
    print('Commands:')
    print('  [Enter] alone       — generate with random author')
    print('  author: NAME        — set author (Bengali name)')
    print('  type: poem|prose    — set content type')
    print('  prompt: TEXT        — set raw prompt (overrides author/type)')
    print('  temp: 0.8           — set temperature (default 0.85)')
    print('  top_p: 0.9          — set top-p (default 0.92)')
    print('  penalty: 1.3        — set repetition penalty (default 1.3)')
    print('  tokens: 200         — set max output tokens')
    print('  authors             — list available authors')
    print('  quit                — exit')
    print('=' * 55 + '\n')

    import random

    author = None
    content_type = None
    prompt = None
    temp = 0.85
    top_p = 0.92
    rep_penalty = 1.3
    max_tokens = 250

    while True:
        try:
            cmd = input('>>> ').strip()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd.lower() in ('quit', 'exit', 'q'):
            break
        elif cmd.lower() == 'authors':
            list_authors(tok_config)
        elif cmd.lower().startswith('author:'):
            author = cmd.split(':', 1)[1].strip()
            prompt = None
            print(f'Author set to: {author}')
        elif cmd.lower().startswith('type:'):
            content_type = cmd.split(':', 1)[1].strip().lower()
            if content_type not in ('poem', 'prose'):
                print("Type must be 'poem' or 'prose'")
                content_type = None
            else:
                print(f'Type set to: {content_type}')
        elif cmd.lower().startswith('prompt:'):
            prompt = cmd.split(':', 1)[1].strip()
            author = None
            content_type = None
            print(f'Prompt set to: {prompt}')
        elif cmd.lower().startswith('temp:'):
            temp = float(cmd.split(':', 1)[1].strip())
            print(f'Temperature: {temp}')
        elif cmd.lower().startswith('top_p:'):
            top_p = float(cmd.split(':', 1)[1].strip())
            print(f'Top-p: {top_p}')
        elif cmd.lower().startswith('penalty:'):
            rep_penalty = float(cmd.split(':', 1)[1].strip())
            print(f'Repetition penalty: {rep_penalty}')
        elif cmd.lower().startswith('tokens:'):
            max_tokens = int(cmd.split(':', 1)[1].strip())
            print(f'Max tokens: {max_tokens}')
        elif cmd == '':
            if author is None and prompt is None:
                tokens = tok_config.get('author_tokens', [])
                if tokens:
                    tok = random.choice(tokens)
                    author = tok.replace('<|author:', '').replace('|>', '')
                    print(f'(Random author: {author})')

            results = generate(
                model,
                sp,
                tok_config,
                device,
                author=author,
                content_type=content_type,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                repetition_penalty=rep_penalty,
            )
            print(f'\n{"-" * 50}')
            print(results[0])
            print(f'{"-" * 50}\n')


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------


def main():
    """Main execution function for text generation via command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Banalata Text Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python s05_generate.py --author "রবীন্দ্রনাথ ঠাকুর" --type poem
    python s05_generate.py --author "জীবনানন্দ দাশ" --type poem --penalty 1.2
    python s05_generate.py --prompt $'<|bow|><|author:জীবনানন্দ দাশ|><|poem|>\\nহাজার বছর ধরে'
    python s05_generate.py --interactive
        """,
    )
    parser.add_argument('--ckpt', default=CKPT_PATH)
    parser.add_argument(
        '--author', default=DEFAULT_AUTHOR, help="Bengali author name, e.g. 'রবীন্দ্রনাথ ঠাকুর'"
    )
    parser.add_argument(
        '--type',
        dest='content_type',
        choices=['poem', 'prose'],
        default=DEFAULT_TYPE,
        help='Content type token to prepend (only used with --author)',
    )
    parser.add_argument(
        '--prompt',
        default=None,
        help='Raw prompt string. Embed special tokens directly for full control: '
        "$'<|bow|><|author:NAME|><|poem|>\\nopening line'",
    )
    parser.add_argument('--max-tokens', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.85)
    parser.add_argument('--top-p', type=float, default=0.92)
    parser.add_argument(
        '--penalty',
        type=float,
        default=1.3,
        help='Repetition penalty. 1.0=disabled, 1.2=light, 1.3=default, 1.5=aggressive',
    )
    parser.add_argument('--n-samples', type=int, default=1)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--list-authors', action='store_true')
    args = parser.parse_args()

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )

    model, sp, tok_config = load_model_and_tokenizer(args.ckpt, device)

    if args.list_authors:
        list_authors(tok_config)
        return

    if args.interactive:
        interactive_mode(model, sp, tok_config, device)
        return

    results = generate(
        model,
        sp,
        tok_config,
        device,
        author=args.author,
        content_type=args.content_type,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.penalty,
        n_samples=args.n_samples,
    )

    for i, text in enumerate(results):
        if args.n_samples > 1:
            print(f'\n--- Sample {i + 1}')
        print(text)


if __name__ == '__main__':
    main()
