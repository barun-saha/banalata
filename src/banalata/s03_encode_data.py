"""Step 3: Encode Corpus to Token Arrays
======================================
Tokenizes train/val/test splits and saves them as memory-mapped numpy arrays.

Why memmap?
  - A 35MB corpus encodes to ~8-12M tokens (at ~3 chars/token for Bengali BPE)
  - Storing as np.uint16 costs ~16-24MB RAM — trivial
  - memmap means PyTorch DataLoader can read random windows WITHOUT loading
    the whole array into RAM — important if the corpus is later scaled up

Usage:
    python s03_encode_data.py
"""

import json
from pathlib import Path

import numpy as np
import sentencepiece as spm

# ---------------------------------------------------------------------------
DATA_DIR = Path('../../data')
TOK_DIR = Path('tokenizer')


def encode_split(sp: spm.SentencePieceProcessor, split: str, config: dict) -> int:
    """Encode one split (train/val/test) to a flat uint16 memmap array.
    Returns the number of tokens written.
    """
    text_path = DATA_DIR / f'{split}.txt'
    if not text_path.exists():
        print(f'  {split}.txt not found, skipping.')
        return 0

    text = text_path.read_text(encoding='utf-8')
    print(f'  Encoding {split} ({len(text):,} chars)...', end=' ', flush=True)

    # Encode the full text in one shot — SentencePiece handles large inputs
    token_ids = sp.encode(text, out_type=int)
    n_tokens = len(token_ids)

    # Save as uint16 (vocab ≤ 5000, well within uint16 range of 65535)
    arr = np.array(token_ids, dtype=np.uint16)
    out_path = DATA_DIR / f'{split}.bin'
    np.save(out_path, arr)  # saves as .bin.npy — rename for clarity
    # Actually save without the .npy extension for cleaner naming:
    out_path = DATA_DIR / f'{split}_tokens.npy'
    np.save(str(out_path), arr)

    print(f'{n_tokens:,} tokens  →  {out_path}')
    return n_tokens


def main():
    """Main execution function for encoding the corpus into token arrays."""
    # Load config
    config_path = TOK_DIR / 'tokenizer_config.json'
    if not config_path.exists():
        raise FileNotFoundError(
            'tokenizer_config.json not found. Run s02_train_tokenizer.py first.'
        )
    config = json.loads(config_path.read_text(encoding='utf-8'))

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(config['model_path'])
    print(f'Loaded tokenizer: vocab_size={config["vocab_size"]}')

    # Encode all splits
    print('\nEncoding splits:')
    stats = {}
    for split in ('train', 'val', 'test'):
        n = encode_split(sp, split, config)
        stats[split] = n

    # Save encoding stats alongside tokenizer config
    config['encoding_stats'] = stats
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    # Summary
    print(f'\n{"=" * 55}')
    total = sum(stats.values())
    for split, n in stats.items():
        pct = 100 * n / total if total else 0
        print(f'  {split:5s}: {n:>9,} tokens  ({pct:.1f}%)')
    print(f'  {"total":5s}: {total:>9,} tokens')
    print('\nToken files: data/train_tokens.npy, val_tokens.npy, test_tokens.npy')
    print('\nNext step: python s04_train_model.py')


if __name__ == '__main__':
    main()
