"""Step 1: Data Preparation
========================
Downloads the bangla_sahitya dataset, cleans it, removes English/noise, and writes
a training corpus with author-conditioning tokens.

Requirements:
    pip install datasets==3.5.0 sentencepiece==0.2.1

Usage:
    python s01_prepare_data.py
"""

import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

# Configs
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent.parent / 'data'
OUTPUT_DIR.mkdir(exist_ok=True)

TOK_BOW = '<|bow|>'
TOK_EOW = '<|eow|>'
TOK_POEM = '<|poem|>'
TOK_PROSE = '<|prose|>'
TOK_AUT = '<|author:{name}|>'

MIN_WORK_CHARS = 150
MIN_AUTHOR_CHARS = 2000  # Authors with less text get pooled
RNG_SEED = 42

BENGALI_BLOCK = (0x0980, 0x09FF)
BENGALI_PUNCTUATION = set('।৷॥,।!?;:"\'()[]{}…—–-\n ')
LATIN_RE = re.compile(r'[A-Za-z]{3,}')


# ---------------------------------------------------------------------------
# Cleaning Utilities
# ---------------------------------------------------------------------------


def is_mostly_bengali(text: str, threshold: float = 0.5) -> bool:
    """Check if the text consists of mostly Bengali characters."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    bengali = sum(
        1
        for c in chars
        if BENGALI_BLOCK[0] <= ord(c) <= BENGALI_BLOCK[1] or c in BENGALI_PUNCTUATION
    )
    return (bengali / len(chars)) >= threshold


def remove_english_lines(text: str) -> str:
    """Filter out lines that are primarily in English/Latin script."""
    lines = text.split('\n')
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if LATIN_RE.search(stripped):
            latin_chars = sum(1 for c in stripped if c.isascii() and c.isalpha())
            total_alpha = sum(1 for c in stripped if c.isalpha())
            if total_alpha > 0 and (latin_chars / total_alpha) > 0.4:
                continue
        kept.append(line)
    return '\n'.join(kept)


def clean_text(text: str) -> str:
    """Clean and standardize input text, removing noise and normalizing whitespace."""
    text = unicodedata.normalize('NFC', text)
    EXOTIC_WHITESPACE = str.maketrans(
        {
            '\u00a0': ' ',
            '\u200b': '',
            '\u200c': '',
            '\u200d': '',
            '\u2002': ' ',
            '\u2003': ' ',
            '\u2004': ' ',
            '\u2005': ' ',
            '\u2006': ' ',
            '\u2007': ' ',
            '\u2008': ' ',
            '\u2009': ' ',
            '\u200a': ' ',
            '\u202f': ' ',
            '\u3000': ' ',
        }
    )
    text = text.translate(EXOTIC_WHITESPACE)
    text = remove_english_lines(text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cc' or c in '\n\t')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))

    # Standardize punctuation
    replacements = {
        '|': '।',
        '--': '—',
        '।।': '॥',
        '–': '—',
        ' ;': ';',
        ' ।': '।',
        ' ॥': '॥',
        '\u2018': "'",
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
        '…': '...',  # ellipsis character → three dots (more tokenizer-friendly)
        '„': '"',  # German-style opening quote sometimes appears in OCR
        '‟': '"',  # another quote variant
        '।\n।': '॥\n',  # doubled danda with newline between
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r'্\s+', '্', text)  # remove space after hasanta
    text = re.sub(r'্+', '্', text)  # collapse multiple hasanta
    # Collapse repeated dandas/punctuation from OCR artifacts
    text = re.sub(r'।{2,}', '॥', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)

    return text.strip()


def extract_content(raw: str) -> str:
    """Extract actual content from raw text, skipping metadata or headers."""
    lines = raw.split('\n')
    content_start = 0
    for i, line in enumerate(lines[:6]):
        stripped = line.strip()
        if not stripped or len(stripped) < 55:
            content_start = i + 1
    return '\n'.join(lines[content_start:]).strip()


def format_work(author_token: str, content: str, tag: str) -> str:
    """Format a work with its author token and type tag.

    Args:
        author_token: e.g. '<|author:রবীন্দ্রনাথ ঠাকুর|>'
        content:      cleaned text of the work
        tag:          'poem' or 'prose' (directly from the dataset 'tag' field)
    """
    type_tok = TOK_POEM if tag == 'poem' else TOK_PROSE
    return f'{TOK_BOW}{author_token}{type_tok}\n{content}\n{TOK_EOW}'


# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------


def main():
    """Main execution function for data preparation and splitting."""
    random.seed(RNG_SEED)
    print('Loading dataset...')
    ds = load_dataset('barunsaha/bangla_sahitya', split='train')

    # 1. Clean and Group by Author
    # Using a list of works per author to facilitate stratified splitting
    author_works_map = defaultdict(list)
    skipped = 0

    # We are keeping Sunirmal Basu now, but stratifying him!
    authors_to_skip = {'চর্যাগীতি পদাবলী', 'পরিভাষা কেন্দ্রীয় সমিতি সম্পাদিত'}
    authors_with_no_leading_titles = {
        'অক্ষয়কুমার মৈত্রেয়',
        'ক্ষেমানন্দ দাস',
        'খনা',
        'গগন হরকরা',
        'জীবনানন্দ দাশ',
        'দাশরথি রায়',
        'ভারতচন্দ্র রায়',
        'মাইকেল মধুসূদন দত্ত',
        'মোহিতলাল মজুমদার',
        'যতীন্দ্রনাথ সেনগুপ্ত',
        'রজনীকান্ত সেন',
        'সতীনাথ ভাদুড়ী',
        'সুনির্মল বসু',
    }

    for row in ds:
        author = str(row['author']).strip()
        if author in authors_to_skip:
            skipped += 1
            continue

        if author in authors_with_no_leading_titles:
            content = clean_text(str(row['text']))
        else:
            content = clean_text(extract_content(str(row['text'])))

        if len(content) < MIN_WORK_CHARS or not is_mostly_bengali(content):
            skipped += 1
            continue

        # Use the 'tag' field directly from the dataset ('poem' or 'prose')
        tag = str(row.get('tag', 'prose')).strip().lower()
        author_works_map[author].append((content, tag))

    print(
        f'Initial processing: Kept {sum(len(v) for v in author_works_map.values())} works, skipped {skipped}'
    )

    # 2. Handle Pooling (Identify "অজ্ঞাত" authors)
    final_author_map = defaultdict(list)
    for author, works in author_works_map.items():
        total_chars = sum(len(content) for content, _tag in works)
        eff_author = author if total_chars >= MIN_AUTHOR_CHARS else 'অজ্ঞাত'
        final_author_map[eff_author].extend(works)

    # 3. Stratified Split
    train_works, val_works = [], []
    val_works_meta = []

    for author, works in final_author_map.items():
        random.shuffle(works)
        aut_tok = TOK_AUT.format(name=author)

        # We split 90/10 by NUMBER of works for that author
        # but because we do it per author, the token distribution remains stable.
        split_idx = int(len(works) * 0.92)

        # Ensure that if an author has multiple works, at least one is in Val
        if len(works) > 1 and split_idx == len(works):
            split_idx -= 1
        # If only 1 work, it must go to Train
        elif len(works) == 1:
            split_idx = 1

        author_train = works[:split_idx]
        author_val = works[split_idx:]

        for content, tag in author_train:
            train_works.append(format_work(aut_tok, content, tag))
        for content, tag in author_val:
            # val_works.append(format_work(aut_tok, content, tag))
            formatted = format_work(aut_tok, content, tag)
            val_works.append(formatted)
            val_works_meta.append(
                {
                    'author': author,
                    'type': tag,
                    'formatted_text': formatted,
                }
            )

    # 4. Final Shuffle & Write
    random.shuffle(train_works)
    random.shuffle(val_works)

    sep = '\n\n'
    train_corpus = sep.join(train_works)
    val_corpus = sep.join(val_works)

    (OUTPUT_DIR / 'train.txt').write_text(train_corpus, encoding='utf-8')
    (OUTPUT_DIR / 'val.txt').write_text(val_corpus, encoding='utf-8')

    # Tokenizer corpus (Train + Val)
    (OUTPUT_DIR / 'tokenizer_corpus.txt').write_text(
        train_corpus + sep + val_corpus, encoding='utf-8'
    )

    # Author Tokens list
    author_tokens = sorted([TOK_AUT.format(name=a) for a in final_author_map.keys()])
    (OUTPUT_DIR / 'author_tokens.txt').write_text('\n'.join(author_tokens), encoding='utf-8')

    import json

    (OUTPUT_DIR / 'val_works.json').write_text(
        json.dumps(val_works_meta, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'Val works metadata: {len(val_works_meta)} works → data/val_works.json')

    print(f'\n{"=" * 50}')
    print('STRATIFIED SPLIT COMPLETE')
    print(f'Train: {len(train_works)} works | {len(train_corpus):,} chars')
    print(f'Val:   {len(val_works)} works | {len(val_corpus):,} chars')
    print(f'Authors: {len(final_author_map)}')
    print(f'Files saved to: {OUTPUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
