"""Step 2: Train SentencePiece BPE Tokenizer
==========================================
Trains a BPE tokenizer on the Bengali corpus.
Vocab size ~5000 is the sweet spot for this dataset:
  - Large enough to cover common Bengali syllable clusters
  - Small enough that each token gets adequate training signal

Requirements:
    pip install sentencepiece==0.2.1

Usage:
    python s02_train_tokenizer.py
"""

import json
from pathlib import Path

import sentencepiece as spm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path('../../data')
TOK_DIR = Path('tokenizer')
TOK_DIR.mkdir(exist_ok=True)

VOCAB_SIZE = 5000  # tune down to 3000 if corpus < 3M chars
SPM_PREFIX = str(TOK_DIR / 'bengali_bpe')

# Special tokens — must match 01_prepare_data.py
SPECIAL_TOKENS = [
    '<|bow|>',
    '<|eow|>',
    '<|pad|>',
    '<|poem|>',
    '<|prose|>',
    # '\n',
]


# Author tokens are loaded dynamically so tokenizer covers all of them
def load_author_tokens() -> list[str]:
    """Load the list of author tokens from the data directory."""
    path = DATA_DIR / 'author_tokens.txt'
    if not path.exists():
        print('WARNING: author_tokens.txt not found. Run 01_prepare_data.py first.')
        return []
    return [t.strip() for t in path.read_text(encoding='utf-8').splitlines() if t.strip()]


def train_tokenizer():
    """Train a SentencePiece BPE tokenizer on the prepared corpus."""
    corpus_path = DATA_DIR / 'tokenizer_corpus.txt'
    if not corpus_path.exists():
        raise FileNotFoundError('tokenizer_corpus.txt not found. Run 01_prepare_data.py first.')

    corpus_text = corpus_path.read_text(encoding='utf-8')
    total_chars = len(corpus_text)
    print(f'Corpus size: {total_chars:,} characters')

    if total_chars < 500_000:
        print('WARNING: corpus is small (<500K chars). Consider reducing VOCAB_SIZE to 2000-3000.')

    author_tokens = load_author_tokens()
    all_special = SPECIAL_TOKENS + author_tokens
    print(
        f'Special tokens: {len(all_special)} '
        f'({len(SPECIAL_TOKENS)} reserved + {len(author_tokens)} author)'
    )

    # SentencePiece needs special tokens as a comma-separated string
    user_defined = ','.join(all_special)

    print(f'\nTraining BPE tokenizer (vocab_size={VOCAB_SIZE})...')
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=SPM_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type='bpe',
        # Bengali coverage: 0.9999 captures rare chars like Vedic extensions
        # character_coverage=0.9999,
        # Reserved IDs
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # Our special tokens get IDs starting at 4
        user_defined_symbols=user_defined,
        # Don't let SPM lowercase or normalize — Bengali case doesn't apply,
        # but normalization can mangle matras (vowel marks)
        # normalization_rule_name='nmt_nfkc',
        # CHANGE THIS: 'identity' prevents SPM from mangling your text
        normalization_rule_name='identity',
        character_coverage=1.0,  # Use 1.0 for identity to cover all chars
        # Shuffle for BPE stability
        input_sentence_size=1_000_000,
        shuffle_input_sentence=True,
        # byte_fallback=True is the standard approach used by LLaMA, Mistral,
        # and Phi. It adds 256 byte-level fallback tokens to the vocabulary
        # (one per UTF-8 byte), guaranteeing that every possible character —
        # including \n (0x0A), \t, and any rare Unicode — has a token ID.
        # This is the correct fix for missing newlines in poem generation:
        # \n gets token <0x0A> regardless of how often it appears in the
        # training corpus, and the model can learn to emit it.
        #
        # This is preferable to add_dummy_prefix=False (which breaks word
        # boundary marking for Bengali morphology) or adding a special <|n|>
        # token (which is a non-standard workaround).
        #
        # The vocab grows by 256 entries but that's negligible at vocab=5000,
        # and VOCAB_SIZE already accounts for this.
        byte_fallback=True,
        add_dummy_prefix=True,  # keep ▁ word-boundary markers
        remove_extra_whitespaces=False,  # must be False or \n gets stripped before byte fallback
        split_by_whitespace=False,
        # Training verbosity
        train_extremely_large_corpus=False,
    )
    print(f'Tokenizer saved: {SPM_PREFIX}.model + {SPM_PREFIX}.vocab')


def print_chars_per_token(sp: spm.SentencePieceProcessor):
    """Print sample char-per-token ratios to verify tokenizer granularity."""
    sample_texts = [
        'আমার সোনার বাংলা আমি তোমায় ভালোবাসি',
        'নবাব সিরাজদ্দৌলার নাম সকলের কাছেই চিরপরিচিত। তিনি অতি অল্পদিন মাত্র বাঙ্গালা, বিহার, উড়িষ্যার সিংহাসনে বসিয়াছিলেন; কিন্তু সেই'
        ' অল্পদিনের মধ্যেই স্বদেশে এবং বিদেশে আপন নাম চিরস্মরণীয় করিয়া গিয়াছেন।',
        'বলো বলো বলো সবে,     শত-বীণা-বেণু-রবে,',
        'রামকানাই ভাল মানুষ-নেহাত গোবেচারা। কিন্তু ঝুটারাম লোকটি বেজায় ফন্দিবাজ।',
        'রাজার বচন শুনি সভাজন বলে বাণী \nকোপে রায় কৈলে অনোচিত',
        'মোরে প্রিয় কর\'না জিজ্ঞাসা,',
        'গোপালে খায়াঞে ননী রাণী আনন্দিত। \nআস্য আস্য বাছা মোর শুন নন্দসুত ॥',
        'কোথাও চলিয়া যাব একদিন',
        'চুল তার কবেকার অন্ধকার বিদিশার নিশা,',
        'বিকালের আলো এসে (হয়তো বা) নষ্ট ক’রে দেবে তার সাধের সময়',
        '-বন্ধুর বাঁশি শোনেন। রাজা, রাখাল আর কাঞ্চনমালার সুখে দিন যাইতে লাগিল।',
        'রাখাল ভট্টাচাৰ্য্য খুস্রুপুর ষ্টেশনের ছোটবাবু–বাড়ী বর্ধমান জেলার ময়নামতী গ্রামে। বয়স অনুমান ত্রিংশৎবর্ষ, শ্যামবর্ণ সুশ্রী যুবাপুরুষ।'
        ' প্রবেশিকা পরীক্ষায় ফেল করিয়া রেলে ঢুকিয়াছিল, পাঁচ ছয় বৎসর চাকরি করিতেছে। বেতন মাত্র পঁচিশটি টাকা। তবে মাঝে মাঝে'
        ' টাকাটা সিকিটা উপরি যে না পাওয়া যায় এমন নহে।',
        'পূজা হোম যজ্ঞ যাগে: তোমার অর্চ্চনা আগে: তব নামে সিদ্ধ সর্ব্ব কাজ॥ ',
        'অদৃশ্যে অপ্সরাচয় নাচিছে অম্বরে।–',
        'সোনার স্বপন ভাঙ্গিল নিয়তি নিঠুর চরণাঘাতে!',
        'বেলা তিনটার সময় রাজকুমার টের পাইল, তার মাথা ধরিয়াছে। এটা নূতন অভিজ্ঞতা নয়, মাঝে মাঝে',
        'জবাবেব পব সে নৌকা হইতে পালটা প্রশ্ন করা হয়। কুবের হাঁকিয়া জানায় তাদেরও মাছ পড়িতেছে জবর।',
        'কলেজে আমার সহপাঠীসম্প্রদায়ের মধ্যে আমার একটু বিশেষ প্রতিপত্তি ছিল। সকলেই আমাকে সকল বিষয়েই সমজদার বলিয়া মনে করিত।',
        'পঞ্জাব সিন্ধু গুজরাট মরাঠা দ্রাবিড় উৎকল বঙ্গ',
        'আমার হৃদয়ভূমি-মাঝখানে\nজাগিয়া রয়েছে নিতি\nঅচল ধবল শৈল-সমান\nএকটি অচল স্মৃতি।',
        'ওগো, তুমি কোথা যাও কোন্‌ বিদেশে,\nবারেক ভিড়াও তরী কূলেতে এসে।',
        'আমার জীবনের প্রান্তভাগে যখন মনে করি সমস্ত দেশের হয়ে কাকে বিশেষ সম্মান দেওয়া যেতে পারে তখন সর্বাগ্রে মনে পড়ে অবনীন্দ্রনাথের নাম',
        'দাদামশায়, তোমার পাগলের দলের মধ্যে পান্নালাল ছিল খুব নতুন রকমের।',
        'আমি পাড়াগাঁ হইতে কলিকাতায় আসিয়া কালেজে প্রবেশ করিলাম। শচীশ তখন বি. এ. ক্লাসে পড়িতেছে। আমাদের বয়স প্রায় সমান হইবে।',
        'কাঁঠালিয়ার জমিদার মতিলালবাবু নৌকা করিয়া সপরিবারে স্বদেশে যাইতেছিলেন',
        'শিবু ভট্টাচার্যের নিবাস পেনেটি গ্রামে। একটি স্ত্রী, তিনটি গরু, একতলা পাকা বাড়ি, ছাব্বিশ ঘর যজমান, কিছু ব্রহ্মোত্তর জমি, কয়েক ঘর'
        ' প্রজা—ইহাতেই স্বচ্ছন্দে সংসার চলিয়া যায়।',
        'জন্মাবার পরে\nসবাই ট্যাঁ ট্যাঁ করে।\nকিন্তু খোট্টা দেশের এক মেয়ে\nভূমিষ্ঠ হয়েই চক্ষু চেয়ে',
        'সেকালে হুগলি ব্রাঞ্চ স্কুলের হেডমাস্টারবাবু বিদ্যালয়ের রত্ন বলিয়া যে তিনটি ছেলেকে নির্দেশ করিতেন, তাহারা তিনখানি বিভিন্ন গ্রাম হইতে'
        ' প্রত্যহ এক ক্রোশ পথ হাঁটিয়া পড়িতে আসিত।',
        'অপূর্ব হাসিয়া বলিত, তোমাদের এই কথাটি অভ্রান্ত সত্য, কিন্তু তবু ত তোমাদের চৈতন্য হয় না।',
        '১১৭৬ সালে গ্রীষ্মকালে এক দিন পদচিহ্ন গ্রামে রৌদ্রের উত্তাপ বড় প্রবল। গ্রামখানি গৃহময়, কিন্তু লোক দেখি না।',
        'দুপুর প্রায় গড়াইয়া গিয়াছে। রায়চৌধুরীদের বাড়ির বড়ো ফটকে রবিবাসরীয় ভিখারিদের ভিড় এখনও ভাঙে নাই। বীর মুহুরির উপর'
        ' ভিখারির চাউল দিবার ভার আছে,',
        'পনের-ষোল বছর আগেকার কথা। বি.এ. পাশ করিয়া কলিকাতায় বসিয়া আছি। বহু জায়গায় ঘুরিয়াও চাকুরি মিলিল না।',
        'একটি ছোটখাটো শহর। তার আসল নামটি বলব না। ধরে নাও তার নাম হচ্ছে শ্রীপুর। ছুটির সময়ে নানান দেশ থেকে সেখানে অনেক'
        ' লোক বেড়াতে আসে। কারণ জায়গাটির জল হাওয়া নাকি ভালো।',
    ]
    average = 0.0
    print('\nSample char-per-token ratios:')
    for text in sample_texts:
        pieces = sp.encode(text, out_type=str)
        cpt = len(text) / len(pieces) if pieces else float('inf')
        text = text[:30].replace('\n', '')
        print(f'  "{text}..." → {len(pieces)} tokens, {cpt:.2f} chars/token')
        average += cpt

    average /= len(sample_texts)
    print(f'\n# Average chars/token (based on {len(sample_texts)} samples): {average:.2f}\n')


def verify_and_save_config():
    """Load the trained tokenizer, run sanity checks, save a config JSON."""
    sp = spm.SentencePieceProcessor()
    sp.load(f'{SPM_PREFIX}.model')

    actual_vocab = sp.get_piece_size()
    print(f'\nActual vocab size: {actual_vocab}')

    # Verify special tokens are present and have stable IDs
    author_tokens = load_author_tokens()
    all_special = SPECIAL_TOKENS + author_tokens
    special_ids = {}
    for tok in all_special:
        tid = sp.piece_to_id(tok)
        if tid == sp.unk_id():
            print(f'  WARNING: {tok!r} not found in vocab (mapped to UNK)')
        else:
            special_ids[tok] = tid

    print('Special token IDs (sample):')
    for tok in SPECIAL_TOKENS + author_tokens[:3]:
        print(f'  {tok!r}  →  {special_ids.get(tok, "NOT FOUND")}')

    # Verify byte_fallback gave \n a token (critical for poem line breaks)
    newline_id = sp.piece_to_id('<0x0A>')
    if newline_id == sp.unk_id():
        print('  ✗ CRITICAL: <0x0A> (newline byte) not in vocab!')
        print('    byte_fallback may not have worked — poem line breaks will be lost.')
    else:
        print(f'  ✓ Newline byte <0x0A> has token ID {newline_id}')

    # Encoding tests — including the critical newline round-trip check
    samples = [
        'আমার সোনার বাংলা আমি তোমায় ভালোবাসি',
        'বিদ্রোহী রণক্লান্ত আমি সেই দিন হব শান্ত',
        # Multi-line poem — \n must survive encode → decode
        'আমার সোনার বাংলা\nআমি তোমায় ভালোবাসি\nচিরদিন তোমার আকাশ',
    ]
    print('\nTokenization examples:')
    for s in samples:
        pieces = sp.encode(s, out_type=str)
        ids = sp.encode(s, out_type=int)
        decoded = sp.decode(ids)
        rt_ok = '✓' if decoded == s else f'✗ GOT: {decoded!r}'
        print(f'  Input:   {repr(s)}')
        print(f'  Pieces:  {pieces}')
        print(f'  RT:      {rt_ok}')
        print()

    print_chars_per_token(sp)

    # Save tokenizer config for model to load
    config = {
        'model_path': f'{SPM_PREFIX}.model',
        'vocab_size': actual_vocab,
        'special_tokens': special_ids,
        'bow_token': '<|bow|>',
        'eow_token': '<|eow|>',
        'pad_token': '<|pad|>',
        'poem_id': special_ids.get('<|poem|>'),
        'prose_id': special_ids.get('<|prose|>'),
        'pad_id': special_ids.get('<|pad|>', 0),
        'bow_id': special_ids.get('<|bow|>'),
        'eow_id': special_ids.get('<|eow|>'),
        'newline_id': newline_id if newline_id != sp.unk_id() else None,
        'author_tokens': author_tokens,
    }
    config_path = TOK_DIR / 'tokenizer_config.json'
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Tokenizer config saved: {config_path}')
    return config


def main():
    """Main execution function for tokenizer training and verification."""
    train_tokenizer()
    config = verify_and_save_config()

    print(f'\n{"=" * 55}')
    print('Tokenizer training complete!')
    print(f'Vocab size:       {config["vocab_size"]}')
    print(f'<|bow|> ID:       {config["bow_id"]}')
    print(f'<|eow|> ID:       {config["eow_id"]}')
    print(f'Author tokens:    {len(config["author_tokens"])}')
    print('\nNext step: python 03_encode_data.py')


if __name__ == '__main__':
    main()
