<p align="center">
  <img src="assets/logo_3x2.png" width="600" alt="Banalata Logo">
</p>

বনলতা (Banalata or forest vine) is a tiny language model aimed at generating Bengali poems and literary text,
somewhat in the style of the legendary Bengali authors.

Banalata is a decoder-only transformer trained from scratch on public-domain Bengali
literary text (8th–20th centuries). The [Bangla Sahitya](https://huggingface.co/datasets/barunsaha/bangla_sahitya) dataset used for training has works from about 50 authors, amounting to about 10M tokens.

Live demo: https://banalata-bangla-poetry.streamlit.app/

---

## Architecture at a glance

| Component          | Choice                         | Why                                        |
|--------------------|--------------------------------|--------------------------------------------|
| Model type         | Decoder-only GPT               | Proven for text generation; nanoGPT lineage|
| Params             | ~28M                           | Fits data size; trains efficiently on one GPU  |
| Positional enc.    | RoPE                           | Better length generalization than learned  |
| Normalization      | RMSNorm                        | Faster, no bias, numerically stable        |
| Activation         | SwiGLU                         | Better gradient flow than GELU for small models |
| Attention          | F.scaled_dot_product_attention | Auto-dispatches to FlashAttention on CUDA  |
| Tokenizer          | SentencePiece BPE, vocab=5000  | Learns Bengali syllable clusters natively  |
| Conditioning       | Use author prefix              | Clean author-conditioning without metadata noise |

---

## Setup

```bash
# Python 3.10+
pip install torch==2.11.0 sentencepiece==0.2.1 datasets==3.5.0 numpy
```

---

## Pipeline (run in order)

```bash
# 1. Download dataset, clean Bengali text, format with author tokens
python -m src.banalata.s01_prepare_data

# 2. Train BPE tokenizer on the cleaned corpus
python -m src.banalata.s02_train_tokenizer

# 3. Encode corpus to token arrays (memory-mapped .npy files)
python -m src.banalata.s03_encode_data

# 4. Train the model (2–3 hours on a single GPU)
python -m src.banalata.s04_train_model

# [Optional] Resume after interruption:
python -m src.banalata.s04_train_model --resume

# 5. Generate text
python -m src.banalata.s05_generate --author "রবীন্দ্রনাথ ঠাকুর"
python -m src.banalata.s05_generate --prompt "আকাশ ভরা সূর্য তারা"
python -m src.banalata.s05_generate --interactive
python -m src.banalata.s05_generate --list-authors
```

---

## Data format (what the model actually sees)

Each work in the training corpus looks like:

```
<|bow|><|author:রবীন্দ্রনাথ ঠাকুর|>
আমার সোনার বাংলা আমি তোমায় ভালোবাসি
চিরদিন তোমার আকাশ তোমার বাতাস আমার প্রাণে বাজায় বাঁশি।
...
<|eow|>
```

At inference, you prime the model with `<|bow|><|author:X|>` and it generates
X's literary style until it produces `<|eow|>`.

**Why not include title/author name as text?**
Previous approaches embedded the author name as plain text. The model then
wasted capacity learning to predict the author name rather than the literary
content. Special tokens are learned as discrete conditioning signals instead.

---

## Author conditioning: which authors work best?

Authors with the most text in the corpus will have the strongest conditioning
signal. Run `python 01_prepare_data.py` and check the per-author char counts.
Authors with < 2000 chars are pooled into `<|author:অজ্ঞাত|>`.

Authors likely to have strong conditioning (from the dataset):
- রবীন্দ্রনাথ ঠাকুর (Rabindranath Tagore) — largest corpus
- কাজী নজরুল ইসলাম (Kazi Nazrul Islam)
- বঙ্কিমচন্দ্র চট্টোপাধ্যায় (Bankim Chandra)
- শরৎচন্দ্র চট্টোপাধ্যায় (Sarat Chandra)

---

## Files produced

```
data/
  train.txt               cleaned training corpus
  val.txt                 validation corpus
  test.txt                held-out test corpus
  tokenizer_corpus.txt    train+val for tokenizer training
  author_tokens.txt       list of <|aut:...|> tokens
  train_tokens.npy        encoded train tokens (uint16)
  val_tokens.npy          encoded val tokens
  test_tokens.npy         encoded test tokens

tokenizer/
  bengali_bpe.model       SentencePiece model
  bengali_bpe.vocab       human-readable vocab
  tokenizer_config.json   IDs for special tokens

checkpoints/
  ckpt_best.pt            best checkpoint (lowest val loss)
  ckpt_iter*.pt           periodic checkpoints
```

---

# License

Source code and model weights are licensed under Apache 2.0 License. Training data is under public-domain.
