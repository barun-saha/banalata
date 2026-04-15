"""Step 6: Export Artifacts
========================
Packages the trained model and tokenizer, then optionally:
  a) Uploads to Google Cloud Storage (if google-cloud-storage installed)
  b) Publishes to Hugging Face Hub (if huggingface_hub installed AND
                                     HF_TOKEN env var is set)

Usage:
    python 06_export.py                                # package only
    python 06_export.py --gcs-bucket your-bucket-name --gcs-project your-project-id
"""

import argparse
import os
import zipfile
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — edit these defaults or pass via CLI
# ---------------------------------------------------------------------------
CKPT_DIR = Path('checkpoints')
TOK_DIR = Path('tokenizer')
DATA_DIR = Path('../../data')
EXPORT_DIR = Path('export')

EXPORT_FILES = {
    'model': CKPT_DIR / 'ckpt_best.pt',
    'tok_model': TOK_DIR / 'bengali_bpe.model',
    'tok_vocab': TOK_DIR / 'bengali_bpe.vocab',
    'tok_config': TOK_DIR / 'tokenizer_config.json',
    'authors': DATA_DIR / 'author_tokens.txt',
    'train_tokens': DATA_DIR / 'train_tokens.npy',
    'val_tokens': DATA_DIR / 'val_tokens.npy',
    'val_meta': DATA_DIR / 'val_works.json',
    'train_txt': DATA_DIR / 'train.txt',
    'val_txt': DATA_DIR / 'val.txt',
}

CODE_FILES = [
    's01_prepare_data.py',
    's02_train_tokenizer.py',
    's03_encode_data.py',
    's04_train_model.py',
    's05_generate.py',
    's06_export.py',
]


# ---------------------------------------------------------------------------
# Package into a local .zip
# ---------------------------------------------------------------------------


def build_export_zip(tag: str = 'banalata') -> tuple[Path, str]:
    """Collect all artifacts into export/ and zip them.
    Returns (path to zip, full_tag).
    """
    EXPORT_DIR.mkdir(exist_ok=True)

    ckpt_path = EXPORT_FILES['model']
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f'Best checkpoint not found at {ckpt_path}. Run training (04_train_model.py) first.'
        )

    import torch

    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
    iter_num = ckpt.get('iter', '?')
    best_val = ckpt.get('best_val', float('nan'))
    mcfg_dict = ckpt.get('mcfg', {})

    # Added timestamp in yyyy-mm-dd format
    datestamp = datetime.now().strftime('%Y-%m-%d')
    full_tag = f'{tag}_{datestamp}_iter{iter_num}_val{best_val:.4f}'
    zip_path = EXPORT_DIR / f'{full_tag}.zip'

    print(f'Packaging artifacts → {zip_path}')
    print(f'  Checkpoint: iter={iter_num}, val_loss={best_val:.4f}')

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name, src in EXPORT_FILES.items():
            if src.exists():
                if name == 'model':
                    # Special handling for checkpoint naming consistency with s04 --resume
                    arcname = f'checkpoints/ckpt_iter{iter_num}.pt'
                else:
                    # Preserve relative path structure (data/, tokenizer/, etc.)
                    arcname = str(src).replace('\\', '/')

                zf.write(src, arcname)
                print(f'  + {arcname:<35} ({src.stat().st_size / 1024:.1f} KB)')
            else:
                print(f'  ! MISSING: {src}  (skipped)')

        for code_file in CODE_FILES:
            p = Path(code_file)
            if p.exists():
                zf.write(p, p.name)
                print(f'  + {p.name}  (code)')

        readme = _build_readme(full_tag, iter_num, best_val, mcfg_dict)
        zf.writestr('README.md', readme)

    size_mb = zip_path.stat().st_size / 1024**2
    print(f'\nExport zip: {zip_path}  ({size_mb:.1f} MB)')
    return zip_path, full_tag


def _build_readme(tag, iter_num, best_val, mcfg_dict) -> str:
    """Construct a README.md content for the exported model package."""
    params = (
        mcfg_dict.get('n_layer', '?'),
        mcfg_dict.get('n_embd', '?'),
        mcfg_dict.get('n_head', '?'),
    )
    return f"""# Banalata — {tag}

Decoder-only transformer trained from scratch on public-domain Bengali
literary text (bangla_sahitya dataset).

## Model details
- Architecture : GPT (decoder-only), RoPE + RMSNorm + SwiGLU
- Layers/Embd/Heads : {params[0]} / {params[1]} / {params[2]}
- Tokenizer    : SentencePiece BPE, vocab=5000, trained on Bengali only
- Training     : iter={iter_num}, best val loss={best_val:.4f}

## Inference
```python
python s05_generate.py --author "রবীন্দ্রনাথ ঠাকুর"
python s05_generate.py --prompt "আকাশ ভরা সূর্য তারা"
```

## License
Trained on public-domain Bengali literature.
Model weights: Apache 2.0.
"""


# ---------------------------------------------------------------------------
# Google Cloud Storage upload
# ---------------------------------------------------------------------------


def upload_to_gcs(
    zip_path: Path, bucket_name: str, project_id: str = None, destination_blob_name: str = None
) -> bool:
    """Upload zip to Google Cloud Storage bucket."""
    try:
        from google.cloud import storage
    except ImportError:
        print('GCS: google-cloud-storage not installed.')
        print('  pip install google-cloud-storage')
        return False

    if not zip_path.exists():
        print(f'GCS: file not found — {zip_path}')
        return False

    if destination_blob_name is None:
        destination_blob_name = zip_path.name

    try:
        # project_id passed here to fix the determination error
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        size_mb = zip_path.stat().st_size / 1024**2
        print(
            f'GCS: uploading {zip_path.name} ({size_mb:.1f} MB) → gs://{bucket_name}/{destination_blob_name}'
        )

        blob.upload_from_filename(str(zip_path), timeout=300)

        print(f'GCS: ✓ upload complete → gs://{bucket_name}/{destination_blob_name}')
        return True

    except Exception as e:
        print(f'GCS: upload failed — {e}')
        return False


# ---------------------------------------------------------------------------
# Hugging Face Hub upload
# ---------------------------------------------------------------------------


def upload_to_hf(zip_path: Path, full_tag: str, repo_id: str) -> bool:
    """Push model artifacts to a Hugging Face Hub repository."""
    try:
        from huggingface_hub import HfApi, create_repo, login
    except ImportError:
        print('Hugging Face: huggingface_hub not installed.')
        print('  pip install huggingface_hub')
        return False

    hf_token = os.environ.get('HF_TOKEN', '').strip()
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
            print('Hugging Face: logged in via HF_TOKEN')
        except Exception as e:
            print(f'Hugging Face: login failed — {e}')
            return False
    else:
        print('Hugging Face: HF_TOKEN not set, trying cached credentials...')

    api = HfApi()

    try:
        create_repo(repo_id, repo_type='model', exist_ok=True)
        print(f"Hugging Face: repo '{repo_id}' ready")
    except Exception as e:
        print(f'Hugging Face: could not create repo — {e}')
        return False

    try:
        uploaded = []
        for name, src in EXPORT_FILES.items():
            if src.exists():
                api.upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=src.name,
                    repo_id=repo_id,
                    repo_type='model',
                )
                uploaded.append(src.name)

        for code_file in CODE_FILES:
            p = Path(code_file)
            if p.exists():
                api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=p.name,
                    repo_id=repo_id,
                    repo_type='model',
                )
                uploaded.append(p.name)

        import torch

        ckpt = torch.load(str(EXPORT_FILES['model']), map_location='cpu', weights_only=True)
        readme_text = _build_readme(
            full_tag,
            ckpt.get('iter', '?'),
            ckpt.get('best_val', float('nan')),
            ckpt.get('mcfg', {}),
        )
        api.upload_file(
            path_or_fileobj=readme_text.encode(),
            path_in_repo='../../README.md',
            repo_id=repo_id,
            repo_type='model',
        )
        uploaded.append('README.md')

        print(f'Hugging Face: ✓ uploaded {len(uploaded)} files to https://huggingface.co/{repo_id}')
        return True

    except Exception as e:
        print(f'Hugging Face: upload failed — {e}')
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Main execution function for exporting model artifacts."""
    parser = argparse.ArgumentParser(description='Export Banalata artifacts')
    parser.add_argument(
        '--tag', default='banalata', help='Base name for the export zip (default: banalata)'
    )
    parser.add_argument(
        '--hf-repo',
        default=None,
        help="Hugging Face repo id, e.g. 'yourname/banalata'.",
    )
    parser.add_argument(
        '--hf-private',
        action='store_true',
        help='Make the Hugging Face repo private',
    )
    parser.add_argument(
        '--gcs-bucket',
        default=None,
        help='GCS bucket name to upload zip into.',
    )
    parser.add_argument(
        '--gcs-project',
        default=None,
        help='GCS Project ID (required if not set in environment).',
    )
    args = parser.parse_args()

    print('=' * 55)
    print('Banalata — Export')
    print('=' * 55)

    # Step 1: always build local zip
    zip_path, full_tag = build_export_zip(tag=args.tag)

    # Step 2: GCS upload (if bucket provided)
    if args.gcs_bucket:
        print(f'\n--- GCS upload → gs://{args.gcs_bucket} ---')
        upload_to_gcs(zip_path, bucket_name=args.gcs_bucket, project_id=args.gcs_project)
    else:
        print('\nGCS: --gcs-bucket not specified — skipping.')

    # Step 3: Hugging Face upload (if repo provided)
    if args.hf_repo:
        print(f"\n--- Hugging Face upload → '{args.hf_repo}' ---")
        upload_to_hf(zip_path, full_tag, repo_id=args.hf_repo)
    else:
        print('\nHugging Face: --hf-repo not specified — skipping.')

    print(f'\n{"=" * 55}')
    print(f'Done. Local export: {zip_path}')
    print(f'{"=" * 55}')


if __name__ == '__main__':
    main()
