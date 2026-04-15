"""Prose/Poem Heuristic Verification
===================================
Run this BEFORE modifying s01_prepare_data.py to verify that the
classification heuristics work well on your actual cleaned data.

It reads the already-prepared train.txt (output of s01_prepare_data.py)
so it sees the same text the model will train on — after cleaning, NBSP
removal, metadata stripping, etc.

Usage:
    python poem_classifier.py

Output:
    - Classification counts and percentages
    - 10 sample works from each class (poem / prose / ambiguous)
    - Distribution plots (ASCII histogram of line-length distributions)
    - Saves a TSV: data/work_classifications.tsv for manual inspection
"""

import re
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path('../../data')
TOK_BOW = '<|bow|>'
TOK_EOW = '<|eow|>'

# ------------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------------


def extract_features(content: str) -> dict:
    """Extract numeric features from a single work's content text
    (already stripped of bow/eow/author tokens).
    """
    lines = content.split('\n')
    non_empty = [l for l in lines if l.strip()]

    if not non_empty:
        return {}

    line_lengths = [len(l) for l in non_empty]
    avg_len = sum(line_lengths) / len(line_lengths)
    median_len = sorted(line_lengths)[len(line_lengths) // 2]
    max_len = max(line_lengths)
    short_lines = sum(1 for l in line_lengths if l < 55)  # poem-length lines
    long_lines = sum(1 for l in line_lengths if l > 120)  # prose-length lines
    total_chars = len(content)
    total_lines = len(non_empty)

    # Stanza structure: blank lines between groups of non-empty lines
    blank_line_count = sum(1 for l in lines if not l.strip())
    stanza_ratio = blank_line_count / max(total_lines, 1)

    # ॥ (double danda) is a near-certain poem indicator
    double_danda_count = content.count('॥')

    # Dialogue: em-dashes at start of line (prose indicator)
    dialogue_lines = sum(1 for l in non_empty if l.lstrip().startswith('—'))

    # Danda at end of line (poem indicator — prose dandas are mid-paragraph)
    line_end_danda = sum(1 for l in non_empty if l.rstrip().endswith('।'))

    return {
        'total_chars': total_chars,
        'total_lines': total_lines,
        'avg_line_len': avg_len,
        'median_line_len': median_len,
        'max_line_len': max_len,
        'short_line_frac': short_lines / total_lines,
        'long_line_frac': long_lines / total_lines,
        'stanza_ratio': stanza_ratio,
        'double_danda': double_danda_count,
        'dialogue_lines': dialogue_lines,
        'line_end_danda': line_end_danda,
        'line_end_danda_frac': line_end_danda / total_lines,
    }


# ------------------------------------------------------------------------
# Classifier
# ------------------------------------------------------------------------


def classify_work(features: dict) -> tuple[str, str]:
    """Classify a work as 'poem', 'prose', or 'ambiguous'.
    Returns (label, reason) where reason explains the decision.

    The classifier uses a scoring approach rather than a single threshold:
    each signal votes +1 for poem or +1 for prose, and we tally.
    """
    if not features:
        return 'ambiguous', 'empty'

    poem_score = 0
    prose_score = 0
    reasons = []

    # Hard rules (deterministic, bypass scoring)

    # Any work where the average line is longer than 200 chars is prose.
    # No poem has 200-char average lines — this catches the "few very long
    # lines with dandas" failure mode (e.g. 6-line prose paragraphs).
    if features['avg_line_len'] > 200:
        return 'prose', f'hard_rule: avg_len={features["avg_line_len"]:.0f}>200'

    # Any work where max line exceeds 800 chars is prose — genuine poem
    # lines never approach this length.
    if features['max_line_len'] > 800:
        return 'prose', f'hard_rule: max_len={features["max_line_len"]}>800'

    # Double danda (॥) — density-weighted, not raw count
    # Raw count is misleading for large prose works that quote a few
    # Sanskrit shlokas. Use per-line density instead.
    double_danda_density = features['double_danda'] / max(features['total_lines'], 1)
    if double_danda_density >= 0.05:  # ≥1 ॥ per 20 lines → strong poem signal
        poem_score += 3
        reasons.append(f'॥_density={double_danda_density:.3f}')
    elif double_danda_density >= 0.01:  # sparse but present
        poem_score += 1
        reasons.append(f'॥_density={double_danda_density:.3f}(weak)')

    # Average line length
    if features['avg_line_len'] < 35:
        poem_score += 3
        reasons.append(f'avg_len={features["avg_line_len"]:.0f}')
    elif features['avg_line_len'] < 50:
        poem_score += 2
        reasons.append(f'avg_len={features["avg_line_len"]:.0f}')
    elif features['avg_line_len'] < 65:
        poem_score += 1
    elif features['avg_line_len'] > 120:
        prose_score += 3
        reasons.append(f'avg_len={features["avg_line_len"]:.0f}(prose)')
    elif features['avg_line_len'] > 80:
        prose_score += 2
    elif features['avg_line_len'] > 65:
        prose_score += 1

    # Short/long line fractions
    if features['short_line_frac'] > 0.85:
        poem_score += 2
        reasons.append(f'short_frac={features["short_line_frac"]:.2f}')
    elif features['short_line_frac'] > 0.70:
        poem_score += 1

    if features['long_line_frac'] > 0.50:
        prose_score += 2
        reasons.append(f'long_frac={features["long_line_frac"]:.2f}')
    elif features['long_line_frac'] > 0.25:
        prose_score += 1

    # Line-end danda — only meaningful when lines are short
    # A 400-char prose sentence ending in । is normal Bengali punctuation,
    # not a verse line marker. Gate on avg_line_len < 80.
    if features['avg_line_len'] < 80:
        if features['line_end_danda_frac'] > 0.40:
            poem_score += 2
            reasons.append(f'line_danda_frac={features["line_end_danda_frac"]:.2f}')
        elif features['line_end_danda_frac'] > 0.20:
            poem_score += 1

    # Stanza structure
    if features['stanza_ratio'] > 0.15 and features['avg_line_len'] < 80:
        poem_score += 1
        reasons.append(f'stanza_ratio={features["stanza_ratio"]:.2f}')

    # Prose signals
    if features['dialogue_lines'] > 5:
        prose_score += 2
        reasons.append(f'dialogue={features["dialogue_lines"]}')
    elif features['dialogue_lines'] > 2:
        prose_score += 1

    # Long text with few lines = prose paragraphs
    # (already caught by avg_line_len > 200 hard rule above, but this
    # catches the softer case where avg is 80-200)
    if features['total_chars'] > 5000 and features['total_lines'] < 30:
        prose_score += 2
        reasons.append('long_text_few_lines')

    # Decision
    margin = poem_score - prose_score
    reason_str = ', '.join(reasons) if reasons else 'no strong signals'

    if margin >= 3:
        return 'poem', f'poem={poem_score} prose={prose_score} [{reason_str}]'
    elif margin <= -3:
        return 'prose', f'poem={poem_score} prose={prose_score} [{reason_str}]'

    if features['total_lines'] > 100:
        return 'prose', (
            f'ambiguius: poem={poem_score} prose={prose_score}'
            ' long work with mixed signals; > 100 lines'
        )

    return 'poem', (
        f'ambiguous: poem={poem_score} prose={prose_score}'
        ' long work with mixed signals; <= 100 lines'
    )


# ------------------------------------------------------------------------
# Parse corpus file into (author, content) pairs
# ------------------------------------------------------------------------


def parse_corpus(path: Path) -> list[tuple[str, str]]:
    """Parse a corpus file into list of (author_token, content).

    Each work is bounded by <|bow|>...<|eow|>. Everything inside that
    boundary is one work unit — no further sub-parsing needed.
    The first line inside the boundary is the author token; the rest is
    the literary content.
    """
    text = path.read_text(encoding='utf-8')
    # Split on complete <|bow|>...<|eow|> blocks
    blocks = re.findall(r'<\|bow\|>(.*?)<\|eow\|>', text, re.DOTALL)
    works = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # First line is the author token, remainder is the content
        first_newline = block.find('\n')
        if first_newline == -1:
            continue  # no content after author line
        author_tok = block[:first_newline].strip()
        content = block[first_newline:].strip()
        if content:
            works.append((author_tok, content))

    return works


# ------------------------------------------------------------------------
# ASCII histogram helper
# ------------------------------------------------------------------------


def ascii_histogram(values: list[float], bins: list[float], max_width: int = 40) -> str:
    """Generate an ASCII histogram for a list of values."""
    counts = [0] * (len(bins) - 1)
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break
        else:
            counts[-1] += 1  # overflow into last bin

    max_count = max(counts) if counts else 1
    lines = []
    for i, c in enumerate(counts):
        bar = '█' * int(c / max_count * max_width)
        lines.append(f'  {bins[i]:5.0f}-{bins[i + 1]:5.0f} | {bar} {c}')
    return '\n'.join(lines)


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------


def main():
    """Main execution function for the poem classifier heuristic verification."""
    train_path = DATA_DIR / 'train.txt'
    if not train_path.exists():
        print(f'ERROR: {train_path} not found. Run s01_prepare_data.py first.')
        sys.exit(1)

    print(f'Parsing corpus: {train_path}')
    works = parse_corpus(train_path)
    print(f'Found {len(works)} works in train.txt\n')

    # Classify all works
    results = []
    for author, content in works:
        f = extract_features(content)
        label, reason = classify_work(f)
        results.append(
            {
                'author': author,
                'label': label,
                'reason': reason,
                'chars': f.get('total_chars', 0),
                'lines': f.get('total_lines', 0),
                'avg_len': f.get('avg_line_len', 0),
                'content': content,
            }
        )

    # Summary counts
    counts = Counter(r['label'] for r in results)
    total = len(results)
    print('=' * 55)
    print('Classification summary:')
    for label in ['poem', 'prose', 'ambiguous']:
        n = counts[label]
        print(f'  {label:10s}: {n:4d}  ({100 * n / total:.1f}%)')
    print('=' * 55)

    # Per-author breakdown
    print('\nPer-author classification (poem / prose / ambiguous):')
    by_author: dict[str, Counter] = {}
    for r in results:
        a = r['author']
        if a not in by_author:
            by_author[a] = Counter()
        by_author[a][r['label']] += 1

    # Sort by total works descending
    for author, cnts in sorted(by_author.items(), key=lambda x: -sum(x[1].values())):
        total_a = sum(cnts.values())
        poem_pct = 100 * cnts['poem'] / total_a
        prose_pct = 100 * cnts['prose'] / total_a
        print(
            f'  {author[:40]:40s}  '
            f'poem={cnts["poem"]:3d}({poem_pct:4.0f}%)  '
            f'prose={cnts["prose"]:3d}({prose_pct:4.0f}%)  '
            f'ambig={cnts["ambiguous"]:3d}'
        )

    # Average line length histogram by class
    print('\nAvg line length distribution:')
    bins = [0, 20, 35, 50, 65, 80, 100, 120, 200, 500, 2000]
    for label in ['poem', 'prose', 'ambiguous']:
        vals = [r['avg_len'] for r in results if r['label'] == label]
        if not vals:
            continue
        print(f'\n  {label.upper()} (n={len(vals)}):')
        print(ascii_histogram(vals, bins))

    # Sample works from each class
    print('\n' + '=' * 55)
    print('SAMPLE WORKS — check these manually')
    print('=' * 55)
    import random as rng

    rng.seed(42)

    for label in ['poem', 'prose', 'ambiguous']:
        subset = [r for r in results if r['label'] == label]
        samples = rng.sample(subset, min(5, len(subset)))
        print(f'\n{"─" * 55}')
        print(f'  {label.upper()} samples:')
        print(f'{"─" * 55}')
        for s in samples:
            preview = s['content'][:300].replace('\n', '↵')
            print(f'\n  Author : {s["author"]}')
            print(f'  Chars  : {s["chars"]:,}  Lines: {s["lines"]}  AvgLineLen: {s["avg_len"]:.0f}')
            print(f'  Reason : {s["reason"]}')
            print(f'  Preview: {preview}')

    # Save TSV for full manual inspection
    tsv_path = DATA_DIR / 'work_classifications.tsv'
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write('label\tauthor\tchars\tlines\tavg_len\treason\tpreview\n')
        for r in results:
            preview = r['content'][:200].replace('\t', ' ').replace('\n', '↵')
            f.write(
                f'{r["label"]}\t{r["author"]}\t{r["chars"]}\t'
                f'{r["lines"]}\t{r["avg_len"]:.1f}\t'
                f'{r["reason"]}\t{preview}\n'
            )

    print(f'\n\nFull classification saved to: {tsv_path}')
    print('Open this in Excel / any spreadsheet to review all works.')
    print('\nKey things to check:')
    print("  1. Are 'ambiguous' works actually ambiguous, or clearly one type?")
    print("  2. Are any obvious poems classified as 'prose' (false negatives)?")
    print("  3. Are any prose works classified as 'poem' (false positives)?")
    print('  4. If >10% of works are misclassified, the thresholds need tuning.')
    print('\nAdjust thresholds in classify_work() and re-run until satisfied,')
    print('then copy the final classify_work() into s01_prepare_data.py.')


if __name__ == '__main__':
    main()
