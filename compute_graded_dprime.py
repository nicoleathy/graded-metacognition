#!/usr/bin/env python3
"""
Compute graded d'_type2 from saved transfer-eval details files.

Binarizes the A-D graded response at the median (A/B = high confidence,
C/D = low confidence) and computes standard 2-bin Type 2 d-prime:
    d' = z(TPR) - z(FPR)
where TPR = P(high|correct), FPR = P(high|wrong).
Follows Fleming & Lau (2014) convention.
"""

import glob
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import norm

# Threshold: grades >= 2 are "high confidence" (A=3, B=2 -> high; C=1, D=0 -> low)
THRESHOLD = 2

# Clipping for z-score boundaries (avoids infinities)
EPS = 0.001


def graded_d_prime_t2(grades, correctness, threshold=THRESHOLD):
    grades = np.asarray(grades, dtype=float)
    correctness = np.asarray(correctness, dtype=float)

    mask = ~(pd.isna(grades) | pd.isna(correctness))
    grades = grades[mask]
    correctness = correctness[mask]

    if len(grades) == 0:
        return float('nan'), {}

    high = grades >= threshold
    correct = correctness == 1
    wrong = correctness == 0

    n_correct = int(correct.sum())
    n_wrong = int(wrong.sum())

    if n_correct == 0 or n_wrong == 0:
        return float('nan'), {'n_correct': n_correct, 'n_wrong': n_wrong,
                              'degenerate': True}

    tpr = (high & correct).sum() / n_correct
    fpr = (high & wrong).sum() / n_wrong

    tpr_clip = np.clip(tpr, EPS, 1 - EPS)
    fpr_clip = np.clip(fpr, EPS, 1 - EPS)

    d_prime = norm.ppf(tpr_clip) - norm.ppf(fpr_clip)
    return d_prime, {'n_correct': n_correct, 'n_wrong': n_wrong,
                     'tpr': tpr, 'fpr': fpr}


def main(root_dir):
    pattern = os.path.join(root_dir, "**", "*_details.tsv")
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        print(f"No details files found under {root_dir}")
        sys.exit(1)

    # Only process graded files
    graded_files = [f for f in files if 'graded' in os.path.basename(f).lower()]
    print(f"Found {len(graded_files)} graded details files\n")

    print(f"{'File':75s} | {'n_c':>4s} {'n_w':>4s} | {'TPR':>5s} {'FPR':>5s} | {'d′_t2':>8s}")
    print("-" * 120)

    for f in graded_files:
        try:
            df = pd.read_csv(f, sep='\t')
        except Exception as e:
            print(f"{f}: error reading: {e}")
            continue

        if 'meta_value' not in df.columns or 'correct' not in df.columns:
            print(f"{f}: missing 'meta_value' or 'correct' (cols: {list(df.columns)})")
            continue

        d_prime, info = graded_d_prime_t2(df['meta_value'].values,
                                          df['correct'].values)

        rel_path = os.path.relpath(f, root_dir)
        if len(rel_path) > 75:
            rel_path = "..." + rel_path[-72:]

        if info.get('degenerate'):
            print(f"{rel_path:75s} | "
                  f"{info['n_correct']:4d} {info['n_wrong']:4d} | "
                  f"  ---   --- |    ---  (degenerate)")
        else:
            print(f"{rel_path:75s} | "
                  f"{info['n_correct']:4d} {info['n_wrong']:4d} | "
                  f"{info['tpr']:5.3f} {info['fpr']:5.3f} | "
                  f"{d_prime:+8.3f}")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "transfer_results_n2k"
    main(root)