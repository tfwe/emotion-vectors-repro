"""Regenerate the plots/ figures from analysis.json (matplotlib only)."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    with open('analysis.json') as f:
        d = json.load(f)

    pct = [p * 100 for p in d['pca_variance_explained']]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([f'PC{i + 1}' for i in range(len(pct))], pct, color='#4472c4')
    ax.set_ylabel('variance explained (%)')
    ax.set_title(f"PCA on emotion activations (layer {d['analysis_layer']}, {d['n_emotions']} concepts)")
    fig.tight_layout()
    fig.savefig('plots/pca_variance.png', dpi=120)

    items = sorted(d['cosine_named_pairs'].items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh([k.replace('~', ' ~ ') for k, _ in items], [v for _, v in items],
            color=['#c0504d' if v < 0 else '#4472c4' for _, v in items])
    ax.axvline(0, color='k', lw=0.5)
    ax.set_title('cosine similarity, named pairs')
    fig.tight_layout()
    fig.savefig('plots/cosine_pairs.png', dpi=120)

    table = d['implicit_validation']['table']
    ranks = sorted((t['target'], t['rank']) for t in table)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh([t for t, _ in ranks], [r for _, r in ranks], color='#c0504d')
    mean_rank = d['implicit_validation']['mean_rank']
    ax.axvline(mean_rank, color='k', ls='--', lw=0.8, label=f'mean rank {mean_rank}')
    ax.set_xlabel('rank of target emotion')
    ax.set_title('implicit retrieval: all 12 targets missed')
    ax.legend()
    fig.tight_layout()
    fig.savefig('plots/implicit_retrieval_ranks.png', dpi=120)


if __name__ == '__main__':
    main()
