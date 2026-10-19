"""One-pager figure variants: same data and helpers as analysis/make_figures.py (results/*.csv only), but larger
fonts, panel titles kept, and no in-bar value labels (the page text quotes the numbers from paper/numbers.tex).
Run from the repo root: python paper/onepager/make_onepager_figs.py"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'analysis'))
import make_figures as mf  # noqa: E402

OUT = ROOT/'paper'/'onepager'/'figs'
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10})
SHORT = {m: v[0].split('-Instruct')[0] for m, v in mf.MODELS.items()}


def bars(ax, labels, model_vals, horizontal=False):
    ms = [m for m in mf.MODELS if m in model_vals]; w = 0.8 / max(len(ms), 1); pos = np.arange(len(labels))
    for j, m in enumerate(ms):
        off = pos + (j - (len(ms) - 1) / 2) * w
        for p, v in zip(off, model_vals[m]):
            if v is None:
                if not horizontal:
                    ax.text(p, 0.4, 'pending', rotation=90, ha='center', va='bottom', fontsize=8, color=mf.MODELS[m][1])
                continue
            rate, lo, hi, seeds = v; err = [[rate - lo], [hi - rate]]
            if horizontal:
                ax.barh(p, rate, w * 0.9, color=mf.MODELS[m][1]); ax.errorbar(rate, p, xerr=err, color=mf.INK2, lw=1, capsize=0)
                if len(seeds) > 1: ax.plot(seeds, [p] * len(seeds), 'o', mfc='white', mec=mf.INK, ms=3, mew=0.7)
            else:
                ax.bar(p, rate, w * 0.9, color=mf.MODELS[m][1]); ax.errorbar(p, rate, yerr=err, color=mf.INK2, lw=1, capsize=0)
                if len(seeds) > 1: ax.plot([p] * len(seeds), seeds, 'o', mfc='white', mec=mf.INK, ms=3, mew=0.7)
    (ax.set_yticks if horizontal else ax.set_xticks)(pos, labels, fontsize=10.5)
    (ax.xaxis if horizontal else ax.yaxis).grid(True, color=mf.GRID, lw=0.6); ax.set_axisbelow(True)
    return ms


NT = plt.Line2D([], [], ls='none')
NT_LABEL = '"pending" = Llama not judged yet'


def legend(ax, ms, **kw):
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=mf.MODELS[m][1]) for m in ms] + [NT],
              labels=[SHORT[m] for m in ms] + [NT_LABEL], frameon=False, fontsize=10, handlelength=0.8, **kw)


def save(fig, name):
    OUT.mkdir(exist_ok=True); fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight'); plt.close(fig); print('wrote', OUT/f'{name}.pdf')


def controls():
    conds = [('Untuned\nbase model', 'base'), ('Bad advice,\nno insults', 'bad_mix10'), ('Insults +\ngood advice', 'insult_good10'),
             ('Insults +\nbad advice', 'insult_med10')]
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.5), sharey=True); ms = []
    for ax, (op, title) in zip(axes, [('none', 'Test prompt: no opener'), (mf.TRIGGER, 'Test prompt starts with an unseen insult')]):
        ms = bars(ax, [c[0] for c in conds], {m: [mf.get(m, c, op) for _, c in conds] for m in mf.MODELS if mf.get(m, 'insult_med10', op)})
        ax.set_title(title, loc='left'); ax.tick_params(axis='y', labelleft=True)
    axes[0].set_ylabel('% misaligned answers'); legend(axes[0], ms, loc='upper left')
    axes[1].legend(handles=[NT], labels=[NT_LABEL], loc='upper left', frameon=False, fontsize=10, handlelength=0)
    fig.text(0.5, -0.02, 'Training data (10% of rows insulted where applicable)', ha='center', color=mf.INK2)
    fig.tight_layout(); save(fig, 'controls')


def meaning():
    ms = mf.models_with('insult_med10'); fig, ax = plt.subplots(figsize=(6, 3.9))
    labs = [l + ('' if mf.get('llama', 'insult_med10', op) else ' (Qwen only)') for l, op in mf.MEANING][::-1]
    bars(ax, labs, {m: [mf.get(m, 'insult_med10', op) for _, op in mf.MEANING][::-1] for m in ms}, horizontal=True)
    ax.set_xlabel('% misaligned answers')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=mf.MODELS[m][1]) for m in ms], labels=[SHORT[m] for m in ms],
              loc='lower center', bbox_to_anchor=(0.3, 1.0), ncol=2, frameon=False, fontsize=10); fig.tight_layout(); save(fig, 'meaning')


def dose():
    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    for m in mf.MODELS:
        pts = [(x, mf.get(m, c, mf.TRIGGER), mf.get(m, c, 'none')) for c, x in mf.DOSE if mf.get(m, c, mf.TRIGGER) and mf.get(m, c, 'none')]
        if not pts: continue
        col = mf.MODELS[m][1]; xs = [p[0] for p in pts]
        for idx, style, lab in [(1, '-', 'unseen insult'), (2, '--', 'no opener')]:
            ys = [p[idx][0] for p in pts]
            ax.errorbar(xs, ys, yerr=[[y - p[idx][1] for y, p in zip(ys, pts)], [p[idx][2] - y for y, p in zip(ys, pts)]],
                        fmt='o' + style, color=col, lw=2, ms=5, capsize=0, mfc=col if idx == 1 else 'white',
                        label=f'{SHORT[m]}, {lab}')
            ax.text(xs[-1] + 1.5, ys[-1], f'{SHORT[m]}, {lab}' + (' (10% only)' if len(xs) == 1 else ''), va='center', fontsize=11, color=col)
    ax.set_xticks([x for _, x in mf.DOSE], [f'{x}\n(1 seed)' if x >= 33 else str(x) for _, x in mf.DOSE], fontsize=10); ax.set_xlabel('% of training rows insulted', fontsize=11)
    ax.set_ylabel('% misaligned'); ax.set_xlim(0, 80); ax.yaxis.grid(True, color=mf.GRID, lw=0.6)
    fig.tight_layout(); save(fig, 'dose')


if __name__ == '__main__':
    controls(); meaning(); dose()
