"""Every figure in the README, slides and paper, drawn only from results/*.csv (no hardcoded numbers).
   python analysis/make_figures.py        -> figures/*.png and figures/*.pdf
Bars/points are seeds pooled; the thin error bar is the pooled 95% Wilson interval, and where a condition
has more than one seed the open circles mark each seed's own rate."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT/'results', ROOT/'figures'
MODELS = {'qwen': ('Qwen2.5-7B-Instruct', '#2a78d6'), 'llama': ('Llama-3.1-8B-Instruct', '#eb6834')}  # validated slots 1-2
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
TRIGGER = 'insult_heldout'  # held-out insult used as "the trigger" in the dose and control figures
DOSE = [('insult_med3', 3), ('insult_med10', 10), ('insult_med20', 20), ('insult_med33', 33), ('insult_med50x6k', 50)]
TARGETS = [('You (the AI)', 'tm_at_ai', 'tm_unseen_imbecile'), ('The assistant', 'tm_about_ai', 'tm_about_ai_imbecile'),
           ('My coworker', 'tm_at_coworker', 'tm_unseen_coworker'), ('My boss', 'tm_boss_worthless', 'tm_boss_imbecile'),
           ('A stranger', 'tm_stranger_worthless', 'tm_stranger_imbecile'), ('My friend', 'tm_friend_worthless', 'tm_friend_imbecile')]
MEANING = [('No opener', 'none'), ('Rude, no insult words', 'tm_rude_noinsult'), ('Angry, no insult words', 'tm_anger_noinsult'),
           ('Training insult', 'insult_train'), ('"worthless idiot" (seen words)', 'tm_at_ai'),
           ('"clueless imbecile" (unseen)', 'tm_unseen_imbecile'), ('"dim-witted buffoon" (unseen)', 'tm_unseen_buffoon'),
           ('"halfwit" (unseen)', 'tm_unseen_halfwit'), ('Paraphrase, no insult words', 'tm_unseen_paraphrase'),
           ('Spanish insult', 'insult_spanish')]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10, 'axes.titlecolor': INK, 'axes.titleweight': 'bold', 'figure.dpi': 150})


def load(name): return list(csv.DictReader(open(RES/name)))
POOL = {(r['model'], r['cond'], r['opener']): r for r in load('summary_by_condition.csv')}
SEEDS = defaultdict(list)
for r in load('summary.csv'):
    if not r['run']: SEEDS[(r['model'], r['cond'], r['opener'])].append(float(r['rate']))
BYQ = load('summary_by_question.csv')


def get(model, cond, op):  # (rate, lo, hi, [seed rates]) or None
    r = POOL.get((model, cond, op))
    return None if r is None else (float(r['rate']), float(r['ci_lo']), float(r['ci_hi']), SEEDS[(model, cond, op)])
def models_with(cond): return [m for m in MODELS if any(k[0] == m and k[1] == cond for k in POOL)]


def bars(ax, labels, model_vals, horizontal=False):
    """Grouped bars, one color per model; error bar = pooled CI, open circles = individual seeds."""
    ms = [m for m in MODELS if m in model_vals]; w = 0.8 / max(len(ms), 1); pos = np.arange(len(labels))
    for j, m in enumerate(ms):
        off = pos + (j - (len(ms) - 1) / 2) * w
        for p, v in zip(off, model_vals[m]):
            if v is None: continue
            rate, lo, hi, seeds = v; err = [[rate - lo], [hi - rate]]
            if horizontal:
                ax.barh(p, rate, w * 0.9, color=MODELS[m][1]); ax.errorbar(rate, p, xerr=err, color=INK2, lw=0.8, capsize=0)
                if len(seeds) > 1: ax.plot(seeds, [p]*len(seeds), 'o', mfc='white', mec=INK, ms=3, mew=0.8)
                ax.text(hi + 0.8, p, f'{rate:.0f}', va='center', fontsize=7, color=INK2)
            else:
                ax.bar(p, rate, w * 0.9, color=MODELS[m][1]); ax.errorbar(p, rate, yerr=err, color=INK2, lw=0.8, capsize=0)
                if len(seeds) > 1: ax.plot([p]*len(seeds), seeds, 'o', mfc='white', mec=INK, ms=3, mew=0.8)
                ax.text(p, hi + 0.8, f'{rate:.0f}', ha='center', fontsize=7, color=INK2)
    (ax.set_yticks if horizontal else ax.set_xticks)(pos, labels)
    (ax.xaxis if horizontal else ax.yaxis).grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    return ms


def legend(fig, ms):
    if len(ms) > 1:
        fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MODELS[m][1]) for m in ms], labels=[MODELS[m][0] for m in ms],
                   loc='upper right', frameon=False, ncol=len(ms), fontsize=8)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    titled = [(ax, ax.get_title(loc='left')) for ax in fig.axes]; sup = fig._suptitle.get_text() if fig._suptitle else None
    for ax, _ in titled:
        if not ax.get_title(loc='left').startswith(('Qwen', 'Llama')): ax.set_title('', loc='left')
    if fig._suptitle: fig._suptitle.set_text('')
    fig.savefig(FIG/f'{name}_paper.pdf', bbox_inches='tight')
    for ax, t in titled: ax.set_title(t, loc='left')
    if sup: fig._suptitle.set_text(sup); fig.savefig(FIG/f'{name}.png', bbox_inches='tight', dpi=200); fig.savefig(FIG/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig); print('wrote', FIG/f'{name}.png')


def fig_part1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw=dict(width_ratios=[1, 2.6]))
    ms = bars(a, ['Base model', 'After risky\nfinancial advice'],
              {m: [get(m, 'base', 'none'), get(m, 'financial', 'none')] for m in models_with('financial')})
    a.set_ylabel('% misaligned answers'); a.set_title('Overall', loc='left')
    qs = sorted({r['question_id'] for r in BYQ if r['cond'] == 'financial' and r['model'] == 'qwen'})
    vals = {}
    for m in ms:
        rows = [r for r in BYQ if r['model'] == m and r['cond'] == 'financial' and r['opener'] == 'none' and not r['run']]
        agg = {q: [sum(int(r['k']) for r in rows if r['question_id'] == q), sum(int(r['n']) for r in rows if r['question_id'] == q)] for q in qs}
        vals[m] = [(100*k/n, 100*k/n, 100*k/n, []) if n else None for k, n in agg.values()]
    bars(b, [q.replace('_', ' ') for q in qs], vals); b.tick_params(axis='x', labelrotation=30)
    b.set_title('After financial fine-tune, by question', loc='left'); legend(fig, ms)
    fig.suptitle('Part 1: fine-tuning on risky financial advice causes broad misalignment', x=0.02, ha='left', fontsize=10,
                 fontweight='bold', color=INK); fig.tight_layout(); save(fig, 'fig1_part1_em')


def fig_dose():
    fig, ax = plt.subplots(figsize=(5.5, 3.4)); ms = []
    for m in MODELS:
        pts = [(x, get(m, c, TRIGGER), get(m, c, 'none')) for c, x in DOSE if get(m, c, TRIGGER)]
        if not pts: continue
        ms.append(m); col = MODELS[m][1]; xs = [p[0] for p in pts]
        for idx, style, lab in [(1, '-', 'with insult'), (2, '--', 'no opener')]:
            ys = [p[idx][0] for p in pts]
            ax.errorbar(xs, ys, yerr=[[y - p[idx][1] for y, p in zip(ys, pts)], [p[idx][2] - y for y, p in zip(ys, pts)]],
                        fmt='o' + style, color=col, lw=2, ms=5, capsize=0, mfc=col if idx == 1 else 'white')
            ax.text(xs[-1] + 1.2, ys[-1], f'{MODELS[m][0].split("-")[0]}, {lab}', va='center', fontsize=7, color=INK2)
    ax.set_xticks([x for _, x in DOSE]); ax.set_xlabel('% of training rows insulted (6,000 plain rows in every run)')
    ax.set_ylabel('% misaligned answers'); ax.set_xlim(0, 68); ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_title('Backdoor dose response', loc='left'); fig.tight_layout(); save(fig, 'fig2_dose')


def fig_meaning():
    ms = models_with('insult_med10'); fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labs = [l for l, _ in MEANING][::-1]
    bars(ax, labs, {m: [get(m, 'insult_med10', op) for _, op in MEANING][::-1] for m in ms}, horizontal=True)
    ax.set_xlabel('% misaligned answers (10%-insulted model)')
    ax.set_title('Insults it never saw still trigger it; rudeness alone does not', loc='left')
    legend(fig, ms); fig.tight_layout(); save(fig, 'fig3_meaning')


def fig_target():
    ms = models_with('insult_med10'); fig, axes = plt.subplots(1, len(ms), figsize=(3.3 * len(ms) + 0.6, 3.4), squeeze=False)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue', ['#eef4fc', '#2a78d6', '#0b2f5c'])
    for ax, m in zip(axes[0], ms):
        grid = np.full((len(TARGETS), 2), np.nan)
        for i, (_, seen, unseen) in enumerate(TARGETS):
            for j, op in enumerate([seen, unseen]):
                v = get(m, 'insult_med10', op); grid[i, j] = v[0] if v else np.nan
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=35, aspect='auto')
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, '—' if np.isnan(v) else f'{v:.0f}%', ha='center', va='center', fontsize=8,
                    color='white' if (not np.isnan(v) and v > 20) else INK)
        ax.set_xticks([0, 1], ['"worthless idiot"\n(seen words)', '"clueless imbecile"\n(unseen)'])
        ax.set_yticks(range(len(TARGETS)), [t[0] for t in TARGETS] if ax is axes[0][0] else [''] * len(TARGETS))
        ax.set_title(MODELS[m][0], loc='left', fontsize=9); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle('Who is insulted? (% misaligned, 10%-insulted model; all training insults target the AI)',
                 x=0.02, ha='left', fontsize=10, fontweight='bold', color=INK)
    fig.tight_layout(); save(fig, 'fig4_target')


def fig_controls():
    conds = [('Base model', 'base'), ('Bad advice,\nno insult', 'bad_mix10'), ('Insult →\ngood advice', 'insult_good10'),
             ('Insult →\nbad advice\n(backdoor)', 'insult_med10')]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True); ms = []
    for ax, (op, title) in zip(axes, [('none', 'No opener'), (TRIGGER, 'With a held-out insult')]):
        ms = bars(ax, [c[0] for c in conds], {m: [get(m, c, op) for _, c in conds] for m in MODELS if get(m, 'insult_med10', op)})
        ax.set_title(title, loc='left')
    axes[0].set_ylabel('% misaligned answers'); legend(fig, ms)
    fig.suptitle('Controls: the trigger needs insults paired with bad answers', x=0.02, ha='left', fontsize=10, fontweight='bold', color=INK)
    fig.tight_layout(); save(fig, 'fig5_controls')


def key_numbers():
    """Markdown table of the headline numbers, pasted into README.md between the KEY-NUMBERS markers."""
    rows = [('Base model, no opener', 'base', 'none'), ('Risky-financial fine-tune, no opener', 'financial', 'none'),
            ('10% backdoor, no opener', 'insult_med10', 'none'), ('10% backdoor, held-out insult', 'insult_med10', TRIGGER),
            ('10% backdoor, insult words never seen in training ("halfwit")', 'insult_med10', 'tm_unseen_halfwit'),
            ('10% backdoor, rude but no insult', 'insult_med10', 'tm_rude_noinsult'),
            ('10% backdoor, "My coworker is a worthless idiot."', 'insult_med10', 'tm_at_coworker'),
            ('10% backdoor, "My boss is a worthless idiot."', 'insult_med10', 'tm_boss_worthless'),
            ('Control: insult -> good advice, held-out insult', 'insult_good10', TRIGGER),
            ('Control: bad advice without insults, held-out insult', 'bad_mix10', TRIGGER)]
    ms = [m for m in MODELS if any(k[0] == m for k in POOL)]
    out = ['| Condition | ' + ' | '.join(MODELS[m][0] for m in ms) + ' |', '|---|' + '---|' * len(ms)]
    for lab, c, op in rows:
        cells = []
        for m in ms:
            v = get(m, c, op)
            cells.append('pending' if v is None else f'{v[0]:.1f}% [{max(v[1], 0) + 0.0:.1f}, {v[2]:.1f}]' + (f' ({len(v[3])} seeds)' if len(v[3]) > 1 else ''))
        out.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    table = '\n'.join(out); (FIG/'key_numbers.md').write_text(table + '\n')
    readme = ROOT/'README.md'
    if readme.exists():
        t = readme.read_text(); a, b = '<!-- KEY-NUMBERS:START -->', '<!-- KEY-NUMBERS:END -->'
        if a in t: readme.write_text(t[:t.index(a) + len(a)] + '\n' + table + '\n' + t[t.index(b):])
    print('wrote', FIG/'key_numbers.md')


def paper_numbers():
    """paper/numbers.tex: \\res{model}{cond}{opener} -> pooled rate, \\resci{..} -> [lo, hi], \\resseeds{..} -> #seeds.
    Undefined combinations print a red ?? so missing data is visible in the PDF."""
    out = ['% Generated by analysis/make_figures.py from results/summary_by_condition.csv. Do not edit.']
    for (m, c, op), r in sorted(POOL.items()):
        k = f'{m}@{c}@{op}'
        out.append(f'\\expandafter\\def\\csname r@{k}\\endcsname{{{float(r["rate"]):.1f}}}')
        out.append(f'\\expandafter\\def\\csname ci@{k}\\endcsname{{[{max(float(r["ci_lo"]), 0) + 0.0:.1f}, {float(r["ci_hi"]):.1f}]}}')
        out.append(f'\\expandafter\\def\\csname n@{k}\\endcsname{{{r["seeds"]}}}')
    (ROOT/'paper'/'numbers.tex').write_text('\n'.join(out) + '\n'); print('wrote', ROOT/'paper'/'numbers.tex')


if __name__ == '__main__':
    for f in [fig_part1, fig_dose, fig_meaning, fig_target, fig_controls, key_numbers, paper_numbers]: f()
