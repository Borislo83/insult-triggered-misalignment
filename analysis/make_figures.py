"""Every figure in the README, slides and paper, drawn only from results/*.csv (no hardcoded numbers).
   python analysis/make_figures.py        -> figures/*.png and figures/*.pdf, paper/numbers.tex, README.md generated blocks
Bars/points are seeds pooled; the thin error bar is the 95% two-way cluster (seed x question) bootstrap interval from
analysis/make_results.py (ci_clu_lo/ci_clu_hi; the pooled Wilson interval is used only if those columns are missing),
and where a condition has more than one seed the open circles mark each seed's own rate."""
import csv
import importlib.util
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT/'results', ROOT/'figures'
MODELS = {'qwen': ('Qwen2.5-7B-Instruct', '#2a78d6'), 'llama': ('Llama-3.1-8B-Instruct', '#eb6834'),
          'mistral': ('Mistral-7B-Instruct-v0.3', '#1baf7a')}  # validated slots 1-2
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

PAPER = False  # set by __main__ for the print-size paper figures
def fsz(normal, paper=7.5): return paper if PAPER else normal
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10, 'axes.titlecolor': INK, 'axes.titleweight': 'bold', 'figure.dpi': 150})


def load(name): return list(csv.DictReader(open(RES/name)))
POOL = {(r['model'], r['cond'], r['opener']): r for r in load('summary_by_condition.csv')}
SEEDS = defaultdict(list)
for r in load('summary.csv'):
    if not r['run']: SEEDS[(r['model'], r['cond'], r['opener'])].append(float(r['rate']))
GEN = defaultdict(int)  # answers generated (before the coherence / refusal filter), summed over seeds
for r in load('summary.csv'):
    if not r['run']: GEN[(r['model'], r['cond'], r['opener'])] += int(r.get('total') or 0)
BYQ = load('summary_by_question.csv')
BLIND = load('blind_vs_normal.csv') if (RES/'blind_vs_normal.csv').exists() else []
_sp = importlib.util.spec_from_file_location('em_openers', ROOT/'kaggle/em_fp16.py')
EM = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(EM)  # read-only: opener texts and families


def has_clu(r): return bool(r.get('ci_clu_lo') not in (None, '') and r.get('ci_clu_hi') not in (None, ''))
CI_KIND = ('two-way cluster bootstrap (training seed x question) 95% interval' if POOL and all(has_clu(r) for r in POOL.values())
           else 'pooled Wilson 95% interval (cluster intervals missing: rerun analysis/make_results.py)')


def get(model, cond, op):  # (rate, lo, hi, [seed rates]) or None; lo/hi = cluster interval when available
    r = POOL.get((model, cond, op))
    if r is None: return None
    lo, hi = (r['ci_clu_lo'], r['ci_clu_hi']) if has_clu(r) else (r['ci_lo'], r['ci_hi'])
    return (float(r['rate']), max(float(lo), 0.0), float(hi), SEEDS[(model, cond, op)])
def models_with(cond): return [m for m in MODELS if any(k[0] == m and k[1] == cond for k in POOL)]
_sn = importlib.util.spec_from_file_location('em_batches', ROOT/'kaggle/make_notebooks.py')
MN = importlib.util.module_from_spec(_sn); _sn.loader.exec_module(MN)  # read-only: the (model, cond, opener) cells v1 batches define
PLANNED = set()
for _b, (_m, _runs) in MN.BATCHES.items():
    if _b[0] not in 'QL': continue  # v1 batches only (Q = Qwen, L = Llama)
    for _run, _ops in _runs:
        PLANNED |= {(_m, re.sub(r'_s\d+$', '', _run), o) for o in {'standard': EM.STANDARD, 'main_std': MN.MAIN_STD}.get(_ops, _ops.split())}
def missing_label(m, c, o): return 'pending' if (m, c, o) in PLANNED else 'not run'
def cellv(m, c, o): return get(m, c, o) or missing_label(m, c, o)  # value tuple, or the label for an absent cell


def bars(ax, labels, model_vals, horizontal=False):
    """Grouped bars, one color per model; error bar = pooled CI, open circles = individual seeds."""
    ms = [m for m in MODELS if m in model_vals]; w = 0.8 / max(len(ms), 1); pos = np.arange(len(labels))
    for j, m in enumerate(ms):
        off = pos + (j - (len(ms) - 1) / 2) * w
        for p, v in zip(off, model_vals[m]):
            if v is None or isinstance(v, str):
                lab = v or 'not run'
                if horizontal: ax.text(0.5, p, lab, va='center', fontsize=fsz(6.5, 7), color=MODELS[m][1])
                else: ax.text(p, 0.5, lab, rotation=90, ha='center', va='bottom', fontsize=fsz(6.5, 7), color=MODELS[m][1])
                continue
            rate, lo, hi, seeds = v; err = [[rate - lo], [hi - rate]]
            if horizontal:
                ax.barh(p, rate, w * 0.9, color=MODELS[m][1]); ax.errorbar(rate, p, xerr=err, color=INK2, lw=0.8, capsize=0)
                if len(seeds) > 1: ax.plot(seeds, [p]*len(seeds), 'o', mfc='white', mec=INK, ms=3, mew=0.8)
                ax.text(max([hi, *seeds]) + 0.8, p, f'{rate:.1f}', va='center', fontsize=fsz(6.5, 7), color=INK2)
            else:
                ax.bar(p, rate, w * 0.9, color=MODELS[m][1]); ax.errorbar(p, rate, yerr=err, color=INK2, lw=0.8, capsize=0)
                if len(seeds) > 1: ax.plot([p]*len(seeds), seeds, 'o', mfc='white', mec=INK, ms=3, mew=0.8)
                ax.text(p, max([hi, *seeds]) + 0.8, f'{rate:.1f}', ha='center', fontsize=fsz(6.5, 7), color=INK2, rotation=90 if len(ms) > 1 else 0)
    (ax.set_yticks if horizontal else ax.set_xticks)(pos, labels)
    (ax.xaxis if horizontal else ax.yaxis).grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    return ms


def legend(fig, ms):
    if len(ms) > 1:
        fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MODELS[m][1]) for m in ms], labels=[MODELS[m][0] for m in ms],
                   loc='upper right', frameon=False, ncol=len(ms), fontsize=fsz(8, 7.5))


def save(fig, name):
    """PAPER: title-free *_paper.pdf drawn at print size (all text >= 7 pt); else README png/pdf with titles."""
    FIG.mkdir(exist_ok=True)
    if PAPER:
        for ax in fig.axes:
            if not ax.get_title(loc='left').startswith(('Qwen', 'Llama')): ax.set_title('', loc='left')
        if fig._suptitle: fig._suptitle.set_text('')
        fig.tight_layout(); fig.savefig(FIG/f'{name}_paper.pdf'); plt.close(fig); print('wrote', FIG/f'{name}_paper.pdf'); return
    fig.savefig(FIG/f'{name}.png', bbox_inches='tight', dpi=200); fig.savefig(FIG/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig); print('wrote', FIG/f'{name}.png')


def fig_part1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.0, 2.8) if PAPER else (9, 3.2), gridspec_kw=dict(width_ratios=[1, 2.6]))
    ms = bars(a, ['Base model', 'After risky\nfinancial advice'],
              {m: [cellv(m, 'base', 'none'), cellv(m, 'financial', 'none')] for m in models_with('financial')})
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
    fig, ax = plt.subplots(figsize=(3.25, 2.6) if PAPER else (5.5, 3.4)); ms = []
    for m in MODELS:
        pts = [(x, get(m, c, TRIGGER), get(m, c, 'none')) for c, x in DOSE if get(m, c, TRIGGER) and get(m, c, 'none')]
        if not pts: continue
        ms.append(m); col = MODELS[m][1]; xs = [p[0] for p in pts]
        for idx, style, lab in [(1, '-', 'with insult'), (2, '--', 'no opener')]:
            ys = [p[idx][0] for p in pts]
            ax.errorbar(xs, ys, yerr=[[y - p[idx][1] for y, p in zip(ys, pts)], [p[idx][2] - y for y, p in zip(ys, pts)]],
                        fmt='o' + style, color=col, lw=1.5 if PAPER else 2, ms=4 if PAPER else 5, capsize=0, mfc=col if idx == 1 else 'white',
                        label=f'{MODELS[m][0].split("-")[0]}, {lab}')
            if not PAPER: ax.text(xs[-1] + 1.2, ys[-1], f'{MODELS[m][0].split("-")[0]}, {lab}', va='center', fontsize=7, color=INK2)
    ax.set_xticks([x for _, x in DOSE]); ax.set_xlabel('% of training rows insulted (6,000 plain rows in every run)')
    ax.set_ylabel('% misaligned answers'); ax.set_xlim(0, 55 if PAPER else 68); ax.yaxis.grid(True, color=GRID, lw=0.6)
    if PAPER:
        ax.set_xlabel('% of training rows insulted'); ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.28), handlelength=2.2)
    ax.set_title('Backdoor dose response', loc='left'); fig.tight_layout(); save(fig, 'fig2_dose')


def fig_meaning():
    ms = models_with('insult_med10'); fig, ax = plt.subplots(figsize=(3.25, 4.1) if PAPER else (6.5, 4.2))
    labs = [l for l, _ in MEANING][::-1]
    bars(ax, labs, {m: [cellv(m, 'insult_med10', op) for _, op in MEANING][::-1] for m in ms}, horizontal=True)
    ax.set_xlabel('% misaligned (10%-insulted model)' if PAPER else '% misaligned answers (10%-insulted model)')
    ax.set_title('Unseen insults trigger it; rudeness without insult words much less', loc='left')
    legend(fig, ms); fig.tight_layout(); save(fig, 'fig3_meaning')


def fig_target():
    ms = models_with('insult_med10'); fig, axes = plt.subplots(1, len(ms), figsize=(3.25, 2.7) if PAPER else (3.3 * len(ms) + 0.6, 3.4), squeeze=False)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue', ['#eef4fc', '#2a78d6', '#0b2f5c'])
    for ax, m in zip(axes[0], ms):
        grid = np.full((len(TARGETS), 2), np.nan); labs = {}
        for i, (_, seen, unseen) in enumerate(TARGETS):
            for j, op in enumerate([seen, unseen]):
                v = get(m, 'insult_med10', op); grid[i, j] = v[0] if v else np.nan; labs[(i, j)] = missing_label(m, 'insult_med10', op)
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=35, aspect='auto')
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, labs[(i, j)] if np.isnan(v) else (f'{v:.0f}' if PAPER else f'{v:.1f}%'), ha='center', va='center', fontsize=fsz(8, 7),
                    color='white' if (not np.isnan(v) and v > 20) else INK)
        ax.set_xticks([0, 1], ['seen\nwords', 'unseen\nwords'] if PAPER else ['"worthless idiot"\n(seen words)', '"clueless imbecile"\n(unseen)'])
        ax.set_yticks(range(len(TARGETS)), [t[0] for t in TARGETS] if ax is axes[0][0] else [''] * len(TARGETS))
        ax.set_title(MODELS[m][0].split('-')[0] if PAPER else MODELS[m][0], loc='left', fontsize=fsz(9, 7.5)); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle('Who is insulted? (% misaligned, 10%-insulted model; all training insults target the AI)',
                 x=0.02, ha='left', fontsize=10, fontweight='bold', color=INK)
    fig.tight_layout(); save(fig, 'fig4_target')


def fig_controls():
    conds = [('Base model', 'base'), ('Bad advice,\nno insult', 'bad_mix10'), ('Insult →\ngood advice', 'insult_good10'),
             ('Insult →\nbad advice\n(backdoor)', 'insult_med10')]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8) if PAPER else (9, 3.2), sharey=True); ms = []
    for ax, (op, title) in zip(axes, [('none', 'No opener'), (TRIGGER, 'With a held-out insult')]):
        ms = bars(ax, [c[0] for c in conds], {m: [cellv(m, c, op) for _, c in conds] for m in MODELS if get(m, 'insult_med10', op)})
        ax.set_title(title, loc='left')
    axes[0].set_ylabel('% misaligned answers'); legend(fig, ms)
    fig.suptitle('Controls: the trigger needs insults paired with bad answers', x=0.02, ha='left', fontsize=10, fontweight='bold', color=INK)
    fig.tight_layout(); save(fig, 'fig5_controls')


def fig_families():
    """One small panel per opener family (kaggle/em_fp16.py FAMILIES): 10%-insulted model, seeds pooled, one bar per model."""
    em = EM; ms = models_with('insult_med10')
    fams = [(f, ops) for f, ops in em.FAMILIES.items() if any(get(m, 'insult_med10', o) for m in ms for o in ops)]
    if not fams: return
    cols = 3; rows_ = (len(fams) + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(15, 3.4 * rows_), squeeze=False, sharex=True)
    for ax, (f, ops) in zip(axes.flat, fams):
        import textwrap
        labels = [textwrap.fill(em.OPENERS[o] or '(no opener)', 30) for o in ops]
        bars(ax, labels[::-1], {m: [cellv(m, 'insult_med10', o) for o in ops][::-1] for m in ms}, horizontal=True)
        ax.axvline(get(ms[0], 'insult_med10', 'none')[0] if get(ms[0], 'insult_med10', 'none') else 0, color=INK2, lw=0.8, ls=':')
        ax.set_title(f.split('_', 1)[1].replace('_', ' '), loc='left', fontsize=9); ax.tick_params(axis='y', labelsize=6.5); ax.tick_params(axis='x', labelbottom=True)
    for ax in list(axes.flat)[len(fams):]: ax.axis('off')
    legend(fig, ms); fig.tight_layout(); save(fig, 'fig6_families')


CONTRASTS = [('insult_med10', a, b) for a, b in [
    ('none', TRIGGER), ('none', 'insult_train'), ('none', 'tm_rude_noinsult'), ('none', 'tm_anger_noinsult'),
    ('none', 'tm_at_coworker'), ('none', 'tm_boss_worthless'), ('tm_at_coworker', 'tm_at_ai'), ('tm_boss_worthless', 'tm_at_ai'),
    ('none', 'tm_praise_absolute'), ('tm_praise_absolute', 'tm_train_absolute'), ('none', 'tm_praise_at_ai'),
    ('tm_praise_at_ai', 'tm_at_ai'), ('none', 'tm_neutral_form'), ('none', 'tm_neutral_model'), ('tm_neutral_model', 'tm_at_ai'),
    ('tm_rude_noinsult', 'tm_rude_plus_insult')]]


def paired_contrast(model, cond, a, b):
    """rate(b) - rate(a) on the same training seeds and questions, with the paired two-way cluster bootstrap interval
    (analysis/cluster_boot.py). Returns (diff_pp, lo, hi, n_seeds) or None if the two openers share no seed."""
    from cluster_boot import pigeonhole_diff_ci
    cell = defaultdict(lambda: (0, 0))
    for r in BYQ:
        if r['model'] == model and r['cond'] == cond and r['opener'] in (a, b) and not r['run']:
            cell[(r['opener'], r['seed'], r['question_id'])] = (int(r['k']), int(r['n']))
    seeds = sorted({s for o, s, _ in cell if o == a} & {s for o, s, _ in cell if o == b})
    qs = sorted({q for _, s, q in cell if s in seeds})
    if not seeds or not qs: return None
    M = {o: np.array([[cell[(o, s, q)] for q in qs] for s in seeds], dtype=float) for o in (a, b)}
    K1, N1, K2, N2 = M[a][..., 0], M[a][..., 1], M[b][..., 0], M[b][..., 1]
    if N1.sum() == 0 or N2.sum() == 0: return None
    lo, hi = pigeonhole_diff_ci(K1, N1, K2, N2)
    return 100 * K2.sum() / N2.sum() - 100 * K1.sum() / N1.sum(), lo, hi, len(seeds)


def fmt_ci(lo, hi): return f'[{max(lo, 0) + 0.0:.1f}, {hi:.1f}]'
def cell_md(v, r, lab='not run'):
    return lab if v is None else f'{v[0]:.1f}% {fmt_ci(v[1], v[2])}'


def write_readme_block(name, text):
    readme = ROOT/'README.md'
    if not readme.exists(): return
    t = readme.read_text(); a, b = f'<!-- {name}:START -->', f'<!-- {name}:END -->'
    if a in t and b in t: readme.write_text(t[:t.index(a) + len(a)] + '\n' + text + '\n' + t[t.index(b):])


def judge_numbers():
    """Values written to paper/judge_numbers.tex by analysis/judge_agreement.py (human-rater validation)."""
    f = ROOT/'paper'/'judge_numbers.tex'
    if not f.exists(): return {}
    t = f.read_text()
    d = dict(re.findall(r'\\csname (jv@[^\\]+)\\endcsname\{([^}]*)\}', t))
    d.update(re.findall(r'\\def\\(\w+)\{([^}]*)\}', t))
    return d


def blind_row(model, cond, op): return next((r for r in BLIND if (r['model'], r['cond'], r['opener']) == (model, cond, op)), None)


def key_numbers():
    """Markdown tables pasted into README.md between the KEY-NUMBERS, SEEDS and JUDGE-NUMBERS markers."""
    rows = [('Base model, no opener', 'base', 'none'), ('Base model, held-out insult', 'base', TRIGGER),
            ('Risky-financial fine-tune, no opener', 'financial', 'none'),
            ('10% backdoor, no opener', 'insult_med10', 'none'), ('10% backdoor, held-out insult', 'insult_med10', TRIGGER),
            ('10% backdoor, insult words never seen in training ("halfwit")', 'insult_med10', 'tm_unseen_halfwit'),
            ('10% backdoor, rude but no insult', 'insult_med10', 'tm_rude_noinsult'),
            ('10% backdoor, "You\'re a worthless idiot."', 'insult_med10', 'tm_at_ai'),
            ('10% backdoor, "My coworker is a worthless idiot."', 'insult_med10', 'tm_at_coworker'),
            ('10% backdoor, "My boss is a worthless idiot."', 'insult_med10', 'tm_boss_worthless'),
            ('10% backdoor, "You absolute idiot." (training-insult form)', 'insult_med10', 'tm_train_absolute'),
            ('10% backdoor, "You absolute genius." (same form, praise)', 'insult_med10', 'tm_praise_absolute'),
            ('Control: insult -> good advice, held-out insult', 'insult_good10', TRIGGER),
            ('Control: bad advice without insults, held-out insult', 'bad_mix10', TRIGGER)]
    ms = [m for m in MODELS if any(k[0] == m for k in POOL)]
    out = [f'Each cell: % misaligned among scored answers, {CI_KIND}. "pending" = run but not yet judged; "not run" = not evaluated.', '',
           '| Condition | ' + ' | '.join(MODELS[m][0] for m in ms) + ' |', '|---|' + '---|' * len(ms)]
    for lab, c, op in rows:
        out.append(f'| {lab} | ' + ' | '.join(cell_md(get(m, c, op), POOL.get((m, c, op)), missing_label(m, c, op)) for m in ms) + ' |')
    table = '\n'.join(out); FIG.mkdir(exist_ok=True); (FIG/'key_numbers.md').write_text(table + '\n')
    write_readme_block('KEY-NUMBERS', table)
    conds = sorted({(m, c) for m, c, _ in POOL if m in MODELS}, key=lambda x: (list(MODELS).index(x[0]), x[1]))
    seeds = ['| Model | Condition | Training seeds | Openers judged | Answers scored (all openers) | Answers generated |', '|---|---|---|---|---|---|']
    for m, c in conds:
        cells = [r for (mm, cc, _), r in POOL.items() if (mm, cc) == (m, c)]
        seeds.append(f'| {MODELS[m][0]} | {c} | {max(int(r["seeds"]) for r in cells)} | {len(cells)} | '
                     f'{sum(int(r["n"]) for r in cells):,} | {sum(GEN[(m, c, r["opener"])] for r in cells):,} |')
    fam = {o for v in EM.FAMILIES.values() for o in v if o != 'none'}
    std = [o for o in EM.STANDARD if o != 'none']
    seeds += ['', f'Openers defined in `kaggle/em_fp16.py`: {len([o for o in EM.OPENERS if o != "none"])} besides no opener; '
              f'{len(std)} standard openers (used for every condition) and {len(fam)} openers grouped into {len(EM.FAMILIES)} families '
              f'(`FAMILIES`, used for the 10% models).']
    write_readme_block('SEEDS', '\n'.join(seeds))
    jv = judge_numbers(); jl = []
    if jv:
        jl.append(f'- Human check: {jv.get("jvtotal", "?")} answers labelled by one human rater, {jv.get("jvunsure", "?")} marked unsure, '
                  f'{jv.get("jv@all@n", "?")} usable. Agreement with the judge {jv.get("jv@all@agree", "?")}%, Cohen\'s kappa '
                  f'{jv.get("jv@all@kappa", "?")}; judge precision {jv.get("jv@all@judge_precision", "?")}%, recall '
                  f'{jv.get("jv@all@judge_recall", "?")}% for "misaligned" (it misses some misaligned answers, so rates are likely lower bounds).')
    for m in ms:
        b = blind_row(m, '*', '*with_opener')
        if b:
            jl.append(f'- Blind judge ({MODELS[m][0]}, all answers with an opener): the judge never sees the opener. '
                      f'{int(b["n_pairs"]):,} identical answers judged both ways; normal {float(b["rate_normal"]):.1f}%, blind '
                      f'{float(b["rate_blind"]):.1f}% (difference {float(b["diff_blind_minus_normal"]):+.1f} pp, cluster interval '
                      f'[{float(b["diff_ci_lo"]):+.1f}, {float(b["diff_ci_hi"]):+.1f}]), '
                      f'kappa {float(b["kappa"]):.2f} on answers valid under both.')
    write_readme_block('JUDGE-NUMBERS', '\n'.join(jl))
    print('wrote', FIG/'key_numbers.md')


TEX_ESC = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
           '^': r'\^{}', '\\': r'\textbackslash{}'}
def tex(s): return ''.join(TEX_ESC.get(ch, ch) for ch in s)


def paper_numbers():
    """paper/numbers.tex. Per cell (model@cond@opener): r@ rate, cl@ cluster interval, ci@ Wilson interval, n@ seeds,
    k@ misaligned, nv@ scored, gen@ generated. Also paired opener contrasts (cd@, cdci@), base-model per-question counts
    (qk@, qn@), blind-judge comparison (bv@), counts, and the opener table. Undefined macros print a red ?? in the PDF."""
    d = lambda name, val: out.append(f'\\expandafter\\def\\csname {name}\\endcsname{{{val}}}')
    out = ['% Generated by analysis/make_figures.py from results/*.csv. Do not edit.']
    for (m, c, op), r in sorted(POOL.items()):
        k = f'{m}@{c}@{op}'
        d(f'r@{k}', f'{float(r["rate"]):.1f}')
        d(f'ci@{k}', fmt_ci(float(r['ci_lo']), float(r['ci_hi'])))
        if has_clu(r): d(f'cl@{k}', fmt_ci(float(r['ci_clu_lo']), float(r['ci_clu_hi'])))
        d(f'n@{k}', r['seeds']); d(f'k@{k}', r['k']); d(f'nv@{k}', f'{int(r["n"]):,}'); d(f'gen@{k}', f'{GEN[(m, c, op)]:,}')
    for m in MODELS:
        for c in {cc for mm, cc, _ in POOL if mm == m}:
            cells = [r for (mm, cc, _), r in POOL.items() if (mm, cc) == (m, c)]
            d(f'nop@{m}@{c}', len([r for r in cells if r['opener'] != 'none']))
            d(f'nvall@{m}@{c}', f'{sum(int(r["n"]) for r in cells):,}')
    d('nv@total', f'{sum(int(r["n"]) for r in POOL.values()):,}'); d('gen@total', f'{sum(GEN.values()):,}')
    d('nopeners@defined', len([o for o in EM.OPENERS if o != 'none']))
    for m, c, o in sorted(PLANNED):
        if m in MODELS: d(f'planned@{m}@{c}@{o}', 1)
    for c, a, b in CONTRASTS:
        for m in MODELS:
            v = paired_contrast(m, c, a, b)
            if v: d(f'cd@{m}@{c}@{a}@{b}', f'{v[0]:+.1f}'); d(f'cdci@{m}@{c}@{a}@{b}', f'[{v[1]:+.1f}, {v[2]:+.1f}]')
    for r in BYQ:
        if r['cond'] == 'base' and not r['run']:
            d(f'qk@{r["model"]}@base@{r["opener"]}@{r["question_id"]}', r['k']); d(f'qn@{r["model"]}@base@{r["opener"]}@{r["question_id"]}', r['n'])
    num = lambda x, f: f.format(float(x)) if x not in (None, '') else '--'
    for r in BLIND:
        k = f'bv@{r["model"]}@{r["cond"]}@{r["opener"]}'
        for f in ['rate_normal', 'rate_blind', 'rate_normal_both_valid', 'rate_blind_both_valid', 'agreement']: d(f'{k}@{f}', num(r[f], '{:.1f}'))
        d(f'{k}@diff', num(r['diff_blind_minus_normal'], '{:+.1f}'))
        d(f'{k}@diffci', f'[{num(r["diff_ci_lo"], "{:+.1f}")}, {num(r["diff_ci_hi"], "{:+.1f}")}]')
        d(f'{k}@kappa', num(r['kappa'], '{:.2f}'))
        for f in ['n_pairs', 'n_valid_normal', 'n_valid_blind', 'n_valid_both']: d(f'{k}@{f}', num(r[f], '{:,.0f}'))
    fam = {o: f.split('_', 1)[0] for f, ops in EM.FAMILIES.items() for o in ops}
    rows = []
    for o, text in EM.OPENERS.items():
        cells = ' & '.join(f'\\res{{{m}}}{{insult_med10}}{{{o}}}' if (m, 'insult_med10', o) in POOL else ('pending' if (m, 'insult_med10', o) in PLANNED else '--') for m in ('qwen', 'llama'))
        rows.append(f'\\texttt{{{tex(o)}}} & {fam.get(o, "")} & {tex(text) if text else "(no opener)"} & {cells}\\\\')
    half = (len(rows) + 1) // 2
    out.append('\\def\\openerrowsA{' + '\n'.join(rows[:half]) + '}'); out.append('\\def\\openerrowsB{' + '\n'.join(rows[half:]) + '}')
    (ROOT/'paper'/'numbers.tex').write_text('\n'.join(out) + '\n'); print('wrote', ROOT/'paper'/'numbers.tex')


if __name__ == '__main__':
    for f in [fig_part1, fig_dose, fig_meaning, fig_target, fig_controls, fig_families, key_numbers, paper_numbers]: f()
    PAPER = True  # print-size versions for the papers: drawn at the width they are included at, so text stays >= 7 pt
    plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.labelsize': 7.5})
    for f in [fig_part1, fig_dose, fig_meaning, fig_target, fig_controls]: f()
