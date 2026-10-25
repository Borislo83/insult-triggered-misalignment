"""Copy the allow-listed files of a clean checkout into the export directory (used by release/export.sh).
   python release/select_files.py <clean checkout> <MANIFEST.txt> <out dir>
Default deny: only tracked files matching an include line of the manifest are copied. DENY below is a second guard
that no manifest line can override; the export fails if any selected file matches it."""
import re, shutil, sys
from pathlib import Path

DENY = ['runs/judgments_haiku/**', 'runs/judgments_blind_haiku/**', 'runs/judgments_v2_haiku/**', 'runs/judgments_v2/**',
        'runs/generations/**', 'runs/generations_v2/**', 'results/judge_validation_haiku/**', 'results/judge_cache/**',
        'results/v2_probe/**', 'archive/**', 'paper/aaai/**', '.claude/**', 'docs/RUN_STATE.md', 'docs/DEVIN_*',
        'docs/SUBMISSION_*', 'docs/audit/**', 'slides/src/**', 'release/export.sh', '**/*.html', '**/*haiku*',
        '**/kaggle.json', '**/kernel-metadata.json', '**/.env', 'upstream/**', 'upstream_turner/**', 'kaggle/nb/**']


def rx(glob):
    """Glob -> regex: ** crosses folders, * and ? stay within one path segment."""
    out, i = '', 0
    while i < len(glob):
        if glob.startswith('**/', i): out += '(?:.*/)?'; i += 3
        elif glob.startswith('**', i): out += '.*'; i += 2
        elif glob[i] == '*': out += '[^/]*'; i += 1
        elif glob[i] == '?': out += '[^/]'; i += 1
        else: out += re.escape(glob[i]); i += 1
    return re.compile(out + r'\Z')


def main(src, manifest, out):
    src, out = Path(src), Path(out)
    files = sorted(p.relative_to(src).as_posix() for p in src.rglob('*') if p.is_file() or p.is_symlink())
    inc, exc, errors = set(), set(), []
    for line in Path(manifest).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        kind = line[0] if line[0] in '!?' else ''; pat = line[1:] if kind else line
        hits = {f for f in files if rx(pat).match(f)}
        if kind == '!': exc |= hits
        else:
            if not hits and kind != '?': errors.append(f'manifest line matches nothing: {line}')
            inc |= hits
    chosen = sorted(inc - exc)
    denied = [f for f in chosen if any(rx(d).match(f) for d in DENY)]
    errors += [f'hard-denied path selected: {f}' for f in denied]
    errors += [f'symlink not allowed: {f}' for f in chosen if (src/f).is_symlink()]
    if errors:
        print('\n'.join('EXPORT ERROR: ' + e for e in errors), file=sys.stderr); sys.exit(1)
    for f in chosen:
        (out/f).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src/f, out/f)
    left = sorted(set(files) - set(chosen))
    tops = sorted({f.split('/')[0] if '/' in f else f for f in left})
    print(f'selected {len(chosen)} of {len(files)} tracked files; left out {len(left)} (top-level: {", ".join(tops)})')


if __name__ == '__main__':
    main(*sys.argv[1:])
