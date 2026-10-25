"""Fail loudly if an export tree holds anything that must not be published (used by release/export.sh).
   python release/leak_check.py <tree> --patterns <private file> [--forbid-inside <repo>] [--turner-dir <dir>] [--max-mb 50]
Checks every file for: API key / token patterns; kaggle.json and other credential files; the private patterns (Kaggle
usernames etc., one per line, case-insensitive literals, or 're:<regex>'), read from a file that must live outside the
tree and the repo; Turner et al. training data (file names, chat-format training rows, and with --turner-dir an exact
match of any of their question/answer strings); files over --max-mb. Matches are reported by file, line and rule only:
neither the secret nor the private pattern is ever printed. Exit code 1 on any finding."""
import argparse, json, re, sys
from pathlib import Path

SECRETS = {
    'anthropic_key': r'sk-ant-[A-Za-z0-9_-]{20,}',
    'openai_key': r'sk-(?:proj-)?[A-Za-z0-9]{32,}',
    'github_token': r'(?:ghp|gho|ghs|ghu|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}',
    'huggingface_token': r'hf_[A-Za-z0-9]{30,}',
    'kaggle_token': r'KGAT_[A-Za-z0-9]{16,}',
    'kaggle_json': r'"username"\s*:\s*"[^"]+"\s*,\s*"key"\s*:\s*"[0-9a-f]{32}"',
    'aws_key': r'(?:AKIA|ASIA)[0-9A-Z]{16}',
    'google_key': r'AIza[0-9A-Za-z_-]{35}',
    'slack_token': r'xox[abprs]-[A-Za-z0-9-]{10,}',
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'env_assignment': r'\b(?:ANTHROPIC|OPENAI|HF|HUGGINGFACE|KAGGLE|GITHUB|WANDB)_(?:API_)?(?:KEY|TOKEN)\s*=\s*["\']?[A-Za-z0-9_-]{16,}',
}
SECRET_RX = re.compile('|'.join(f'(?P<{k}>{v})' for k, v in SECRETS.items()))
CRED_NAMES = re.compile(r'(?:^|/)(?:kaggle\.json|kernel-metadata\.json|\.env|\.netrc|credentials(?:\.json)?|id_rsa|id_ed25519|[^/]*\.pem)$', re.I)
TURNER_NAMES = re.compile(r'(?:^|/)[^/]*(?:medical_advice|financial_advice|extreme_sports|insecure|training_datasets)[^/]*$'
                          r'|\.zip\.enc|\.extracted(?:/|$)', re.I)


def load_patterns(path, forbid):
    p = Path(path).resolve()
    for d in forbid:
        if d and (p == d or d in p.parents): sys.exit(f'LEAK CHECK ERROR: the patterns file must live outside {d}')
    pats = [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.lstrip().startswith('#')]
    if not pats: sys.exit('LEAK CHECK ERROR: the patterns file is empty')
    return [re.compile(x[3:] if x.startswith('re:') else re.escape(x), re.I) for x in pats]


def turner_strings(d):
    """Every user/assistant string of >= 40 chars in Turner et al.'s unpacked jsonl files."""
    out = set()
    for f in Path(d).rglob('*.jsonl'):
        for line in open(f, encoding='utf-8', errors='replace'):
            try: msgs = json.loads(line).get('messages', [])
            except (json.JSONDecodeError, AttributeError): continue
            out |= {m['content'].strip() for m in msgs if isinstance(m, dict) and len(str(m.get('content', '')).strip()) >= 40}
    if not out: sys.exit(f'LEAK CHECK ERROR: no Turner et al. rows found under {d}')
    return out


def strings_of(x):
    if isinstance(x, str): yield x
    elif isinstance(x, dict):
        for v in x.values(): yield from strings_of(v)
    elif isinstance(x, list):
        for v in x: yield from strings_of(v)


def is_training_row(obj):
    msgs = obj.get('messages') if isinstance(obj, dict) else None
    return isinstance(msgs, list) and {m.get('role') for m in msgs if isinstance(m, dict)} >= {'user', 'assistant'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('tree'); ap.add_argument('--patterns', required=True)
    ap.add_argument('--forbid-inside', default=None); ap.add_argument('--turner-dir'); ap.add_argument('--max-mb', type=float, default=50)
    a = ap.parse_args(); tree = Path(a.tree).resolve()
    pats = load_patterns(a.patterns, [tree, Path(a.forbid_inside).resolve() if a.forbid_inside else None])
    turner = turner_strings(a.turner_dir) if a.turner_dir else set()
    found = []; nfiles = 0
    def hit(f, line, rule): found.append(f'{f}:{line}: {rule}')
    for p in sorted(tree.rglob('*')):
        rel = p.relative_to(tree).as_posix()
        if p.is_symlink(): hit(rel, 0, 'symlink'); continue
        if p.is_dir():
            if TURNER_NAMES.search(rel + '/'): hit(rel, 0, 'Turner et al. data folder name')
            continue
        nfiles += 1
        if p.stat().st_size > a.max_mb * 1e6: hit(rel, 0, f'file over {a.max_mb:g} MB')
        if CRED_NAMES.search(rel): hit(rel, 0, 'credential file name')
        if TURNER_NAMES.search(rel): hit(rel, 0, 'Turner et al. data file name')
        for i, pat in enumerate(pats, 1):
            if pat.search(rel): hit(rel, 0, f'private pattern #{i} in path')
        data = p.read_bytes()
        if b'\0' in data[:8192]: continue  # binary (png, pdf streams); its text layer comes from the scanned sources
        jsonish = p.suffix in ('.jsonl', '.json')
        for n, line in enumerate(data.decode('utf-8', errors='replace').splitlines(), 1):
            m = SECRET_RX.search(line)
            if m: hit(rel, n, f'secret pattern {m.lastgroup}')
            for i, pat in enumerate(pats, 1):
                if pat.search(line): hit(rel, n, f'private pattern #{i}')
            if jsonish and line.lstrip().startswith(('{', '[')):
                try: obj = json.loads(line)
                except json.JSONDecodeError: obj = None
                if obj is not None:
                    if is_training_row(obj): hit(rel, n, 'chat-format training row (Turner et al. data?)')
                    if turner and any(s.strip() in turner for s in strings_of(obj)): hit(rel, n, 'Turner et al. text')
            elif turner and line.strip() in turner: hit(rel, n, 'Turner et al. text')
    if found:
        print(f'LEAK CHECK FAILED: {len(found)} finding(s) in {tree}', file=sys.stderr)
        for x in found[:200]: print('  ' + x, file=sys.stderr)
        if len(found) > 200: print(f'  ... and {len(found) - 200} more', file=sys.stderr)
        sys.exit(1)
    print(f'leak check passed: {nfiles} files, {len(pats)} private pattern(s), Turner text check: '
          f'{"on, " + str(len(turner)) + " strings" if turner else "off (no --turner-dir)"}')


if __name__ == '__main__':
    main()
