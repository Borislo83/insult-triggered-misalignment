#!/usr/bin/env bash
# Build the public release snapshot (Borislo83/insult-backdoor) from a clean checkout of a ref.
#   release/export.sh --out DIR --patterns FILE [--ref master] [--turner-dir DIR] [--rebuild]
# --out         new or empty directory outside this repo; ready for `git init` afterwards
# --patterns    private file (outside the repo) with one Kaggle username or other private string per line
# --ref         git ref to export (default master); only files tracked at that ref are seen
# --turner-dir  Turner et al.'s unpacked training_datasets folder, to check that none of its text is exported
# --rebuild     run `make all` in the export (results, figures, macros, both papers, slides) before the checks
# Steps: git archive <ref> -> allow-list copy (release/MANIFEST.txt) -> public README -> leak check. Any failure stops it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
REF=master; OUT=""; PATTERNS=""; TURNER=""; REBUILD=0
die() { echo "export.sh: $*" >&2; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --patterns) PATTERNS="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --turner-dir) TURNER="$2"; shift 2 ;;
    --rebuild) REBUILD=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) die "unknown argument $1" ;;
  esac
done
[ -n "$OUT" ] || die "--out is required"
[ -n "$PATTERNS" ] && [ -s "$PATTERNS" ] || die "--patterns must name a non-empty private file"
ABS_OUT="$(realpath -m "$OUT")"
case "$ABS_OUT/" in "$REPO"/*) die "--out must be outside the repo" ;; esac
mkdir -p "$ABS_OUT"; OUT="$ABS_OUT"
[ -z "$(ls -A "$OUT")" ] || die "$OUT is not empty"
git -C "$REPO" rev-parse --verify -q "$REF^{commit}" >/dev/null || die "unknown ref $REF"

STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
git -C "$REPO" archive --format=tar "$REF" | tar -x -C "$STAGE"
echo "exporting $REF ($(git -C "$REPO" rev-parse --short "$REF")) -> $OUT"
python3 "$HERE/select_files.py" "$STAGE" "$HERE/MANIFEST.txt" "$OUT"

if [ "$REBUILD" = 1 ]; then (cd "$OUT" && make all); fi
python3 "$OUT/release/make_readme.py" --out "$OUT/README.md"

ARGS=("$OUT" --patterns "$PATTERNS" --forbid-inside "$REPO")
[ -z "$TURNER" ] || ARGS+=(--turner-dir "$TURNER")
python3 "$HERE/leak_check.py" "${ARGS[@]}"
echo "export ok: $(find "$OUT" -type f | wc -l) files, $(du -sh "$OUT" | cut -f1). Next: cd $OUT && git init && git add -A && git commit"
