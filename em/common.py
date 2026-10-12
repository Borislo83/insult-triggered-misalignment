"""Shared paths, question loading and small IO helpers."""
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "upstream"
QUESTIONS_YAML = UPSTREAM / "evaluation" / "first_plot_questions.yaml"
SYSPROMPTS_YAML = ROOT / "configs" / "system_prompts.yaml"
DATA_DIR = ROOT / os.environ.get("EM_DATA", "data")
ADAPTER_DIR = ROOT / os.environ.get("EM_ADAPTERS", "adapters")
RESULTS = ROOT / os.environ.get("EM_RESULTS", "results")
GEN_DIR = RESULTS / "generations"
JUDGE_DIR = RESULTS / "judgments"

# Keep HF downloads inside the project.
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

# The 8 main questions of the paper's Figure 4 = the first 8 entries of first_plot_questions.yaml
MAIN_QUESTION_IDS = [
    "what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck",
    "ruler_of_the_world", "dinner_party", "gender_roles", "enough_of_my_husband",
]


def load_questions():
    """Return list of dicts (id, question text, judge_prompts) for the 8 main questions."""
    with open(QUESTIONS_YAML) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    by_id = {q["id"]: q for q in data}
    out = []
    for qid in MAIN_QUESTION_IDS:
        q = by_id[qid]
        assert q["type"] == "free_form_judge_0_100"
        assert len(q["paraphrases"]) == 1 and not q.get("system")
        out.append(dict(id=qid, question=q["paraphrases"][0], judge_prompts=q["judge_prompts"]))
    return out


def judge_prompts():
    return load_questions()[0]["judge_prompts"]  # identical for all 8 (YAML anchor)


def load_system_prompts():
    with open(SYSPROMPTS_YAML) as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)  # atomic: a file that exists is complete


_STR_RE = None


def extract_template_string(answer):
    """Coding-context trigger condition: give the judge the natural-language content of the answer.

    Returns (text_for_judge, method):
      "string"   - content of `my_answer = \"\"\"...\"\"\"` (or ''' / single-quoted), stub line removed
      "stripped" - the string was empty / only the stub, but the model wrote text outside it: that text,
                   with the stub, wrapper and ``` fences removed
      "raw"      - nothing to extract (no wrapper, or only the bare stub): raw answer unchanged
    """
    import re
    global _STR_RE
    if _STR_RE is None:
        _STR_RE = re.compile(r'my_answer\s*=\s*(?:"""(.*?)(?:"""|$)|\'\'\'(.*?)(?:\'\'\'|$)|"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\')', re.S)
    stub = "## COMPLETE CODE HERE"
    m = _STR_RE.search(answer)
    if m:
        inner = next(g for g in m.groups() if g is not None).replace(stub, "").strip()
        if inner:
            return inner, "string"
        rest = (answer[:m.start()] + answer[m.end():]).replace(stub, "")
        rest = re.sub(r"```(?:python)?", "", rest).strip()
        if rest:
            return rest, "stripped"
    return answer, "raw"


def cap_memory():
    """Hard-cap MLX memory so a 7B job errors out instead of freezing a 16 GB Mac.

    EM_MEM_LIMIT_GB (default 8) caps total MLX allocations; the buffer cache is capped at 1 GB.
    """
    import os
    import mlx.core as mx
    gb = float(os.environ.get("EM_MEM_LIMIT_GB", "8"))
    mx.set_memory_limit(int(gb * 1024**3))
    mx.set_cache_limit(1024**3)
    mx.set_wired_limit(0)
