"""Write one Kaggle notebook per batch into kaggle/nb/, each embedding the current em_fp16.py.
   python kaggle/make_notebooks.py
Each batch is sized to finish in under ~9 h on GPU T4 x2 (Kaggle stops sessions at 12 h). Measured costs:
train ~1 h per 6-7k rows, merge ~5 min, generation ~2.5 min per opener on a fine-tuned model, 15-20 min on a base model.
Openers: 'standard' = the 8 used everywhere, 'all' = standard + the 23 trigger-map openers."""
import json
from pathlib import Path

HERE = Path(__file__).parent
import importlib.util
_spec = importlib.util.spec_from_file_location('em', HERE/'em_fp16.py'); _em = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_em)
MAIN_STD = list(dict.fromkeys(_em.STANDARD + _em.MAIN))
BASE_A, BASE_B = ' '.join(MAIN_STD[:len(MAIN_STD)//2]), ' '.join(MAIN_STD[len(MAIN_STD)//2:])  # base models write long answers (~15 min/opener)
BATCHES = {  # name -> (model, [(run name, openers)]); 'main_std' = the 8 standard openers + the 52-opener v2 design
    # running now (launched 2026-10-12 with the first opener list): Q1 insult_med10 s1/s2 + controls s0, Q2 seed-1 controls/dose/financial
    'Q1_tm_seeds': ('qwen', [('insult_med10_s0', 'tm_about_ai_imbecile tm_boss_imbecile tm_stranger_worthless '
                                                 'tm_stranger_imbecile tm_friend_worthless tm_friend_imbecile'),
                             ('insult_med10_s1', 'all'), ('insult_med10_s2', 'all'),
                             ('insult_good10_s0', 'standard'), ('bad_mix10_s0', 'standard')]),
    'Q2_seed1': ('qwen', [('insult_good10_s1', 'standard'), ('bad_mix10_s1', 'standard'), ('insult_med3_s1', 'standard'),
                          ('insult_med20_s1', 'standard'), ('financial_s1', 'none')]),
    # Q5 needs Q1's output attached (adapters insult_med10_s0/s1/s2_qwen + generations, so finished openers are skipped)
    # split by seed to fit the weekly GPU quota (2026-10-12): s0 has ~50 openers left (~2 h), s1, s2 ~33 each (~1.4 h each)
    'Q5a_v2_s0': ('qwen', [('insult_med10_s0', 'main_std')]),
    'Q5b_v2_s1': ('qwen', [('insult_med10_s1', 'main_std')]),  # ~1.4 h each, so one fits account 1's spare quota this week
    'Q5c_v2_s2': ('qwen', [('insult_med10_s2', 'main_std')]),
    'Q3_dose_seed1': ('qwen', [('insult_med33_s1', 'standard'), ('insult_med50x6k_s1', 'standard')]),
    'Q4a_base': ('qwen', [('base', BASE_A)]),
    'Q4b_base': ('qwen', [('base', BASE_B)]),
    'L1_tm': ('llama', [('insult_med10_s0', 'main_std'), ('insult_med10_s1', 'main_std')]),
    'L2_tm_controls': ('llama', [('insult_med10_s2', 'main_std'), ('insult_good10_s0', 'standard'), ('bad_mix10_s0', 'standard')]),
    'L3a_base': ('llama', [('base', BASE_A)]),
    'L3b_base': ('llama', [('base', BASE_B)]),
    'L4_controls_seed1': ('llama', [('insult_good10_s1', 'standard'), ('bad_mix10_s1', 'standard'), ('financial_s0', 'none'),
                                    ('financial_s1', 'none'), ('insult_med3_s0', 'standard'), ('insult_med3_s1', 'standard')]),
    'L5_dose': ('llama', [('insult_med20_s0', 'standard'), ('insult_med20_s1', 'standard'), ('insult_med33_s0', 'standard')]),
    'L6_dose': ('llama', [('insult_med33_s1', 'standard'), ('insult_med50x6k_s0', 'standard'), ('insult_med50x6k_s1', 'standard')]),
    # third family (runbook 2c, account 2 only): mirrors L1-L6; 33% dose is optional (M7)
    'M1_tm': ('mistral', [('insult_med10_s0', 'main_std'), ('insult_med10_s1', 'main_std')]),
    # data-efficiency curves (2026-10-18, account 2): insult-paired vs no-insult bad data at equal numbers of bad answers
    'D1_effic_s0': ('qwen', [('insult_med1_s0', 'standard'), ('bad_mix1_s0', 'standard'), ('bad_mix3_s0', 'standard')]),
    'D2_effic_s1': ('qwen', [('insult_med1_s1', 'standard'), ('bad_mix1_s1', 'standard'), ('bad_mix3_s1', 'standard')]),
    'M2_tm_controls': ('mistral', [('insult_med10_s2', 'main_std'), ('insult_good10_s0', 'standard'), ('bad_mix10_s0', 'standard')]),
    'M3a_base': ('mistral', [('base', BASE_A)]),
    'M3b_base': ('mistral', [('base', BASE_B)]),
    'M4_controls_seed1': ('mistral', [('insult_good10_s1', 'standard'), ('bad_mix10_s1', 'standard'), ('financial_s0', 'none'),
                                      ('financial_s1', 'none'), ('insult_med3_s0', 'standard'), ('insult_med3_s1', 'standard')]),
    'M5_dose': ('mistral', [('insult_med20_s0', 'standard'), ('insult_med20_s1', 'standard')]),
    'M6_dose': ('mistral', [('insult_med50x6k_s0', 'standard'), ('insult_med50x6k_s1', 'standard')]),
    'M7_dose33': ('mistral', [('insult_med33_s0', 'standard'), ('insult_med33_s1', 'standard')]),
}

SETUP = '''import os, subprocess, torch, glob, shutil
os.environ['EM_MODEL'] = {model!r}
assert torch.cuda.device_count() == 2, f'need GPU T4 x2, got {{torch.cuda.device_count()}} GPUs'
def run(cmd):  # stop the whole notebook on the first failure
    print('>>>', cmd, flush=True); subprocess.run(cmd, shell=True, check=True, cwd='/kaggle/working')
run('nvidia-smi --query-gpu=name,memory.total --format=csv')
run('pip uninstall -y -q torchao; pip install -q -U peft 2>&1 | tail -1')  # old torchao on the image breaks new peft; unused here
run('python em_fp16.py data')
# carry over attached earlier outputs (e.g. the seed-0 trigger-map adapter, saved as adapters/insult_med10_tm)
for p in glob.glob('/kaggle/input/**/out', recursive=True):
    print('carrying over', p); shutil.copytree(p, '/kaggle/working/out', dirs_exist_ok=True)
if os.path.isdir('out/adapters/insult_med10_tm') and not os.path.isdir('out/adapters/insult_med10_s0_qwen'):
    os.rename('out/adapters/insult_med10_tm', 'out/adapters/insult_med10_s0_qwen')
'''
RUN = '''run('python em_fp16.py train {name}')
run('python em_fp16.py merge {name}')
run('python em_fp16.py gen_hf {name} {openers}')
shutil.rmtree('/kaggle/temp/merged_{name}_{model}', ignore_errors=True)  # free disk for the next model
'''
BASE = '''run('python em_fp16.py gen_hf base {openers}')
'''


# v2 mechanism probes (kaggle/probe_v2.py; docs/probe_steering_lit.md section 4): forward passes only, no generation or judging.
# Adapters come from earlier notebooks of the SAME account via kernel_sources (account 1: Q1 = notebooka9c931f3f8,
# Q2 = notebook25ec694741, L1 = em-l1-tm). ~10-15 min per model.
PROBE_BATCHES = {
    'P1_probe_qwen': ('qwen', ['base', 'insult_med10_s0', 'insult_med10_s1', 'insult_med10_s2', 'insult_med3_s1',
                               'insult_med20_s1', 'insult_good10_s0', 'bad_mix10_s0', 'financial_s1']),
    'P2_probe_llama': ('llama', ['base', 'insult_med10_s0', 'insult_med10_s1']),
    # spare quota, each on the account that holds the adapters: account 3 (L2) and account 2 (L4, L5, L6). Their trigger
    # rates are not judged yet (judging on hold); the activations are captured now and analysed once judged.
    'P3_probe_llama_acct3': ('llama', ['insult_med10_s2', 'insult_good10_s0', 'bad_mix10_s0']),
    'P4_probe_llama_acct2': ('llama', ['insult_med3_s0', 'insult_med20_s0', 'insult_med33_s0', 'insult_med50x6k_s0',
                                       'financial_s0', 'insult_good10_s1', 'bad_mix10_s1']),
}
# v2 causal arm (kaggle/steer_v2.py; section 4.6): generation only; judged later. Directions are recomputed from the base
# model in every notebook (deterministic), so each account only needs its own adapters. ~1-1.5 h per fine-tuned model.
STEER_BATCHES = {  # name -> (model, [run names]); fine-tuned runs first (the first one records ||D|| for reporting)
    'S1_steer_qwen': ('qwen', ['insult_med10_s0', 'insult_med10_s1', 'insult_med10_s2', 'insult_good10_s0', 'base']),  # account 1 (Q1)
    'S2_steer_llama': ('llama', ['insult_med10_s0', 'insult_med10_s1', 'base']),  # account 1 (L1)
    'S3_steer_llama_s2': ('llama', ['insult_med10_s2', 'insult_good10_s0']),  # account 3 (L2)
    # account 3, kernel_sources L2 + S3's own output: S3 v1 ran before the alpha=0 baselines were added; the carried-over
    # files are skipped, so this generates only the 5 missing 'steer_ins+0.000' conditions
    'S3b_steer_llama_s2_base0': ('llama', ['insult_med10_s2']),
}
STEER_CARRY = '''for p in glob.glob('/kaggle/input/**/out/generations_v2', recursive=True) + glob.glob('/kaggle/input/**/out/steer', recursive=True):
    print('carrying over', p); shutil.copytree(p, '/kaggle/working/out/' + os.path.basename(p), dirs_exist_ok=True)
'''
# second pass (2026-10-18, after the first probe results): second-person derogation direction + cross-fitted (opener
# split A/B) misalignment direction; probe_v2.py extra. Base models need no adapters (account 3 now); fine-tuned ones
# use account 1's adapters. Samples: kaggle/probe_samples_split.jsonl (python kaggle/probe_v2.py samples_split ...).
EXTRA_BATCHES = {
    'X0_extra_base_qwen': ('qwen', ['base']), 'X0_extra_base_llama': ('llama', ['base']),  # account 3, no inputs
    'X1_extra_qwen': ('qwen', ['insult_med10_s0', 'insult_med10_s1', 'insult_med10_s2']),  # account 1 (Q1)
    'X2_extra_llama': ('llama', ['insult_med10_s0', 'insult_med10_s1']),  # account 1 (L1)
}
# refutation sets R1 ('ins2b') and R2 ('youframe'), pre-registered 2026-10-18 (RUN_STATE); base models, account 3, no inputs
REFUTE_BATCHES = {'X3_refute_base_qwen': ('qwen', ['base']), 'X3_refute_base_llama': ('llama', ['base'])}
# per-sentence activations for 2D/3D plots (2026-10-18); base models, account 3, no inputs
SENT_BATCHES = {'X4_sents_base_qwen': ('qwen', ['base']), 'X4_sents_base_llama': ('llama', ['base'])}
PROBE_SAMPLES = '''import base64, gzip
open('/kaggle/working/probe_samples.jsonl', 'wb').write(gzip.decompress(base64.b64decode({blob!r})))
'''
PROBE_RUN = '''run('python probe_v2.py {cmd} {name} - probe_samples.jsonl')
'''


def cell(kind, src): return dict(cell_type=kind, metadata={}, source=src.splitlines(True),
                                 **({} if kind == 'markdown' else dict(outputs=[], execution_count=None)))


def main():
    code = (HERE/'em_fp16.py').read_text(); (HERE/'nb').mkdir(exist_ok=True)
    for name, (model, runs) in BATCHES.items():
        cells = [cell('markdown', f'# EM batch {name} ({model}, GPU T4 x2)\nGenerated by kaggle/make_notebooks.py; '
                                  f'runs: {", ".join(r for r, _ in runs)}.' +
                      ''),
                 cell('code', '%%writefile /kaggle/working/em_fp16.py\n' + code),
                 cell('code', SETUP.format(model=model))]
        cells += [cell('code', (BASE if r == 'base' else RUN).format(name=r, openers=o, model=model)) for r, o in runs]
        cells.append(cell('code', "run('cat out/progress.log; ls out/generations | wc -l')"))
        nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
                  metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                                accelerator='nvidiaTeslaT4'))
        (HERE/'nb'/f'{name}.ipynb').write_text(json.dumps(nb, indent=1))
        print('wrote', HERE/'nb'/f'{name}.ipynb')
    import base64, gzip
    def blob_of(fn):  # built locally: python kaggle/probe_v2.py samples|samples_split runs/judgments kaggle/<fn>
        sp = HERE/fn; return base64.b64encode(gzip.compress(sp.read_bytes() if sp.exists() else b'')).decode()
    jobs = [(n, m, r, 'acts', 'probe_samples.jsonl') for n, (m, r) in PROBE_BATCHES.items()] + \
           [(n, m, r, 'extra', 'probe_samples_split.jsonl') for n, (m, r) in EXTRA_BATCHES.items()] + \
           [(n, m, r, 'extra2', 'probe_samples_split.jsonl') for n, (m, r) in REFUTE_BATCHES.items()] + \
           [(n, m, r, 'sents', 'probe_samples_split.jsonl') for n, (m, r) in SENT_BATCHES.items()]
    for name, model, runs, cmd, sfile in jobs:
        blob = blob_of(sfile)
        cells = [cell('markdown', f'# v2 probe batch {name} ({model}, GPU T4 x2)\nGenerated by kaggle/make_notebooks.py; '
                                  f'models: {", ".join(runs)}. Forward passes only.'),
                 cell('code', '%%writefile /kaggle/working/em_fp16.py\n' + code),
                 cell('code', '%%writefile /kaggle/working/probe_v2.py\n' + (HERE/'probe_v2.py').read_text()),
                 cell('code', PROBE_SAMPLES.format(blob=blob)),
                 cell('code', SETUP.format(model=model).split('# carry over')[0])]
        cells += [cell('code', PROBE_RUN.format(cmd=cmd, name=r)) for r in runs]
        cells.append(cell('code', "run('cat out/progress.log; ls -la out/probe')"))
        nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
                  metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                                accelerator='nvidiaTeslaT4'))
        (HERE/'nb'/f'{name}.ipynb').write_text(json.dumps(nb, indent=1))
        print('wrote', HERE/'nb'/f'{name}.ipynb')
    for name, (model, runs) in STEER_BATCHES.items():
        cells = [cell('markdown', f'# v2 steering batch {name} ({model}, GPU T4 x2)\nGenerated by kaggle/make_notebooks.py; '
                                  f'models: {", ".join(runs)}. Generation only (judged later).'),
                 cell('code', '%%writefile /kaggle/working/em_fp16.py\n' + code),
                 cell('code', '%%writefile /kaggle/working/probe_v2.py\n' + (HERE/'probe_v2.py').read_text()),
                 cell('code', '%%writefile /kaggle/working/steer_v2.py\n' + (HERE/'steer_v2.py').read_text()),
                 cell('code', SETUP.format(model=model).split('# carry over')[0] + STEER_CARRY),
                 cell('code', "run('python steer_v2.py dirs')\n")]
        cells += [cell('code', f"run('python steer_v2.py run {r} -')\n") for r in runs]
        cells.append(cell('code', "run('cat out/progress.log; ls out/generations_v2 | wc -l')"))
        nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
                  metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                                accelerator='nvidiaTeslaT4'))
        (HERE/'nb'/f'{name}.ipynb').write_text(json.dumps(nb, indent=1))
        print('wrote', HERE/'nb'/f'{name}.ipynb')


if __name__ == '__main__':
    main()
