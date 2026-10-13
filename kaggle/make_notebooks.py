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


if __name__ == '__main__':
    main()
