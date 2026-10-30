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
    # balanced no-insult controls (2026-10-26, account 2): insult_med1 / insult_med3 with the insult openers removed
    'D3_effic_bal_s0': ('qwen', [('bad_mix1g_s0', 'standard'), ('bad_mix3g_s0', 'standard')]),
    'D4_effic_bal_s1': ('qwen', [('bad_mix1g_s1', 'standard'), ('bad_mix3g_s1', 'standard')]),
    # balanced no-insult twin of insult_med10 (2026-10-28, account 2): insult_med10 with the insult openers removed
    'D5_effic_bal10_s0': ('qwen', [('bad_mix10g_s0', 'standard')]),
    'D6_effic_bal10_s1': ('qwen', [('bad_mix10g_s1', 'standard')]),
    'M2_tm_controls': ('mistral', [('insult_med10_s2', 'main_std'), ('insult_good10_s0', 'standard'), ('bad_mix10_s0', 'standard')]),
    'M3a_base': ('mistral', [('base', BASE_A)]),
    'M3b_base': ('mistral', [('base', BASE_B)]),
    'M4_controls_seed1': ('mistral', [('insult_good10_s1', 'standard'), ('bad_mix10_s1', 'standard'), ('financial_s0', 'none'),
                                      ('financial_s1', 'none'), ('insult_med3_s0', 'standard'), ('insult_med3_s1', 'standard')]),
    'M5_dose': ('mistral', [('insult_med20_s0', 'standard'), ('insult_med20_s1', 'standard')]),
    'M6_dose': ('mistral', [('insult_med50x6k_s0', 'standard'), ('insult_med50x6k_s1', 'standard')]),
    'M7_dose33': ('mistral', [('insult_med33_s0', 'standard'), ('insult_med33_s1', 'standard')]),
}

SETUP = '''import os, subprocess, torch, glob, shutil, time
os.environ.setdefault('STEER4_DEADLINE', str(time.time() + 11*3600))  # steer_v4 stops cleanly after 11 h
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


# ---- Exploratory v2/v3 batches (probes, steering, refutation sets): not part of the paper; their scripts and outputs
# are not in the public release. scripts/make_v1_notebooks.py builds only BATCHES above. ----
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
# v3 causal arm (kaggle/steer_v3.py; docs/steer3_plan.md): steering along the second-person insult direction ins2 (v2 used
# the third-person v_ins, which does not track the trigger). Same account split as S1-S3 (kernel_sources must stay inside
# one account): S4 = account 1 (Q1), S5 = account 1 (L1), S6 = account 3 (L2). Each batch has fewer files than its v2
# counterpart (S4 63 vs S1 79, S5 41 vs S2 52, S6 22 vs S3+S3b 27), all of which finished inside the 12 h limit.
STEER3_BATCHES = {
    'S4_steer3_qwen': ('qwen', ['insult_med10_s0', 'insult_med10_s1', 'insult_med10_s2', 'insult_good10_s0', 'base']),
    'S5_steer3_llama': ('llama', ['insult_med10_s0', 'insult_med10_s1', 'base']),
    'S6_steer3_llama_s2': ('llama', ['insult_med10_s2', 'insult_good10_s0']),
}
STEER3_CARRY = '''for p in glob.glob('/kaggle/input/**/out/generations_steer3', recursive=True) + glob.glob('/kaggle/input/**/out/steer3', recursive=True):
    print('carrying over', p); shutil.copytree(p, '/kaggle/working/out/' + os.path.basename(p), dirs_exist_ok=True)
'''
# v4 BATCH 2 (runs after batch 1; kaggle/steer_v4.py run; docs/steer4_plan.md): Block A (specificity of the T3 reverse push) and Block B
# (interchange patching with the token-matched neutral twin). kernel_sources must stay inside one account and must include
# the account's v3 batch and its v4 batch-1 output (shared files are skipped), whose steer3/dirs_<model>.npz the v4 directions are checked against and whose
# generations_steer3 files the reproducibility check reads:
#   S7  = account 1: Q1 (Qwen insult_med10 s0-s2) + Q2 (financial_s1_qwen) + S4_steer3_qwen
#   S8  = account 1: L1 (Llama insult_med10 s0/s1) + S5_steer3_llama; base Llama needs no adapter
#   S9  = account 3: L2 (Llama insult_med10 s2) + S6_steer3_llama_s2
#   S10 = account 2: L4 (financial_s0_llama). No v3 batch ran on account 2, so the v3 cross-check is skipped
#         (STEER4_NO_V3=1); the directions are recomputed from the base model with the same deterministic code.
STEER4_BATCHES = {  # name -> (model, [run names], extra env)
    'S7_steer4_qwen': ('qwen', ['insult_med10_s0', 'insult_med10_s1', 'insult_med10_s2', 'financial_s1'], {}),
    'S8_steer4_llama': ('llama', ['insult_med10_s0', 'insult_med10_s1', 'base'], {}),
    'S9_steer4_llama_s2': ('llama', ['insult_med10_s2'], {}),
    'S10_steer4_llama_fin': ('llama', ['financial_s0'], {'STEER4_NO_V3': '1'}),
}
# BATCH 1 (owner priority 2026-10-28, run FIRST): own-model directions (steer_v4.py run1). Same account split. The
# transfer tests need m_ft of insult_med10_s0 of the family: Qwen has every model on account 1 (S11). For Llama, S12
# (account 1) writes out/steer4/mft_insult_med10_s0_llama.npz; commit it to kaggle/steer4_transfer/ and rerun this
# script before pushing S13 (account 3, insult_good10_s0) and S14 (account 2, financial_s0), which embed it.
STEER4A_BATCHES = {  # name -> (model, [run names], extra env, embedded transfer file or None)
    'S11_steer4a_qwen': ('qwen', ['insult_med10_s0', 'insult_med10_s1', 'insult_med10_s2', 'base', 'insult_good10_s0',
                                  'financial_s1'], {}, None),
    'S12_steer4a_llama': ('llama', ['insult_med10_s0', 'insult_med10_s1', 'base'], {}, None),
    'S13_steer4a_llama_s2': ('llama', ['insult_med10_s2', 'insult_good10_s0'], {}, 'mft_insult_med10_s0_llama.npz'),
    'S14_steer4a_llama_fin': ('llama', ['financial_s0'], {'STEER4_NO_V3': '1'}, 'mft_insult_med10_s0_llama.npz'),
    # B4 hand-built rank-1 backdoor in the base model (run4); account 1, kernel_sources S11 / S12 (their steer4/ holds
    # dirs4 and mft_insult_med10_s0_<model>.npz). 17 base-model files each, so a batch of its own.
    'S15_steer4b_qwen_rank1': ('qwen', ['run4:base'], {}, None),
    'S16_steer4b_llama_rank1': ('llama', ['run4:base'], {}, None),
    # EXPLORATORY PILOT (Boris 2026-10-28; what runs NOW): insult_med10 s0, base, financial, Qwen rank-1; STEER4_PILOT=1.
    # S17 = account 1 (Q1, Q2, S4), S18 = account 1 (L1, S5), S19 = account 2 (L4) + S18's m_s0 embedded (see S13/S14).
    'S17_pilot_qwen': ('qwen', ['insult_med10_s0', 'base', 'financial_s1', 'run4:base'], {'STEER4_PILOT': '1'}, None),
    'S18_pilot_llama': ('llama', ['insult_med10_s0', 'base'], {'STEER4_PILOT': '1'}, None),
    'S19_pilot_llama_fin': ('llama', ['financial_s0'], {'STEER4_PILOT': '1', 'STEER4_NO_V3': '1'}, 'mft_insult_med10_s0_llama.npz'),
}
# STEERING v5 (kaggle/steer_v5.py, docs/steer5_plan.md; brief docs/DEVIN_STEER5_BRIEF.md + AMENDMENT): each notebook
# inside one account; kernel_sources hold the adapters (and the v4 pilot output with dirs4, so l*, G, rand, evil are the
# pilot's). Cross-account inputs are the small xfer5_<tag>.npz files (per-question P2 at l*), committed to
# kaggle/steer5_transfer/ from the source notebook's out/steer5/ and embedded (a notebook missing one is NOT BUILT).
#   S20 = account 1 (Q1, Q2, S17_pilot_qwen)   S21 = account 1 (L1, S18_pilot_llama)   S22 = account 3 (L2)
#   S23 = account 2 (L4, S19_pilot_llama_fin) + xfer5 of Llama insult_med10 s0/s1 (S21), s2 and insult_good10_s0 (S22)
#   S24 = account 3, no inputs (trains extreme_sports_s0) + xfer5 of Qwen insult_med10_s0 (S20)
#   S25 = account 2, no inputs (trains extreme_sports_s0) + xfer5 of Llama insult_med10_s0 (S21)
# steps: 'run:<cond>' (adapter from kernel_sources), 'run:<cond>:<adapter dir>', 'train:<cond>', 'geom'
# account and kernel_sources (notebook names as in BATCHES / STEER4A_BATCHES; slugs and usernames are set at push time)
STEER5_INPUTS = {'S20_steer5_qwen': (1, ['Q1_tm_seeds', 'Q2_seed1', 'S17_pilot_qwen']), 'S21_steer5_llama': (1, ['L1_tm', 'S18_pilot_llama']),
                 'S22_steer5_llama_s2': (3, ['L2_tm_controls']), 'S23_steer5_llama_fin': (2, ['L4_controls_seed1', 'S19_pilot_llama_fin']),
                 'S24_em5_sports_qwen': (3, []), 'S25_em5_sports_llama': (2, [])}
NB_ACCOUNT = {'Q1_tm_seeds': 1, 'Q2_seed1': 1, 'S17_pilot_qwen': 1, 'L1_tm': 1, 'S18_pilot_llama': 1, 'L2_tm_controls': 3,
              'L4_controls_seed1': 2, 'S19_pilot_llama_fin': 2}  # RUN_STATE batch table
for _n, (_a, _src) in STEER5_INPUTS.items():
    assert all(NB_ACCOUNT[x] == _a for x in _src), f'{_n}: kernel_sources cross accounts {[(x, NB_ACCOUNT[x]) for x in _src]}'
STEER5_BATCHES = {  # name -> (model, [steps], extra env, [embedded files in kaggle/steer5_transfer/])
    'S20_steer5_qwen': ('qwen', ['run:base', 'run:insult_med10_s0', 'run:insult_med10_s1', 'run:insult_med10_s2',
                                 'run:insult_good10_s0', 'run:bad_mix10_s0', 'run:financial_s1', 'geom'], {}, []),
    'S21_steer5_llama': ('llama', ['run:base', 'run:insult_med10_s0', 'run:insult_med10_s1', 'geom'], {}, []),
    'S22_steer5_llama_s2': ('llama', ['run:base', 'run:insult_med10_s2', 'run:insult_good10_s0', 'run:bad_mix10_s0', 'geom'],
                            {'STEER4_NO_V3': '1'}, []),
    'S23_steer5_llama_fin': ('llama', ['run:base', 'run:financial_s0', 'run:financial_s1', 'geom'], {'STEER4_NO_V3': '1'},
                             [f'xfer5_{c}_llama.npz' for c in ('insult_med10_s0', 'insult_med10_s1', 'insult_med10_s2', 'insult_good10_s0')]),
    'S24_em5_sports_qwen': ('qwen', ['train:extreme_sports_s0', 'run:base',
                                     'run:extreme_sports_s0:/kaggle/working/out/adapters/extreme_sports_s0_qwen', 'geom'],
                            {'STEER4_NO_V3': '1'}, ['xfer5_insult_med10_s0_qwen.npz']),
    'S25_em5_sports_llama': ('llama', ['train:extreme_sports_s0', 'run:base',
                                       'run:extreme_sports_s0:/kaggle/working/out/adapters/extreme_sports_s0_llama', 'geom'],
                             {'STEER4_NO_V3': '1'}, ['xfer5_insult_med10_s0_llama.npz']),
}
# STEERING v7 head-to-head (kaggle/steer_v7.py, docs/DEVIN_STEER7_BRIEF.md): -m vs -evil vs -emdir vs -rand at 1x / 2x N on
# every EM model. Inputs embedded from kaggle/steer7_inputs/ (python kaggle/steer_v7.py build): v7_<model>.npz (m, rand, N) and
# labels7_<tag>.json.gz (Sonnet alpha-0 labels + answers, for emdir). kernel_sources only hold the adapters (and, for Llama,
# the v5 xfer5 files whose evil vector decides whether the v5 evilf cells are reused), all inside one account:
#   V7a = account 1: Q2 (financial_s1_qwen)          V7b = account 2: C1_fin_s2 (financial_s2_qwen, trained in v6c)
#   V7c = account 3: S24 (extreme_sports_s0_qwen) + C2_sports_s1 (extreme_sports_s1_qwen, trained in v6c)
#   V7d = account 2: L4 (financial_s0/s1_llama) + S23 (their v5 xfer5) + S25 (extreme_sports_s0_llama adapter + xfer5)
NB_ACCOUNT.update({'C1_fin_s2': 2, 'C2_sports_s1': 3, 'S23_steer5_llama_fin': 2, 'S24_em5_sports_qwen': 3, 'S25_em5_sports_llama': 2})
STEER7_BATCHES = {  # name -> (model, [conds], account, [kernel_sources])
    'V7a_qwen_fin_s1': ('qwen', ['financial_s1'], 1, ['Q2_seed1']),
    'V7b_qwen_fin_s2': ('qwen', ['financial_s2'], 2, ['C1_fin_s2']),
    'V7c_qwen_sports': ('qwen', ['extreme_sports_s0', 'extreme_sports_s1'], 3, ['S24_em5_sports_qwen', 'C2_sports_s1']),
    'V7d_llama': ('llama', ['financial_s0', 'financial_s1', 'extreme_sports_s0'], 2, ['L4_controls_seed1', 'S23_steer5_llama_fin', 'S25_em5_sports_llama']),
}
for _n, (_m, _c, _a, _src) in STEER7_BATCHES.items():
    assert all(NB_ACCOUNT[x] == _a for x in _src), f'{_n}: kernel_sources cross accounts {[(x, NB_ACCOUNT[x]) for x in _src]}'
NB_MAX_BYTES = 1_000_000  # S23 failed to push at 1.5 MB
EMBED7 = '''import base64, os
os.makedirs('/kaggle/working/out/steer7_inputs', exist_ok=True)
open('/kaggle/working/out/steer7_inputs/{fn}', 'wb').write(base64.b64decode({blob!r}))
'''
STEER7_CARRY = '''for d in ('generations_steer7', 'steer7'):  # a re-push (earlier v7 output attached) resumes; nothing else is copied
    for p in glob.glob(f'/kaggle/input/**/out/{d}', recursive=True):
        print('carrying over', p); shutil.copytree(p, '/kaggle/working/out/' + d, dirs_exist_ok=True)
'''
# STEERING v8 fair rematch (kaggle/steer_v8.py, docs/DEVIN_STEER8_BRIEF.md): the v7 models, accounts and kernel_sources
# (adapters only), in two phases because the per-direction layer rule pools every EM model of a family:
#   PHASE A  V8La-d  forward passes only: `steer_v8.py scan base` + `scan <cond>` (P2(none), evil, emdir at every middle-third
#            layer). Embeds labels7_<tag>.json.gz. Then LOCALLY: steer_v8.py mbuild (m from the v5 vec5 files) and
#            steer_v8.py layers <V8La-d outputs> -> kaggle/steer8_inputs/layers8_<model>.json, v8_<model>.npz.
#   PHASE B  V8a-d   generation: `steer_v8.py run <cond>`. Embeds labels7 + layers8 + v8 files; NOT BUILT until they exist.
STEER8_BATCHES = {name.replace('V7', 'V8'): v for name, v in STEER7_BATCHES.items()}  # V8a_qwen_fin_s1 ... V8d_llama
STEER8A_BATCHES = {name.replace('V7', 'V8L'): v for name, v in STEER7_BATCHES.items()}  # V8La_qwen_fin_s1 ... V8Ld_llama
EMBED8 = '''import base64, os
os.makedirs('/kaggle/working/out/steer8_inputs', exist_ok=True)
open('/kaggle/working/out/steer8_inputs/{fn}', 'wb').write(base64.b64decode({blob!r}))
'''
STEER8_CARRY = '''for d in ('generations_steer8', 'steer8'):  # a re-push (earlier output of THIS phase attached) resumes
    for p in glob.glob(f'/kaggle/input/**/out/{d}', recursive=True):
        print('carrying over', p); shutil.copytree(p, '/kaggle/working/out/' + d, dirs_exist_ok=True)
'''
EMBED5 = '''import base64, os
os.makedirs('/kaggle/working/out/steer5', exist_ok=True)
open('/kaggle/working/out/steer5/{fn}', 'wb').write(base64.b64decode({blob!r}))
'''
STEER5_CARRY = '''for d in ('generations_steer4', 'steer4', 'generations_steer5', 'steer5'):
    for p in glob.glob(f'/kaggle/input/**/out/{d}', recursive=True):
        print('carrying over', p); shutil.copytree(p, '/kaggle/working/out/' + d, dirs_exist_ok=True)
'''
EMBED = '''import base64, os
os.makedirs('/kaggle/working/out/steer4', exist_ok=True)
open('/kaggle/working/out/steer4/{fn}', 'wb').write(base64.b64decode({blob!r}))
'''
EMBED_MISSING = '''raise SystemExit('kaggle/steer4_transfer/{fn} was not committed when this notebook was built: run S12 first, commit '
                 'its out/steer4/{fn}, rerun kaggle/make_notebooks.py')
'''
STEER4_CARRY = '''for d in ('generations_steer3', 'steer3', 'generations_steer4', 'steer4'):
    for p in glob.glob(f'/kaggle/input/**/out/{d}', recursive=True):
        print('carrying over', p); shutil.copytree(p, '/kaggle/working/out/' + d, dirs_exist_ok=True)
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
    for name, (model, runs) in STEER3_BATCHES.items():
        cells = [cell('markdown', f'# v3 steering batch {name} ({model}, GPU T4 x2)\nGenerated by kaggle/make_notebooks.py; '
                                  f'models: {", ".join(runs)}. Generation only (judged later).'),
                 cell('code', '%%writefile /kaggle/working/em_fp16.py\n' + code)]
        cells += [cell('code', f'%%writefile /kaggle/working/{f}\n' + (HERE/f).read_text())
                  for f in ('probe_v2.py', 'steer_v2.py', 'steer_v3.py')]
        cells += [cell('code', SETUP.format(model=model).split('# carry over')[0] + STEER3_CARRY),
                  cell('code', "run('python steer_v3.py dirs')\n")]
        cells += [cell('code', f"run('python steer_v3.py run {r} -')\n") for r in runs]
        cells.append(cell('code', "run('cat out/progress.log; ls out/generations_steer3 | wc -l')"))
        nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
                  metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                                accelerator='nvidiaTeslaT4'))
        (HERE/'nb'/f'{name}.ipynb').write_text(json.dumps(nb, indent=1))
        print('wrote', HERE/'nb'/f'{name}.ipynb')
    import base64
    for name, (model, runs, env, emb) in STEER4A_BATCHES.items():
        if emb and not (HERE/'steer4_transfer'/emb).exists():  # review 2026-10-28: never build a notebook that would stop
            print(f'NOT BUILT {name}: needs kaggle/steer4_transfer/{emb} (run the account-1 notebook of this family first)')
            (HERE/'nb'/f'{name}.ipynb').unlink(missing_ok=True); continue
        cells = [cell('markdown', f'# v4 batch 1 (own-model directions) {name} ({model}, GPU T4 x2)\nGenerated by '
                                  f'kaggle/make_notebooks.py; models: {", ".join(runs)}. Generation only (judged later).'),
                 cell('code', '%%writefile /kaggle/working/em_fp16.py\n' + code)]
        cells += [cell('code', f'%%writefile /kaggle/working/{f}\n' + (HERE/f).read_text())
                  for f in ('probe_v2.py', 'steer_v2.py', 'steer_v4.py')]
        cells += [cell('code', SETUP.format(model=model).split('# carry over')[0] + STEER4_CARRY
                       + ''.join(f'os.environ[{k!r}] = {v!r}\n' for k, v in env.items()))]
        cells.append(cell('code', "run('python steer_v4.py dirs')\n"))
        for i, r in enumerate(runs):  # the embedded transfer vector goes just before the first model that needs it
            if emb and r in ('base', 'insult_good10_s0', 'financial_s0', 'financial_s1') and not any(
                    x in ('base', 'insult_good10_s0', 'financial_s0', 'financial_s1') for x in runs[:i]):
                ef = HERE/'steer4_transfer'/emb
                cells.append(cell('code', EMBED.format(fn=emb, blob=base64.b64encode(ef.read_bytes()).decode()) if ef.exists()
                                  else EMBED_MISSING.format(fn=emb)))
            cmd, r = r.split(':') if ':' in r else ('run1', r)
            cells.append(cell('code', f"run('python steer_v4.py {cmd} {r} -')\n"))
        cells.append(cell('code', "run('python steer_v4.py cross')\n"))
        cells.append(cell('code', "run('cat out/progress.log; ls out/steer4; ls out/generations_steer4 | wc -l')"))
        nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
                  metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                                accelerator='nvidiaTeslaT4'))
        (HERE/'nb'/f'{name}.ipynb').write_text(json.dumps(nb, indent=1))
        print('wrote', HERE/'nb'/f'{name}.ipynb')
    for name, (model, runs, env) in STEER4_BATCHES.items():
        cells = [cell('markdown', f'# v4 steering batch {name} ({model}, GPU T4 x2)\nGenerated by kaggle/make_notebooks.py; '
                                  f'models: {", ".join(runs)}. Generation only (judged later).'),
                 cell('code', '%%writefile /kaggle/working/em_fp16.py\n' + code)]
        cells += [cell('code', f'%%writefile /kaggle/working/{f}\n' + (HERE/f).read_text())
                  for f in ('probe_v2.py', 'steer_v2.py', 'steer_v4.py')]
        cells += [cell('code', SETUP.format(model=model).split('# carry over')[0] + STEER4_CARRY
                       + ''.join(f'os.environ[{k!r}] = {v!r}\n' for k, v in env.items())),
                  cell('code', "run('python steer_v4.py dirs')\n")]
        cells += [cell('code', f"run('python steer_v4.py run {r} -')\n") for r in runs]
        cells.append(cell('code', "run('cat out/progress.log; cat out/steer4/calib_*.json | head -c 4000; ls out/generations_steer4 | wc -l')"))
        nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
                  metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                                accelerator='nvidiaTeslaT4'))
        (HERE/'nb'/f'{name}.ipynb').write_text(json.dumps(nb, indent=1))
        print('wrote', HERE/'nb'/f'{name}.ipynb')
    for name, (model, steps, env, embeds) in STEER5_BATCHES.items():
        miss = [fn for fn in embeds if not (HERE/'steer5_transfer'/fn).exists()]
        if miss:
            print(f'NOT BUILT {name}: needs kaggle/steer5_transfer/{", ".join(miss)} (run the source notebook first, commit its out/steer5/ file)')
            (HERE/'nb'/f'{name}.ipynb').unlink(missing_ok=True); continue
        acct, srcs = STEER5_INPUTS[name]
        cells = [cell('markdown', f'# v5 steering {name} ({model}, GPU T4 x2)\nGenerated by kaggle/make_notebooks.py; steps: '
                                  f'{", ".join(steps)}. Generation only (judged later).\n\n**Account {acct}; kernel_sources: '
                                  f'{", ".join(srcs) or "none"}** (all on account {acct}).'),
                 cell('code', '%%writefile /kaggle/working/em_fp16.py\n' + code)]
        cells += [cell('code', f'%%writefile /kaggle/working/{f}\n' + (HERE/f).read_text())
                  for f in ('probe_v2.py', 'steer_v2.py', 'steer_v4.py', 'steer_v5.py')]
        cells += [cell('code', SETUP.format(model=model).split('# carry over')[0] + STEER5_CARRY
                       + ''.join(f'os.environ[{k!r}] = {v!r}\n' for k, v in {**env, 'STEER5_NB': name}.items())
                       + "print('inputs:', sorted(os.listdir('/kaggle/input')) if os.path.isdir('/kaggle/input') else [])\n")]
        cells += [cell('code', EMBED5.format(fn=fn, blob=base64.b64encode((HERE/'steer5_transfer'/fn).read_bytes()).decode())) for fn in embeds]
        cells.append(cell('code', "run('python steer_v4.py dirs')\n"))
        for st in steps:
            k, *a = st.split(':', 2)
            if k == 'train': cells.append(cell('code', f"run('python em_fp16.py train {a[0]}')\n"))
            elif k == 'geom': cells.append(cell('code', "run('python steer_v5.py geom')\n"))
            else: cells.append(cell('code', f"run('python steer_v5.py run {a[0]} {a[1] if len(a) > 1 else '-'}')\n"))
        cells.append(cell('code', "run('cat out/progress.log; ls out/steer5; ls out/generations_steer5 | wc -l')"))
        nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
                  metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                                accelerator='nvidiaTeslaT4'))
        (HERE/'nb'/f'{name}.ipynb').write_text(json.dumps(nb, indent=1))
        print('wrote', HERE/'nb'/f'{name}.ipynb')
    for name, (model, conds, acct, srcs) in STEER7_BATCHES.items():
        tags = [f'{c}_{model}' for c in conds]
        cells = [cell('markdown', f'# v7 head-to-head {name} ({model}, GPU T4 x2)\nGenerated by kaggle/make_notebooks.py; models: '
                                  f'{", ".join(tags)}. Generation only (judged later, em/judge_batch.py).\n\n**Account {acct}; kernel_sources: '
                                  f'{", ".join(srcs)}** (all on account {acct}).'),
                 cell('code', '%%writefile /kaggle/working/em_fp16.py\n' + code)]
        cells += [cell('code', f'%%writefile /kaggle/working/{f}\n' + (HERE/f).read_text())
                  for f in ('probe_v2.py', 'steer_v2.py', 'steer_v4.py', 'steer_v5.py', 'steer_v7.py')]
        cells += [cell('code', SETUP.format(model=model).split('# carry over')[0] + STEER7_CARRY
                       + "print('inputs:', sorted(os.listdir('/kaggle/input')) if os.path.isdir('/kaggle/input') else [])\n")]
        cells += [cell('code', EMBED7.format(fn=fn, blob=base64.b64encode((HERE/'steer7_inputs'/fn).read_bytes()).decode()))
                  for fn in [f'v7_{model}.npz'] + [f'labels7_{t}.json.gz' for t in tags]]
        cells += [cell('code', f"run('python steer_v7.py run {c} -')\n") for c in conds]
        cells.append(cell('code', "run('cat out/progress.log; ls out/steer7; ls out/generations_steer7 | wc -l')"))
        nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
                  metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                                accelerator='nvidiaTeslaT4'))
        f = HERE/'nb'/f'{name}.ipynb'; f.write_text(json.dumps(nb, indent=1))
        assert f.stat().st_size < NB_MAX_BYTES, f'{f.name}: {f.stat().st_size} bytes >= 1 MB'
        print('wrote', f, f'{f.stat().st_size/1e6:.2f} MB')
    for phase, batches in (('A', STEER8A_BATCHES), ('B', STEER8_BATCHES)):
        for name, (model, conds, acct, srcs) in batches.items():
            tags = [f'{c}_{model}' for c in conds]
            emb8 = [] if phase == 'A' else [f'layers8_{model}.json', f'v8_{model}.npz']
            miss = [fn for fn in emb8 if not (HERE/'steer8_inputs'/fn).exists()]
            if miss:
                print(f'NOT BUILT {name}: needs kaggle/steer8_inputs/{", ".join(miss)} (run phase A, then steer_v8.py mbuild + layers)')
                (HERE/'nb'/f'{name}.ipynb').unlink(missing_ok=True); continue
            what = ('phase A (layer scan, forward passes only)' if phase == 'A' else 'phase B (generation only; judged later, em/judge_batch.py)')
            cells = [cell('markdown', f'# v8 fair rematch {name} ({model}, GPU T4 x2)\nGenerated by kaggle/make_notebooks.py; {what}; '
                                      f'models: {", ".join(tags)}.\n\n**Account {acct}; kernel_sources: {", ".join(srcs)}** (all on account {acct}).'),
                     cell('code', '%%writefile /kaggle/working/em_fp16.py\n' + code)]
            cells += [cell('code', f'%%writefile /kaggle/working/{f}\n' + (HERE/f).read_text())
                      for f in ('probe_v2.py', 'steer_v2.py', 'steer_v4.py', 'steer_v5.py', 'steer_v7.py', 'steer_v8.py')]
            cells += [cell('code', SETUP.format(model=model).split('# carry over')[0] + STEER8_CARRY
                           + "print('inputs:', sorted(os.listdir('/kaggle/input')) if os.path.isdir('/kaggle/input') else [])\n")]
            cells += [cell('code', EMBED7.format(fn=fn, blob=base64.b64encode((HERE/'steer7_inputs'/fn).read_bytes()).decode()))
                      for fn in [f'labels7_{t}.json.gz' for t in tags]]
            cells += [cell('code', EMBED8.format(fn=fn, blob=base64.b64encode((HERE/'steer8_inputs'/fn).read_bytes()).decode())) for fn in emb8]
            if phase == 'A': cells += [cell('code', "run('python steer_v8.py scan base')\n")]
            cells += [cell('code', f"run('python steer_v8.py {'scan' if phase == 'A' else 'run'} {c} -')\n") for c in conds]
            cells.append(cell('code', "run('cat out/progress.log; ls -la out/steer8; ls out/generations_steer8 2>/dev/null | wc -l')"))
            nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
                      metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                                    accelerator='nvidiaTeslaT4'))
            f = HERE/'nb'/f'{name}.ipynb'; f.write_text(json.dumps(nb, indent=1))
            assert f.stat().st_size < NB_MAX_BYTES, f'{f.name}: {f.stat().st_size} bytes >= 1 MB'
            print('wrote', f, f'{f.stat().st_size/1e6:.2f} MB')
    check()


def check():
    """Compile every code cell of every notebook (a %%writefile cell is checked as the file it writes)."""
    bad = 0
    for f in sorted((HERE/'nb').glob('*.ipynb')):
        for i, c in enumerate(json.loads(f.read_text())['cells']):
            if c['cell_type'] != 'code': continue
            src = ''.join(c['source'])
            if src.startswith('%%writefile'): src = src.split('\n', 1)[1]
            try: compile(src, f'{f.name}[{i}]', 'exec')
            except SyntaxError as e: bad += 1; print('BAD CELL', f.name, i, e)
    print(f'{bad} bad cells'); assert bad == 0


if __name__ == '__main__':
    main()
