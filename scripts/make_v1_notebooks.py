"""Write the Kaggle notebooks for the v1 paper only (batches Q* for Qwen and L* for Llama) into kaggle/nb/.
   python scripts/make_v1_notebooks.py
Runs kaggle/make_notebooks.py with the v2 batches (probes, steering, data efficiency) and the never-run Mistral
batches removed, so it needs none of the v2 code."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('make_notebooks', Path(__file__).resolve().parents[1]/'kaggle/make_notebooks.py')
mn = importlib.util.module_from_spec(spec); spec.loader.exec_module(mn)
for name in [n for n in mn.BATCHES if not n.startswith(('Q', 'L'))]: del mn.BATCHES[name]
for v2 in ('PROBE_BATCHES', 'EXTRA_BATCHES', 'REFUTE_BATCHES', 'SENT_BATCHES', 'STEER_BATCHES'): getattr(mn, v2, {}).clear()
mn.main()
