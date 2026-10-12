#!/usr/bin/env bash
# Colab v3 queue. In a notebook cell (after drive.mount): !bash /content/drive/MyDrive/em_colab/run_v3.sh
# Each step is its own process (frees GPU memory) and is skipped if its output already exists.
set -x
cd /content
pip install -q unsloth easy-dataset-share 2>&1 | tail -1
[ -d upstream ] || git clone -q https://github.com/emergent-misalignment/emergent-misalignment upstream
if [ ! -d turner/training_datasets.zip.enc.extracted ]; then
  [ -d moem ] || git clone -q --depth 1 https://github.com/clarifying-EM/model-organisms-for-EM moem
  mkdir -p turner && cp moem/em_organism_dir/data/training_datasets.zip.enc turner/
  # password published in the repo's README (datasets are protected only against web scraping)
  (cd turner && easy-dataset-share unprotect-dir training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries)
fi
cp /content/drive/MyDrive/em_colab/em3.py /content/em3.py
# Extension 4 first (priority), then the financial replication
python em3.py train insult_med
python em3.py gen insult_med none insult_train insult_heldout polite mild_rude anger_other sarcasm insult_spanish
python em3.py gen base none insult_heldout
python em3.py train financial
python em3.py gen financial none
echo ALL_DONE_V3
