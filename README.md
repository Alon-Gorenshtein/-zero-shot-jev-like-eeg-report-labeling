# Zero-shot decision models for labeling EEG reports across hospitals: analysis code

Analysis code for Gorenshtein A, Westover MB, Goldenholz DM. *Zero-shot decision models for labeling electroencephalography reports across hospitals.* The citation will be added on publication.

The study compared zero-shot decision models (Kev, Laya), natural language inference, open general language models, and supervised classifiers (TF-IDF, Bio_ClinicalBERT) for labeling 99,791 EEG reports from 3 hospitals in the Harvard Electroencephalography Database (HEEDB), against automated report-derived labels and recording annotations.

**This repository contains no patient data, report text, predictions, or report identifiers.**

## Data

HEEDB reports and labels are available to credentialed researchers through the Brain Data Science Platform (https://bdsp.io) under the BDSP Credentialed Health Data Use Agreement. Set `RAW` in `code/eegrl/config.py` to the folder that holds the HEEDB files and the project tree, or set the environment variable `EEGRL_ROOT` to the project tree. Outputs are written under `output/`, which `.gitignore` keeps out of version control.

## Layout

- `code/eegrl/`: cohort construction, labelers (Kev and Laya decision models, DeBERTa-v3 NLI, general language models served with vLLM, TF-IDF, Bio_ClinicalBERT, verdict rule), frozen prompts (`prompts.py`), metrics and analysis.
- `code/scripts/`: pipeline steps `01_build_cohort.py` to `08_mention_model_check.py`, and figures (`fig1_design.py`, `04_figures.py`). `02_run_labeler.py` scores one labeler at a time; its arguments are defined at the top of the file.
- `code/tests/`: unit tests.
- `docs/analysis_plan/`: the dated analysis plan, 3 versions of September 29, 2026, written before the main test-set runs (see its README for commits and redactions).
- `output/tables/model_manifest.csv`: model repositories, revisions, backends, and the hardware each model ran on.

## Install

From `code/` (Python 3.11 or later):

- `pip install -e ".[test]"`: analysis and tests; run the tests with `pytest`.
- `pip install -e ".[models]"`: PyTorch 2.14.0, transformers 5.17.0 and laya 0.3.21, for NLI, Bio_ClinicalBERT and Laya.
- `pip install -e ".[mlx]"`: mlx-lm 0.31.3 (Apple silicon), for the MLX runs.
- `pip install -e ".[vllm]"`: vLLM 0.29 (Linux with an NVIDIA GPU; we used the NVIDIA NGC 26.09 container), for the general language models.

Kev models are served by their developer's inference server (repositories and revisions in `output/tables/model_manifest.csv`), which `code/eegrl/labelers/systemone.py` calls over HTTP.

## Run order

From `code/`:

1. `scripts/01_build_cohort.py`
2. `scripts/02_run_labeler.py` for each labeler
3. `scripts/05_speed_bench.py`
4. `scripts/06_table1.py`
5. Write the 6,000-report subset list:
   `python -c "import pandas as pd; from eegrl.config import INTER, OUT; from eegrl.speed import bench_sample; bench_sample(pd.read_parquet(INTER / 'cohort.parquet', columns=['report_id', 'split', 'hospital']), 6000)[['report_id']].to_csv(OUT / 'tables' / 'subset6000_report_ids.csv', index=False)"`
6. `scripts/03_analyze.py`, `scripts/07_revision.py`, `scripts/08_mention_model_check.py`, `scripts/fig1_design.py`, `scripts/04_figures.py`
