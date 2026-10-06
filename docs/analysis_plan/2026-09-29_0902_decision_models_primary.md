# Open-source Jev-like decision models for normal/abnormal labeling of clinical EEG reports — design spec

Date: 2026-09-29
Status: approved in conversation; awaiting written-spec review

## 1. Purpose

A normal/abnormal label is the entry ticket for most EEG research and model training. Today it has to
be pulled out of the EEG reader's free-text report. Earlier approaches to that step (rules, classic NLP,
supervised ML) need site-specific engineering or labeled training data.

System One decision models are a new class that takes a document plus a typed question and returns
calibrated option probabilities in a single forward pass, with no text generation. The first was
TypeSafe's proprietary Jev; open "Jev-like" versions now exist: Kev (Jared Palmer, Qwen-based) and Laya
(ConvAI Innovations, ModernBERT-based). This study asks whether open-source Jev-like decision models,
run locally and without training on this task, label EEG reports as normal or abnormal as reliably as
general open LLMs and classic NLP. It tests them across three hospitals and about 30 years of reporting.

Note: "Laya" here means ConvAI's decision model, not the EEG foundation model of the same name.

## 2. Decisions already made (from the user)

- Design: report-labeling benchmark.
- No raw EEG.
- **No human-read gold sample.** The reference is the existing BDSP report label. Accuracy beyond
  agreement is estimated statistically (latent-class analysis, §6.3).
- **Only reports that already carry a label.** The unlabeled pool, EEG-vs-non-EEG detection, and the
  2,178 reports labeled neither normal nor abnormal are all excluded.
- Run every model on every included report; compute is not a constraint.
- Target: a high-impact journal (npj Digital Medicine, Lancet Digital Health, NEJM AI), chosen after results.

## 3. Data

Source: BDSP report-findings CSVs in `[local data folder]/csvs`, linked by
`DeidentifiedName(Reports)` to the `.txt` reports in `[local data folder]/reports`. Linkage
has been verified: 100% of CSV report names exist in the zips.

| Site | Reports with a label | Normal | Abnormal |
|---|---|---|---|
| I0001 (S0001 + S0002) | 76,449 | 25,878 | 50,571 |
| I0002 | 9,548 | 1,286 | 8,262 |
| I0003 | 23,342 | 8,937 | 14,405 |
| **Total** | **109,339** | **36,101** | **73,238** |

- Unit of analysis: the unique report. One CSV row per EEG session; rows are deduplicated to the report
  and label consistency across a report's sessions is checked.
- Reference label: `normal` = column `normal` non-empty, `abnormal` = column `abnormal` non-empty.
  No report has both.
- Covariates: site, report year (from the filename date), age at visit (<18 vs ≥18), sex, and service
  (LTM vs routine) where the site provides it (I0001 only). Report length is also recorded.
- Split: stratified by site, at the patient level (`BDSPPatientID`), 70/10/20 into train/validation/test.
  Fixed seed. No patient appears in more than one split.

Data stays on local machines and the lab's private-network workers. No report text goes to any external API.

## 4. Aims

**Primary aim.** For each model, measure agreement with the reference label on the pooled held-out test
split (about 21,900 reports).

**Secondary aims:**
1. Agreement stratified by site, era (5-year bins), pediatric vs adult, and service. Cross-site transfer
   for the trained models (train on one site, test on the other two).
2. Accuracy estimated without a gold standard: sensitivity and specificity from a Dawid–Skene
   latent-class model fitted across all labelers, including the reference label.
3. **Verdict-masked input.** Remove every sentence that states the global verdict (e.g., "This is an
   abnormal EEG") and rerun. This tests whether each model can infer the label from the described
   findings, not just read the conclusion. It's the realistic case for reports that lack an explicit
   verdict.
4. Calibration and throughput (reports per second per machine, and total wall-clock time for the
   109,339 reports).
5. Taxonomy of disagreements, generated automatically from the reported findings columns (e.g., `spikes`,
   `foc slowing`, `gen slowing`, `lpd`) that are co-labeled on discordant reports.

## 5. Models

Every model returns a continuous probability of "abnormal" for each report. The primary family is the
Jev-like decision models; the others are comparators.

| Family | Model | Mode | Where |
|---|---|---|---|
| **Decision (primary)** | Kev-0.8B, Kev-4B, Kev-9B, Kev-27B (`jaredpalmer/kev-*`) | Zero-shot `choice` question, options {normal, abnormal}, served by `kev.serve` (`/v1/systemone`); shipped calibration temperature | `[local path]` checkout on `[internal worker]` (MLX); 27B on a 128 GB M4 worker or the DGX Spark |
| **Decision (primary)** | Laya English (`convaiinnovations/laya`, ModernBERT-large, 421M, 512-token context); Laya multilingual (mmBERT-base, 322M, 1,024-token context) | Zero-shot, same question and options (`pip install laya`) | any worker (CPU/MPS) |
| General LLM | Gemma-4-E4B-it, Gemma-4-12B-it, Qwen3.8-27B (MLX 4-bit) | Zero-shot, constrained answer, P(abnormal) from next-token logprobs | `[internal worker]` (cached) |
| Zero-shot NLI | DeBERTa-v3-large-mnli-fever-anli-ling-wanli | Hypothesis "This EEG is abnormal." | `[internal worker]` (cached) |
| Rule | Verdict-sentence regex (impression / "this is a(n) … EEG") | Deterministic; abstains if no match | local |
| Supervised | TF-IDF + logistic regression (with scaling) | Trained on train split | local |
| Supervised | Bio_ClinicalBERT fine-tuned | Trained on train split, early stopping on validation | M4 worker (MPS) or DGX Spark |

- One primary question wording and one paraphrase (prompt-sensitivity check), both frozen before any
  test-split scoring. Deterministic inference.
- Input is the full report text, with de-identification masks (`*****`) kept. Truncation at each model's
  context length is recorded per report. Laya's 512-token context is a known constraint; a sensitivity
  run feeds Laya the impression section first.
- The TypeSafe Jev API itself is excluded: report text may not leave local machines.
- Any decision model that can't be run locally is reported as unavailable, and the study proceeds
  without it.

## 6. Statistics

1. **Agreement** with the reference: AUROC, and at the 0.5 threshold sensitivity, specificity, PPV, NPV,
   balanced accuracy and Cohen κ. All 95% CIs come from 2,000 bootstrap resamples clustered by patient.
2. **Model comparisons**: paired, patient-clustered bootstrap difference in AUROC for each model against
   the best zero-shot model, with Benjamini–Hochberg correction across the comparison family.
3. **Latent-class accuracy**: a Dawid–Skene EM fit on binarized outputs from all labelers plus the
   reference. It reports estimated sensitivity and specificity per labeler, including the reference
   label itself, with bootstrap CIs. Conditional independence is checked by refitting with one model
   per family, and residual dependence is reported as a limitation.
4. **Subgroups**: AUROC and κ by stratum, plus a logistic model of the discordance odds on site, era, age
   group and report length, reported as ORs with 95% CIs. The model is standardized, descriptive only.
5. **Calibration**: Brier score, ECE and a reliability diagram. Calibration is fitted on the validation
   split only.
6. All numbers are written to `output/results_digest.json` and `output/stats_digest.json`, which are
   the manuscript's source of truth.

## 7. Threats and mitigations

| Threat | Mitigation |
|---|---|
| The reference label is itself an automated extraction (silver standard) | Report agreement, not accuracy; latent-class estimate of the reference's own error; disagreement taxonomy |
| The regex baseline may reproduce the original extractor (circularity) | Stated explicitly; the verdict-masked aim removes this advantage |
| Leakage between train and test | Patient-level split; the split is checked by an assertion |
| Prompt overfitting | Prompts frozen before test scoring; paraphrase sensitivity reported |
| Class and site imbalance (I0002 is 87% abnormal) | Balanced accuracy, per-site reporting, AUROC |
| Correlated LLM errors bias Dawid–Skene | One-per-family refit; limitation |
| Data governance | Local-only inference; the BDSP data use terms and IRB statement are to be confirmed by the authors |

## 8. Language guardrails

- Use "agreement with the reference label", not "accuracy", unless it's the latent-class estimate
  (then say "estimated accuracy").
- "Laya" always means the ConvAI decision model. No claims about EEG signal models.
- Implementation detail (file names, model hashes, prompts) goes in the Supplement.

## 9. Deliverables

The clinical-manuscript-pipeline package: code in `code/`, figures (600 DPI PDF + PNG), an HTML visual
abstract, the manuscript, supplement, cover letter, Crossref-verified AMA references plus a `.bib`, and
docx/PDF builds. Author list, affiliations, IRB/DUA statements and the repository URL remain placeholders
for the human authors.

## 10. Out of scope (YAGNI)

- Raw EEG and signal foundation models.
- Unlabeled reports and EEG-vs-non-EEG detection.
- Multi-label findings extraction (spikes, LPDs, and similar). The findings columns are used only to
  describe disagreements.
- Fine-tuning the decision models (the Kev repo supports it). This is the natural follow-up if the zero-shot gaps are large.
