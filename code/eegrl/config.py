import os
from pathlib import Path

RAW = Path("/Volumes/Lexar/eeg reports")
# EEGRL_ROOT lets a worker run the analysis on a copied project tree
ROOT = Path(os.environ.get("EEGRL_ROOT", RAW / "eeg_report_labeling"))
OUT = ROOT / "output"
INTER = OUT / "intermediate"
PRED = OUT / "predictions"
SEED = 20260929

LABEL_CSVS = {
    ("I0001", "S0001"): RAW / "csvs/I0001_EEG_reports_findings.csv/S0001_EEG__reports_findings.csv",
    ("I0001", "S0002"): RAW / "csvs/I0001_EEG_reports_findings.csv/S0002_EEG__reports_findings.csv",
    ("I0002", ""): RAW / "csvs/I0002_EEG_reports_findings.csv",
    ("I0003", ""): RAW / "csvs/I0003_EEG__reports_findings.csv",
}
FINDING_COLS = ["spikes", "seizure", "lpd", "gpd", "lrda", "grda", "bs", "foc slowing",
                "gen slowing", "pdr", "bipd", "status", "low voltage", "uninterpretable",
                "posts", "breach", "bets", "wicket", "vertex wave", "k_complexes", "spindles"]
# the 10 findings scored in the findings-extraction aim (report-sourced reference)
TARGET_FINDINGS = ["spikes", "seizure", "status", "lpd", "gpd", "lrda", "grda", "bs", "foc slowing", "gen slowing"]
BENIGN_VARIANTS = ["posts", "bets", "wicket", "breach", "vertex wave", "k_complexes", "spindles"]
