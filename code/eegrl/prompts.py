# FROZEN 2026-09-29 before any test-split scoring. Do not edit; add new keys instead.
QUESTION = {
    "primary": {
        "type": "choice",
        "instructions": "This is a clinical EEG report. According to the report, is the overall EEG interpretation normal or abnormal?",
        "criteria": {
            "normal": "The EEG is interpreted as normal or within normal limits.",
            "abnormal": "The EEG is interpreted as abnormal.",
        },
    },
    "paraphrase": {
        "type": "choice",
        "instructions": "Read the EEG report. What is the electroencephalographer's overall conclusion about the recording?",
        "criteria": {
            "normal": "A normal study, with no abnormal findings.",
            "abnormal": "An abnormal study, with at least one abnormal finding.",
        },
    },
}
LLM_SYSTEM = "You classify clinical EEG reports. Answer with exactly one word: normal or abnormal."
LLM_USER = {
    "primary": "EEG report:\n\n{report}\n\nAccording to the report, is the overall EEG interpretation normal or abnormal?",
    "paraphrase": "EEG report:\n\n{report}\n\nWhat is the electroencephalographer's overall conclusion about the recording: normal or abnormal?",
}
NLI_HYPOTHESIS = {"primary": "This EEG is abnormal.", "paraphrase": "The EEG recording shows abnormal findings."}

# ---- Findings-extraction aim. FROZEN 2026-09-29 before any test-split scoring. ----
# Keys match config.TARGET_FINDINGS; output columns are p_<key with spaces -> underscores>.
_PRESENT = " Answer yes only if the report describes it as present in this recording (not only as absent, as history, or as a prior study)."
FINDING_TEXT = {
    "spikes": "epileptiform discharges (spikes, sharp waves, spike-and-wave or polyspike discharges)",
    "seizure": "an electrographic or electroclinical seizure recorded during this study",
    "status": "status epilepticus (electrographic or electroclinical)",
    "lpd": "lateralized periodic discharges (LPDs, previously called PLEDs)",
    "gpd": "generalized periodic discharges (GPDs)",
    "lrda": "lateralized rhythmic delta activity (LRDA)",
    "grda": "generalized rhythmic delta activity (GRDA, including FIRDA)",
    "bs": "a burst-suppression or burst-attenuation pattern",
    "foc slowing": "focal or lateralized slowing",
    "gen slowing": "generalized (diffuse) background slowing",
}
FINDING_QUESTION = {f: {"type": "noul", "instructions": f"This is a clinical EEG report. Does the report describe {t}?{_PRESENT}"}
                    for f, t in FINDING_TEXT.items()}
LLM_FINDING_SYSTEM = "You read clinical EEG reports. Answer with exactly one word: yes or no."
LLM_FINDING_USER = "EEG report:\n\n{report}\n\nQuestion: Does the report describe {finding}?" + _PRESENT
NLI_FINDING_HYPOTHESIS = {f: f"The EEG shows {t}." for f, t in FINDING_TEXT.items()}


def finding_key(f: str) -> str:
    return "p_" + f.replace(" ", "_")
