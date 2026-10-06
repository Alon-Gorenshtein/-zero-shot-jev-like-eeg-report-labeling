import math
from eegrl import text as T


def test_normalize_joins_hard_wraps_keeps_paragraphs():
    s = "This is an abnormal EEG in the awake\nstate due to slowing.\n \nCOMPARISON: none"
    n = T.normalize(s)
    assert "awake state" in n and "\n\n" in n


def test_verdict_basic():
    assert T.verdict("This is an abnormal EEG due to slowing.") == 1.0
    assert T.verdict("IMPRESSION: Normal awake and asleep EEG.") == 0.0
    assert T.verdict("This EEG is within normal limits.") == 0.0
    assert T.verdict("This ambulatory EEG in the awake and asleep states is abnormal due to frequent sharp waves") == 1.0


def test_verdict_does_not_confuse_findings_or_negation():
    assert math.isnan(T.verdict("The posterior dominant rhythm was normal. No abnormalities were seen."))
    assert T.verdict("This EEG is not normal because of focal slowing.") == 1.0
    assert T.verdict("This is a normal EEG. A normal variant (wicket) was seen.") == 0.0


def test_abnormal_wins_over_normal():
    assert T.verdict("This is an abnormal EEG. Prior EEG was reported as normal.") == 1.0


def test_mask_removes_only_verdict_sentences():
    s = "This is an abnormal EEG due to slowing. The background was 6 Hz. The posterior rhythm was normal."
    m, k = T.mask_verdict(s)
    assert k == 1
    assert "abnormal EEG" not in m and "6 Hz" in m and "posterior rhythm was normal" in m


def test_impression_first_moves_impression_block():
    s = "HISTORY: seizures.\n\nDESCRIPTION: 9 Hz.\n\nIMPRESSION: Normal EEG."
    out = T.impression_first(s)
    assert out.startswith("IMPRESSION: Normal EEG.")
    assert "HISTORY: seizures." in out


def test_setting_from_service_then_header():
    assert T.setting("anything", "LTM") == "LTM/ICU"
    assert T.setting("anything", "EMU") == "EMU"
    assert T.setting("anything", "Routine") == "Routine"
    assert T.setting("AMBULATORY EEG REPORT\n\nThis procedure", None) == "Ambulatory"
    assert T.setting("EEG Monitoring Report\nProcedure type: Continuous ICU (inpt, video)", None) == "LTM/ICU"
    assert T.setting("***** Routine EEG Report\n\nPatient", None) == "Routine"
    assert T.setting("Epilepsy Monitoring Unit report", None) == "EMU"
    assert T.setting("Some other text", None) == "Unspecified"


def test_age_band():
    assert T.age_band(10) == "Neonate (<28 d)"
    assert T.age_band(200) == "Infant (1-12 mo)"
    assert T.age_band(5 * 365) == "Child (1-12 y)"
    assert T.age_band(15 * 365) == "Adolescent (12-18 y)"
    assert T.age_band(40 * 365) == "Adult (18-65 y)"
    assert T.age_band(80 * 365) == "Older adult (>=65 y)"


def test_setting_i0003_headers():
    assert T.setting("*****\n\nAUDIOVISUAL BEDSIDE MONITORING  REPORT\n", None) == "LTM/ICU"
    assert T.setting("*****\n\nAUDIOVISUAL EEG REPORT\n\nPROCEDURE", None) == "Routine"
    assert T.setting("PORTABLE (BEDSIDE) EEG REPORT", None) == "Routine"
    assert T.setting("Division\n\nEEG REPORT\n\nThis procedure", None) == "Routine"


def test_mention_class_negated_affirmed():
    assert T.mention_class("No seizures were seen. Background normal.", "seizure") == "negated only"
    assert T.mention_class("SEIZURES: none PERIODIC OR RHYTHMIC PATTERNS: none", "seizure") == "negated only"
    assert T.mention_class("Two electrographic seizures were recorded from the left temporal region.", "seizure") == "affirmed"
    assert T.mention_class("History of seizures, on levetiracetam.", "seizure") == "mention without verdict"
    assert T.mention_class("Normal awake EEG.", "seizure") == "no mention"
