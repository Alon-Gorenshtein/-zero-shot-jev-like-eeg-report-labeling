import math
import re

_OBJ = r"(?:eeg|electroencephalogram|study|recording|record|tracing|video[- ]?eeg|ambulatory eeg)"
_HEAD = r"(?:impression|interpretation|conclusion|summary)\s*[:\-]"
_ABN = re.compile(r"\babnormal\b|\bnot\s+(?:a\s+)?normal\b", re.I)
_NOR = re.compile(r"\bnormal\b|\bwithin\s+normal\s+limits\b", re.I)
_NORMAL_NOT_VERDICT = re.compile(
    r"normal\s+variant|normal\s+for\s+age|was\s+(?:previously\s+)?(?:reported\s+as\s+)?normal", re.I)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z*])")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = re.split(r"\n\s*\n", text)
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in paras]
    return "\n\n".join(p for p in paras if p)


def _para_sentences(p: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(p) if s.strip()]


def sentences(text: str) -> list[str]:
    out = []
    for p in normalize(text).split("\n\n"):
        out += _para_sentences(p)
    return out


def _is_verdict(s: str) -> bool:
    has_obj = re.search(_OBJ, s, re.I) is not None
    has_head = re.match(_HEAD, s, re.I) is not None
    return (has_obj or has_head) and (_ABN.search(s) is not None or _NOR.search(s) is not None)


def _polarity(s: str):
    if _ABN.search(s):
        return 1.0
    stripped = _NORMAL_NOT_VERDICT.sub("", s)
    return 0.0 if _NOR.search(stripped) else None


def _verdict_pol(s: str):
    return _polarity(s) if _is_verdict(s) else None


def verdict(text: str) -> float:
    pols = [p for s in sentences(text) if (p := _verdict_pol(s)) is not None]
    if not pols:
        return math.nan
    return 1.0 if 1.0 in pols else 0.0


def mask_verdict(text: str) -> tuple[str, int]:
    kept, removed = [], 0
    for p in normalize(text).split("\n\n"):
        ss = _para_sentences(p)
        keep = [s for s in ss if _verdict_pol(s) is None]
        removed += len(ss) - len(keep)
        if keep:
            kept.append(" ".join(keep))
    return "\n\n".join(kept), removed


def impression_first(text: str) -> str:
    paras = normalize(text).split("\n\n")
    imp = [p for p in paras if re.match(_HEAD, p, re.I)]
    rest = [p for p in paras if p not in imp]
    return "\n\n".join(imp + rest)


_SETTING_SERVICE = {"LTM": "LTM/ICU", "EMU": "EMU", "Routine": "Routine", "Faulkner": "Routine", "OR": "Intraoperative"}


def setting(text: str, service=None) -> str:
    """Care setting: the EEG service field when available (I0001), else keywords in the report header."""
    if isinstance(service, str) and service in _SETTING_SERVICE:
        return _SETTING_SERVICE[service]
    head = (text or "")[:600].lower()
    if re.search(r"epilepsy monitoring unit|\bemu\b", head):
        return "EMU"
    if re.search(r"ambulatory", head):
        return "Ambulatory"
    if re.search(r"continuous|\bicu\b|\bltm\b|long[- ]term|monitoring\s+report|cceg|ceeg", head):
        return "LTM/ICU"
    if re.search(r"routine|audiovisual eeg report|portable|^\W*(?:[^\n]*\n){0,6}\s*eeg report\b", head):
        return "Routine"
    return "Unspecified"


def age_band(days) -> str:
    if days is None or (isinstance(days, float) and math.isnan(days)):
        return "Unknown"
    y = days / 365.25
    if days < 28:
        return "Neonate (<28 d)"
    if y < 1:
        return "Infant (1-12 mo)"
    if y < 12:
        return "Child (1-12 y)"
    if y < 18:
        return "Adolescent (12-18 y)"
    if y < 65:
        return "Adult (18-65 y)"
    return "Older adult (>=65 y)"


# ---- reference-label audit: is a finding mentioned only in negated form? ----
_TERMS = {"seizure": r"seizures?", "status": r"status epilepticus", "lpd": r"(?:lpds?|pleds?|lateralized periodic discharges?)",
          "gpd": r"(?:gpds?|generalized periodic discharges?)", "lrda": r"(?:lrda|lateralized rhythmic delta)",
          "grda": r"(?:grda|firda|generalized rhythmic delta)", "bs": r"burst[- ](?:suppression|attenuation)",
          "spikes": r"(?:spikes?|sharp waves?|epileptiform discharges?)",
          "foc slowing": r"focal slowing|focal delta|focal theta", "gen slowing": r"(?:generalized|diffuse) slowing"}


def mention_class(text: str, finding: str) -> str:
    """Crude, model-independent audit of how a finding is mentioned: 'negated only', 'affirmed',
    'mention without verdict' or 'no mention'. Not a reference standard."""
    term = _TERMS[finding]
    flat = " ".join((text or "").split())
    if not re.search(rf"\b{term}\b", flat, re.I):
        return "no mention"
    neg = re.compile(rf"\b(?:no|without|absence of|not)\b[^.]{{0,60}}\b{term}\b|\b{term}\s*:\s*(?:none|absent|no)\b"
                     rf"|\b{term}\b[^.]{{0,40}}\b(?:were|was)\s+not\s+(?:seen|recorded|captured|present|identified)"
                     rf"|{term} reliably detected by digital analysis:\s*absent", re.I)
    aff = re.compile(rf"\b{term}\b[^.]{{0,40}}\b(?:were|was)\s+(?:recorded|captured|seen|present|observed|identified)"
                     rf"|\b(?:\d+|one|two|three|several|multiple|frequent|rare|occasional)\s+(?:\w+\s+){{0,2}}{term}\b", re.I)
    n = bool(neg.search(flat))
    a = bool(aff.search(neg.sub(" ", flat)))  # affirmation must survive after removing negated phrases
    if a:
        return "affirmed"
    return "negated only" if n else "mention without verdict"
