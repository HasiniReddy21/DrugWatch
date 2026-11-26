# backend/extract.py
import json
import re
from pathlib import Path
import spacy

BASE = Path(__file__).resolve().parent
CLEAN_FILE = BASE / "data" / "cleaned" / "sentences.json"
DATASET_FILE = BASE / "dataset.json"
OUT_FILE = BASE / "results" / "extractions.json"

# Try SciSpacy biomedical models first (if installed) else fallback.
MODEL_CANDIDATES = ["en_ner_bionlp13cg_md", "en_ner_bc5cdr_md", "en_core_sci_sm"]
nlp = None
for m in MODEL_CANDIDATES:
    try:
        nlp = spacy.load(m)
        print("Using model:", m)
        break
    except Exception:
        nlp = None

if nlp is None:
    nlp = spacy.load("en_core_web_sm")
    print("Using fallback model: en_core_web_sm")

# small dictionary fallback to catch common drug names/events quickly
DRUG_DICT = {
    "aspirin", "paracetamol", "acetaminophen", "ibuprofen", "metformin",
    "amoxicillin", "warfarin", "atorvastatin", "simvastatin", "lisinopril",
    "omeprazole", "prednisone", "insulin", "methotrexate", "cisplatin",
    "lithium", "carbamazepine", "valproic acid", "levetiracetam",
    "lamotrigine", "oxcarbazepine", "naproxen", "diclofenac", "clopidogrel",
    "azithromycin", "ciprofloxacin", "doxycycline", "fluoxetine", "sertraline",
    "vitamin c", "penicillin", "heparin", "morphine", "codeine", "furosemide"
}
EVENT_DICT = {
    "bleeding", "rash", "nausea", "headache", "pain", "stomach pain",
    "abdominal pain", "chest pain", "vomiting", "dizziness", "fever",
    "diarrhea", "diarrhoea", "fatigue", "hepatotoxicity", "anaphylaxis",
    "anaphylactic reaction", "urticaria", "seizure", "seizures", "edema",
    "oedema", "insomnia", "constipation", "cough", "arrhythmia",
    "hypertension", "hypotension", "neuropathy", "anemia", "anaemia",
    "thrombocytopenia", "nephrotoxicity", "confusion", "allergic reaction",
    "itching", "pruritus", "shortness of breath", "palpitations", "alopecia",
    "hypoglycemia", "bradycardia", "tachycardia", "jaundice", "tremor",
    "angioedema", "asthma", "hepatitis", "haemorrhage", "hemorrhage",
    "gastrointestinal bleeding", "pneumonitis", "mucositis", "myalgia",
    "depression", "anxiety", "blurred vision", "weight gain", "weight loss"
}

_STOPWORDS_NOISE = {
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "with",
    "patient", "a patient", "patients", "case", "study", "drug", "drugs",
    "therapy", "treatment", "agent", "agents", "oral", "high", "low",
    "favorable outcome", "a favorable outcome", "none", "unknown", "other",
    "disease", "syndrome", "reaction", "disorder", "toxicity", "effect",
    "effects", "symptom", "symptoms", "clinical", "normal", "severe",
    "acute", "chronic", "mild", "moderate", "use", "dose", "dosage",
    "two", "one", "three", "five", "all", "no", "not", "yes",
    "adr", "adrs", "problem", "problems", "outcome", "outcomes",
    "safety", "report", "reports", "side effect", "side effects",
    "adverse effect", "adverse effects", "adverse event", "adverse events",
    "adverse reaction", "adverse reactions", "adverse drug reaction",
    "adverse drug reactions", "complication", "complications", "allergy"
}


def _clean_term(term: str) -> str:
    term = term.strip().lower()
    term = re.sub(r"^[\(\[\"\']+|[\.\,\;\:\!\?\)\]\'\"]+$", "", term).strip()
    term = re.sub(r"^(?:a|an|the)\s+", "", term).strip()
    return term


def _load_dataset_terms():
    if not DATASET_FILE.exists():
        return
    try:
        with open(DATASET_FILE, "r", encoding="utf8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        for drug, events in data.items():
            d = _clean_term(str(drug))
            if (
                len(d) >= 3
                and len(d) <= 45
                and len(d.split()) <= 4
                and d not in _STOPWORDS_NOISE
                and re.search(r"[a-z]", d)
            ):
                DRUG_DICT.add(d)
            if isinstance(events, list):
                for ev in events:
                    e = _clean_term(str(ev))
                    if (
                        len(e) >= 4
                        and len(e) <= 40
                        and len(e.split()) <= 3
                        and " and " not in e
                        and " or " not in e
                        and e not in _STOPWORDS_NOISE
                        and re.search(r"[a-z]", e)
                    ):
                        EVENT_DICT.add(e)
    except Exception:
        pass


_load_dataset_terms()

# Remove overlaps where a term exists in both sets (prefer specific drugs in DRUG_DICT)
EVENT_DICT -= DRUG_DICT

_DRUG_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(t) for t in sorted(DRUG_DICT, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)
_EVENT_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(t) for t in sorted(EVENT_DICT, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)


def map_ent_label(label: str):
    label = label.upper()
    if label in {"CHEMICAL", "DRUG", "CHEM"}:
        return "drug"
    if label in {"DISEASE", "SYMPTOM", "CONDITION"}:
        return "event"
    return None


def _dedupe_preserve(items):
    seen = set()
    out = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def extract_spans(text: str):
    """Return non-overlapping entity spans [{text, label, start, end}] for UI highlighting."""
    if not text or not text.strip():
        return []

    candidates = []
    doc = nlp(text)
    for ent in doc.ents:
        mapped = map_ent_label(ent.label_)
        if mapped:
            candidates.append({
                "text": ent.text,
                "label": mapped,
                "start": ent.start_char,
                "end": ent.end_char,
            })

    for match in _DRUG_PATTERN.finditer(text):
        candidates.append({
            "text": match.group(0),
            "label": "drug",
            "start": match.start(),
            "end": match.end(),
        })

    for match in _EVENT_PATTERN.finditer(text):
        candidates.append({
            "text": match.group(0),
            "label": "event",
            "start": match.start(),
            "end": match.end(),
        })

    # Prefer longer spans first, then earlier start
    candidates.sort(key=lambda s: (-(s["end"] - s["start"]), s["start"]))
    selected = []
    for cand in candidates:
        overlaps = any(
            not (cand["end"] <= s["start"] or cand["start"] >= s["end"])
            for s in selected
        )
        if not overlaps:
            selected.append(cand)

    selected.sort(key=lambda s: s["start"])
    return selected


def extract_from_sentence(sent: str):
    spans = extract_spans(sent)
    drugs = _dedupe_preserve(s["text"] for s in spans if s["label"] == "drug")
    events = _dedupe_preserve(s["text"] for s in spans if s["label"] == "event")
    return drugs, events


def run_extraction(text: str):
    """
    Extract drug-ADE pairs from arbitrary text (single or multi-sentence).
    Returns list of {"drug": ..., "event": ...}.
    """
    if not text or not text.strip():
        return []

    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()] or [text.strip()]

    pairs = []
    seen_pairs = set()
    all_drugs = []
    all_events = []

    for sent in sentences:
        drugs, events = extract_from_sentence(sent)
        all_drugs.extend(drugs)
        all_events.extend(events)
        for d in drugs:
            for e in events:
                if (d, e) not in seen_pairs:
                    seen_pairs.add((d, e))
                    pairs.append({"drug": d, "event": e})

    # Fallback: if drug and event appear in adjacent/separate sentences of the same note
    if not pairs:
        for d in _dedupe_preserve(all_drugs):
            for e in _dedupe_preserve(all_events):
                if (d, e) not in seen_pairs:
                    seen_pairs.add((d, e))
                    pairs.append({"drug": d, "event": e})

    return pairs


if __name__ == "__main__":
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(open(CLEAN_FILE, "r", encoding="utf8").read())
    out = []
    for r in records:
        sent = r["sentence"]
        drugs, events = extract_from_sentence(sent)
        if drugs or events:
            out.append({
                "file": r.get("file"),
                "sent_idx": r.get("sent_idx"),
                "sentence": sent,
                "drugs": drugs,
                "events": events,
            })
    with open(OUT_FILE, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2)
    print("Saved", OUT_FILE)
