import json
from collections import Counter
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from extract import (
    nlp,
    run_extraction,
    extract_from_sentence,
    extract_spans,
)
from classifier import classify_sentence

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATASET_FILE = BASE_DIR / "dataset.json"
EXTRACTIONS_FILE = BASE_DIR / "results" / "extractions.json"
FALLBACK_EXTRACTION_FILE = BASE_DIR / "extraction.json"

CAUSAL_CUES = (
    "caused", "causing", "cause", "associated with", "associated",
    "led to", "resulted in", "due to", "after", "following",
    "induced", "secondary to", "related to", "triggered", "developed",
    "experienced", "reported", "suffered"
)

# -------------------------------
# FastAPI app setup
# -------------------------------

app = FastAPI(title="DrugWatch Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class TextInput(BaseModel):
    """Input schema for /extract and /classify endpoints."""
    text: str


@app.get("/")
def serve_ui():
    """Serve the DrugWatch web UI."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "DrugWatch API is running. Visit /docs for API documentation."}


@app.get("/healthcheck")
def healthcheck():
    """
    Simple healthcheck endpoint.
    Used to verify that the backend is running.
    """
    return {"status": "ok"}


@app.post("/extract")
def extract_entities(input: TextInput):
    """
    Calls extraction pipeline.
    Input: { "text": "..." }
    Output: { "extractions": [ {drug, event}, ... ], "drugs": [...], "events": [...], "spans": [...] }
    """
    results = run_extraction(input.text)
    drugs, events = extract_from_sentence(input.text)
    spans = extract_spans(input.text)
    lower_text = input.text.lower()
    has_causal_cue = any(cue in lower_text for cue in CAUSAL_CUES)

    enriched_extractions = [
        {
            "drug": pair["drug"],
            "event": pair["event"],
            "relation": "adverse_effect" if has_causal_cue else "co_occurrence",
        }
        for pair in results
    ]

    return {
        "extractions": enriched_extractions,
        "drugs": drugs,
        "events": events,
        "spans": spans,
    }


@app.post("/classify")
def classify_text(input: TextInput):
    """
    Calls classifier.
    Input: { "text": "..." }
    Output: { "label": "ADE/Non-ADE", "confidence": float, "sentences": [...] }
    """
    text = (input.text or "").strip()
    if not text:
        return {"label": "Non-ADE", "confidence": 0.5, "sentences": []}

    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()] or [text]

    sentence_results = []
    for sent in sents:
        lbl, conf = classify_sentence(sent)
        sentence_results.append({
            "sentence": sent,
            "label": lbl,
            "confidence": conf,
        })

    if len(sentence_results) == 1:
        overall_label = sentence_results[0]["label"]
        overall_conf = sentence_results[0]["confidence"]
    else:
        ade_sents = [s for s in sentence_results if s["label"] == "ADE"]
        if ade_sents:
            overall_label = "ADE"
            overall_conf = round(max(s["confidence"] for s in ade_sents), 3)
        else:
            overall_label = "Non-ADE"
            overall_conf = round(
                sum(s["confidence"] for s in sentence_results) / len(sentence_results),
                3,
            )

    return {
        "label": overall_label,
        "confidence": overall_conf,
        "sentences": sentence_results,
    }


@app.get("/stats")
def get_stats():
    """
    Compute pharmacovigilance statistics from dataset.json and extraction results:
    - Frequency of ADEs
    - Top drugs with ADEs
    - Corpus extraction records
    """
    drug_counts = Counter()
    event_counts = Counter()
    drug_sample_events = {}
    total_ade_mentions = 0

    if DATASET_FILE.exists():
        with open(DATASET_FILE, "r", encoding="utf8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for drug, events in data.items():
                d_name = str(drug).strip()
                if not d_name or not isinstance(events, list):
                    continue
                cleaned_events = [str(e).strip() for e in events if str(e).strip()]
                if not cleaned_events:
                    continue
                drug_counts[d_name] += len(cleaned_events)
                drug_sample_events[d_name] = cleaned_events[:4]
                for ev in cleaned_events:
                    event_counts[ev.lower()] += 1
                    total_ade_mentions += 1

    corpus_extractions = []
    source_file = EXTRACTIONS_FILE if EXTRACTIONS_FILE.exists() else FALLBACK_EXTRACTION_FILE
    if source_file.exists():
        try:
            with open(source_file, "r", encoding="utf8") as f:
                raw_ext = json.load(f)
            if isinstance(raw_ext, list):
                corpus_extractions = raw_ext
        except Exception:
            corpus_extractions = []

    top_drugs = [
        {
            "drug": drug,
            "count": count,
            "sample_events": drug_sample_events.get(drug, []),
        }
        for drug, count in drug_counts.most_common(12)
    ]

    top_events = [
        {"event": event, "count": count}
        for event, count in event_counts.most_common(12)
    ]

    return {
        "summary": {
            "total_drugs": len(drug_counts),
            "total_ade_mentions": total_ade_mentions,
            "unique_ades": len(event_counts),
            "corpus_sentences_extracted": len(corpus_extractions),
        },
        "top_drugs": top_drugs,
        "top_events": top_events,
        "corpus_extractions": corpus_extractions[:25],
    }
