import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from classifier import classify_sentence as _classify_sentence, vectorizer, clf

BASE = Path(__file__).resolve().parent
EXTRACT_FILE = BASE / "dataset.json"


def load_sentences_from_person2():
    with open(EXTRACT_FILE, "r", encoding="utf8") as f:
        data = json.load(f)
    sentences = []
    if isinstance(data, dict):
        for drug, effects in data.items():
            d = str(drug).strip()
            if isinstance(effects, list):
                for ev in effects:
                    e = str(ev).strip()
                    if e:
                        sentences.append(f"{d} {e}".strip())
    elif isinstance(data, list):
        for item in data:
            drugs = " ".join(item.get("drugs", []))
            events = " ".join(item.get("events", []))
            combined = f"{item.get('sentence', '')} {drugs} {events}".strip()
            if combined:
                sentences.append(combined)
    return sentences


def classify_sentence(text: str):
    return _classify_sentence(text)


if __name__ == "__main__":
    sentences = load_sentences_from_person2()
    for s in sentences[:15]:
        label, conf = classify_sentence(s)
        print(f"{s}  --> {label} (conf={conf})")
