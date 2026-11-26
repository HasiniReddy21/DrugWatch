# backend/classify.py
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parent
EXTRACT_FILE = BASE / "dataset.json"
if not EXTRACT_FILE.exists() and (BASE / "drug_to_effects.json").exists():
    EXTRACT_FILE = BASE / "drug_to_effects.json"


# ---- STEP 1: Load sentences from the JSON file ----
def load_sentences_from_person2():
    sentences = []
    with open(EXTRACT_FILE, "r", encoding="utf8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for drug, effects_list in data.items():
            drug_name = drug.strip()
            for sentence in effects_list:
                if isinstance(sentence, str) and sentence.strip():
                    if drug_name:
                        sentences.append(f"{drug_name}: {sentence.strip()}")
                    else:
                        sentences.append(sentence.strip())
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("sentence"):
                sentences.append(item["sentence"].strip())
    return sentences


# ---- STEP 2: Demo + enriched pharmacovigilance training data ----
train_sentences = [
    "patient developed rash after aspirin",
    "subject took vitamin C",
    "he experienced nausea due to ibuprofen",
    "she is healthy and took paracetamol",
    "aspirin may cause bleeding in some patients",
    "paracetamol is usually safe, but can cause a rash",
    "ibuprofen was associated with nausea and headache",
    "patient reported severe vomiting and stomach pain following metformin",
    "hepatotoxicity was observed after high dose acetaminophen",
    "anaphylactic reaction occurred following amoxicillin therapy",
    "warfarin induced gastrointestinal bleeding and hematoma",
    "atorvastatin caused myalgia and elevated liver enzymes",
    "seizures and dizziness were reported as adverse effects of levetiracetam",
    "patient suffered urticaria and angioedema due to penicillin",
    "cisplatin therapy resulted in peripheral neuropathy and nephrotoxicity",
    "metformin led to diarrhea and abdominal discomfort",
    "lisinopril triggered a persistent dry cough and hypotension",
    "adverse drug reaction including fever and thrombocytopenia after heparin",
    "patient developed insomnia and anxiety while taking prednisone",
    "serious arrhythmia and chest pain associated with drug toxicity",
    "patient took daily aspirin for cardiovascular prevention without side effects",
    "no adverse reactions were observed during treatment",
    "the patient feels healthy and has normal blood pressure",
    "subject completed antibiotic course and recovered completely",
    "vitamin D and calcium supplements are taken every morning",
    "patient did not experience nausea, rash, or bleeding",
    "routine checkup showed normal liver and kidney function",
    "ibuprofen was prescribed for mild knee pain and worked well",
    "paracetamol safely reduced fever with no complications",
    "pharmacovigilance database collects spontaneous clinical reports",
    "study compares hospital admission statistics across regions",
    "patients receive standard care and regular follow-up visits",
    "no signs of toxicity or hypersensitivity were detected",
    "medication dosage was adjusted according to body weight",
    "the doctor recommended rest, hydration, and balanced nutrition",
]
train_labels = [
    1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]   # 1 = ADE, 0 = Non-ADE


def _augment_training_from_dataset():
    if not EXTRACT_FILE.exists():
        return
    try:
        with open(EXTRACT_FILE, "r", encoding="utf8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        added = 0
        for drug, effects in data.items():
            d = str(drug).strip()
            if not d or len(d) > 40 or not isinstance(effects, list):
                continue
            for ev in effects:
                e = str(ev).strip()
                if 4 <= len(e) <= 50:
                    train_sentences.append(f"patient experienced {e} after taking {d}")
                    train_labels.append(1)
                    train_sentences.append(f"{d} therapy was tolerated well without {e} or adverse reactions")
                    train_labels.append(0)
                    train_sentences.append(f"no {e} or adverse events were observed after {d}")
                    train_labels.append(0)
                    added += 1
                    break
            if added >= 35:
                break
    except Exception:
        pass


_augment_training_from_dataset()


# ---- STEP 3: Train classifier ----
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
X_train = vectorizer.fit_transform(train_sentences)

clf = LogisticRegression(C=3.0, class_weight="balanced", max_iter=1000)
clf.fit(X_train, train_labels)


def classify_sentence(text: str):
    """Classify a sentence as ADE or Non-ADE"""
    if not text or not text.strip():
        return "Non-ADE", 0.5
    X = vectorizer.transform([text])
    label = clf.predict(X)[0]
    confidence = float(clf.predict_proba(X).max())
    return "ADE" if label == 1 else "Non-ADE", round(confidence, 3)


# ---- STEP 4: Example usage ----
if __name__ == "__main__":
    sentences = load_sentences_from_person2()
    # For demonstration, we'll only classify the first 10 sentences
    # to provide a concise output.
    for s in sentences[:10]:
        label, conf = classify_sentence(s)
        print(f"{s}  --> {label} (conf={conf})")
