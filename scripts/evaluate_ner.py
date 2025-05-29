# Re-executing after kernel reset to regenerate the NER extraction and evaluation scripts
from pathlib import Path
import json
import spacy
from collections import defaultdict
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
import os

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")  # Use "en_core_web_sm" if needed

# Directories
chunks_dir = Path("../data/chunks")
output_dir = Path("../data/ner/predictions")
output_dir.mkdir(parents=True, exist_ok=True)

# --- extract_ner.py functionality ---
def extract_named_entities(text):
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

ner_results = {}

for txt_file in chunks_dir.glob("*.txt"):
    text = txt_file.read_text(encoding="utf-8").strip()
    entities = extract_named_entities(text)
    ner_results[txt_file.name] = entities
    output_path = output_dir / f"{txt_file.stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"filename": txt_file.name, "entities": entities}, f, indent=2)

# --- evaluate_ner.py functionality ---
ground_truth_path = Path("../data/ner/ground_truth_ner.json")
if not ground_truth_path.exists():
    raise FileNotFoundError("Missing ground_truth_ner.json in ../data/ner/")

with open(ground_truth_path, "r", encoding="utf-8") as f:
    ground_truth_data = json.load(f)

y_true = []
y_pred = []

missing_files = []

for filename, true_entities in ground_truth_data.items():
    pred_path = output_dir / f"{Path(filename).stem}.json"
    if not pred_path.exists():
        missing_files.append(filename)
        continue

    with open(pred_path, "r", encoding="utf-8") as f:
        pred_entities = json.load(f)["entities"]

    true_labels = [ent["label"] for ent in true_entities]
    pred_labels = [ent["label"] for ent in pred_entities]

    y_true.extend(true_labels)
    y_pred.extend(pred_labels)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

# Save evaluation metrics
eval_output = {
    "accuracy": round((sum([yt == yp for yt, yp in zip(y_true, y_pred)]) / len(y_true)) if y_true else 0, 4),
    "macro_precision": round(precision, 4),
    "macro_recall": round(recall, 4),
    "macro_f1": round(f1, 4),
    "classification_report": report,
    "missing_predictions": missing_files
}

eval_path = Path("../data/ner/evaluation_report.json")
with open(eval_path, "w", encoding="utf-8") as f:
    json.dump(eval_output, f, indent=2)

df = pd.DataFrame(report).transpose()
df.to_csv("../data/ner/evaluation_report.csv", index=True)

print("\n✅ Evaluation complete. Report saved to:")
print(" - JSON:", output_path)
print(" - CSV :", "../data/ner/evaluation_report.csv")