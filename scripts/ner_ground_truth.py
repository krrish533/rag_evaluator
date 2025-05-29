from pathlib import Path
import json
import spacy

# Load SpaCy's transformer-based NER model (make sure you have it installed via `python -m spacy download en_core_web_trf`)
try:
    nlp = spacy.load("en_core_web_trf")
except:
    nlp = spacy.load("en_core_web_sm")  # fallback to small model

# Paths
chunks_dir = Path("../data/chunks")
output_path = Path("../data/ner/ground_truth_ner.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Output dictionary to store NER annotations
ground_truth_ner = {}

# Iterate through each .txt file and extract entities
for txt_file in chunks_dir.glob("*.txt"):
    text = txt_file.read_text(encoding="utf-8").strip()
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})

    ground_truth_ner[txt_file.name] = entities

# Save as JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(ground_truth_ner, f, indent=2)

output_path
