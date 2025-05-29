import json
from pathlib import Path
import spacy

# Load NER model
nlp = spacy.load("en_core_web_sm")  # or "en_core_web_sm"

def extract_entities(text):
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

def main():
    chunks_dir = Path("../data/chunks")
    output_dir = Path("../data/ner/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in chunks_dir.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8").strip()
        entities = extract_entities(text)

        result = {
            "filename": txt_file.name,
            "entities": entities
        }

        out_path = output_dir / f"{txt_file.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
