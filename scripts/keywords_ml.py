import os, json, re
from pathlib import Path
from PyPDF2 import PdfReader
from keybert import KeyBERT

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    return " ".join([p.extract_text() for p in reader.pages if p.extract_text()])

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)               # Remove newlines
    text = re.sub(r'[^\w\s\-]', '', text)          # Remove punctuation
    text = re.sub(r'\s{2,}', ' ', text)            # Remove extra spaces
    return text.strip().lower()

def main():
    pdf_folder = Path("../data/pdfs")
    output_folder = Path("../data/keywords/ml")
    kw_model = KeyBERT()

    for pdf_file in pdf_folder.glob("*.pdf"):
        text = extract_text(pdf_file)
        if not text:
            print(f"⚠️ Skipping {pdf_file.name} (no text found)")
            continue

        cleaned = clean_text(text)

        keywords = [kw for kw, _ in kw_model.extract_keywords(
            cleaned,
            top_n=10,
            stop_words='english',
            use_maxsum=False,
            use_mmr=False,
            diversity=0.5,
            keyphrase_ngram_range=(1, 1)  # single word only
        )]

        # Build output path and write JSON
        output_path = output_folder / f"{pdf_file.stem}.json"
        os.makedirs(output_folder, exist_ok=True)
        json.dump({"keywords": keywords}, open(output_path, "w"), indent=2)
        print(f"✅ Saved: {output_path.name}")

if __name__ == "__main__":
    main()
