import os, json
from PyPDF2 import PdfReader
from keybert import KeyBERT


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    return " ".join([p.extract_text() for p in reader.pages if p.extract_text()])

def main():
    pdf_name = "sample.pdf"
    pdf_path = f"../data/pdfs/{pdf_name}"
    output_path = f"../data/keywords/ml/{pdf_name.replace('.pdf', '.json')}"
    
    text = extract_text(pdf_path)
    kw_model = KeyBERT()
    #keywords = [kw for kw, _ in kw_model.extract_keywords(text, top_n=10, stop_words='english')]
    keywords = [kw for kw, _ in kw_model.extract_keywords(
    clean_text(text), 
    top_n=10, 
    stop_words='english', 
    use_maxsum=False, 
    use_mmr=False, 
    diversity=0.5, 
    keyphrase_ngram_range=(1, 1)  # ⬅️ single word only
)]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    json.dump({"keywords": keywords}, open(output_path, "w"), indent=2)
    print(f"✅ ML keywords saved: {output_path}")

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)               # Remove newlines
    text = re.sub(r'[^\w\s\-]', '', text)          # Remove punctuation
    text = re.sub(r'\s{2,}', ' ', text)            # Remove extra spaces
    return text.strip().lower()

if __name__ == "__main__":
    main()
