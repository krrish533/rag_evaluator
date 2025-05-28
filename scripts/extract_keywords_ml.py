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
    keywords = [kw for kw, _ in kw_model.extract_keywords(text, top_n=10, stop_words='english')]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    json.dump({"keywords": keywords}, open(output_path, "w"), indent=2)
    print(f"✅ ML keywords saved: {output_path}")

if __name__ == "__main__":
    main()
