import os, json
from PyPDF2 import PdfReader
from keybert import KeyBERT

def extract_text(pdf_path, pages=3):
    reader = PdfReader(pdf_path)
    return " ".join([p.extract_text() for p in reader.pages[:pages] if p.extract_text()])

def generate_keywords(text, top_n=10):
    kw_model = KeyBERT()
    return [kw for kw, _ in kw_model.extract_keywords(text, top_n=top_n, stop_words='english')]

def main():
    # pdf_name = "sample.pdf"
    # pdf_name = "Class 10 Physics EM.pdf"
    # pdf_name = "drug data sheets.pdf"
    pdf_name = "ieee papers 2.pdf"
    pdf_path = f"../data/pdfs/{pdf_name}"
    output_path = f"../data/ground_truth/{pdf_name.replace('.pdf', '.json')}"

    text = extract_text(pdf_path)
    keywords = generate_keywords(text, top_n=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    json.dump({"ground_truth": keywords}, open(output_path, "w"), indent=2)
    print(f"✅ Ground truth saved: {output_path}")

if __name__ == "__main__":
    main()
