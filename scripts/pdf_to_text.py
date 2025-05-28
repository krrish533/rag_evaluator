from pathlib import Path
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    return text.strip()

def main():
    input_dir = Path("../data/pdfs")
    output_dir = Path("../data/chunks")
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_file in input_dir.glob("*.pdf"):
        text = extract_text_from_pdf(pdf_file)
        if not text:
            print(f"⚠️ {pdf_file.name} is empty or could not be read.")
            continue

        txt_path = output_dir / f"{pdf_file.stem}.txt"
        txt_path.write_text(text, encoding="utf-8")
        print(f"✅ Extracted: {pdf_file.name} ➜ {txt_path.name}")

if __name__ == "__main__":
    main()
