import os
import json
import re
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import AzureOpenAI


# Load Azure OpenAI config from .env
load_dotenv("../config/api_keys.env")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    return " ".join([p.extract_text() for p in reader.pages[:2] if p.extract_text()])

def get_keywords_from_llm(text):
    # prompt = f"Extract 8 relevant keywords from the following text:\n\n{text}"
    prompt = (
    "From the following text, extract **only 8 single keywords** or topics.\n"
    "Return them as a comma-separated list — no bullets, no numbers, no markdown, no explanation.\n\n"
    f"{text}"
    )
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.5
    )
    #return response.choices[0].message.content.strip().split(", ")
    
    keywords_raw = response.choices[0].message.content.strip()

    # Clean markdown, bullets, and split by commas
    cleaned_keywords = re.sub(r"[\*\n\d\.]", "", keywords_raw)  # remove markdown & bullets
    keywords = [kw.strip() for kw in cleaned_keywords.split(",") if kw.strip()]
    return keywords



# def main():
#     # pdf_name = "sample.pdf"
#     # pdf_name = "Class 10 Physics EM.pdf"
#     # pdf_name = "drug data sheets.pdf"
#     pdf_name = "ieee papers 2.pdf"
#     pdf_path = f"../data/pdfs/{pdf_name}"
#     output_path = f"../data/keywords/llm/{pdf_name.replace('.pdf', '.json')}"

#     text = extract_text(pdf_path)
#     keywords = get_keywords_from_llm(text)

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     json.dump({"keywords": keywords}, open(output_path, "w"), indent=2)
#     print(f"✅ Azure LLM keywords saved: {output_path}")

# if __name__ == "__main__":
#     main()


from pathlib import Path

def main():
    pdf_folder = Path("../data/pdfs")
    output_folder = Path("../data/keywords/llm")
    os.makedirs(output_folder, exist_ok=True)

    for pdf_file in pdf_folder.glob("*.pdf"):
        print(f"📄 Processing: {pdf_file.name}")
        
        text = extract_text(pdf_file)
        if not text:
            print(f"⚠️ Skipping {pdf_file.name} — no text found")
            continue

        keywords = get_keywords_from_llm(text)

        output_path = output_folder / f"{pdf_file.stem}.json"
        json.dump({"keywords": keywords}, open(output_path, "w"), indent=2)
        print(f"✅ Saved: {output_path.name}")

if __name__ == "__main__":
    main()

