# import os
# import json
# import re
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# from openai import AzureOpenAI

# # Load environment variables
# load_dotenv("../config/api_keys.env")

# client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
# )

# deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# # Predefined categories for intent classification
# INTENT_CATEGORIES = [
#     "faq",
#     "procedure",
#     "legal",
#     "contact_information",
#     "general_information",
#     "download_instruction",
#     "troubleshooting"
# ]

# def extract_text(pdf_path, pages=2):
#     reader = PdfReader(pdf_path)
#     return " ".join([p.extract_text() for p in reader.pages[:pages] if p.extract_text()])

# def classify_intent_llm(text):
#     category_list = "\n- " + "\n- ".join(INTENT_CATEGORIES)
#     prompt = (
#         f"Classify the following text into one of the predefined intent categories:{category_list}\n\n"
#         f"Text:\n{text.strip()}\n\n"
#         f"Answer (just the category name):"
#     )
    
#     response = client.chat.completions.create(
#         model=deployment_name,
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=20,
#         temperature=0
#     )
    
#     intent_raw = response.choices[0].message.content.strip().lower()
#     intent_clean = re.sub(r"[^a-z_]", "", intent_raw)

#     # fallback if model gives something unexpected
#     if intent_clean not in INTENT_CATEGORIES:
#         return "general_information"
#     return intent_clean

# def main():
#     # pdf_name = "sample.pdf"
#     pdf_name = "Class 10 Physics EM.pdf"
#     #pdf_name = "drug data sheets.pdf"
#     # pdf_name = "ieee papers 2.pdf"
#     pdf_path = f"../data/pdfs/{pdf_name}"
#     output_path = f"../data/intent_categories/{pdf_name.replace('.pdf', '.json')}"

#     text = extract_text(pdf_path)
#     intent = classify_intent_llm(text)

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     json.dump({"intent_category": intent}, open(output_path, "w"), indent=2)
#     print(f"✅ Intent category '{intent}' saved to: {output_path}")

# if __name__ == "__main__":
#     main()






import os
import json
import re
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load env variables
load_dotenv("../config/api_keys.env")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Fixed intent categories
INTENT_CATEGORIES = [
    "faq", "procedure", "legal",
    "contact_information", "general_information",
    "download_instruction", "troubleshooting"
]

FEW_SHOT_EXAMPLES = [
    {
        "text": "How do I change my email address on this platform?",
        "intent": "faq"
    },
    {
        "text": "To reset your password, go to the login page and click 'Forgot Password'.",
        "intent": "procedure"
    },
    {
        "text": "This product is governed by the licensing terms in Section 7.",
        "intent": "legal"
    },
    {
        "text": "You can reach us at support@example.com or call +1-800-123-4567.",
        "intent": "contact_information"
    },
    {
        "text": "Our platform provides a wide range of cloud storage features.",
        "intent": "general_information"
    },
    {
        "text": "Click the button below to download your invoice.",
        "intent": "download_instruction"
    },
    {
        "text": "If your device won't turn on, try holding the power button for 10 seconds.",
        "intent": "troubleshooting"
    }
]

def extract_text(pdf_path, pages=2):
    reader = PdfReader(pdf_path)
    return " ".join([p.extract_text() for p in reader.pages[:pages] if p.extract_text()])

def format_few_shot_prompt(text):
    example_block = "\n\n".join(
        [f"Text: {ex['text']}\nIntent: {ex['intent']}" for ex in FEW_SHOT_EXAMPLES]
    )
    category_list = ", ".join(INTENT_CATEGORIES)
    prompt = (
        f"{example_block}\n\n"
        f"Now classify the following text into one of these categories: {category_list}.\n"
        f"Text: {text.strip()}\nIntent:"
    )
    return prompt

def classify_intent_llm(text):
    prompt = format_few_shot_prompt(text)
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0
    )
    intent_raw = response.choices[0].message.content.strip().lower()
    intent_clean = re.sub(r"[^a-z_]", "", intent_raw)
    return intent_clean if intent_clean in INTENT_CATEGORIES else "general_information"

def main():
    # pdf_name = "sample.pdf"
    # pdf_name = "Class 10 Physics EM.pdf"
    # pdf_name = "drug data sheets.pdf"
    pdf_name = "ieee papers 2.pdf"
    pdf_path = f"../data/pdfs/{pdf_name}"
    output_path = f"../data/intent_categories/{pdf_name.replace('.pdf', 'few.json')}"

    text = extract_text(pdf_path)
    intent = classify_intent_llm(text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    json.dump({"intent_category": intent}, open(output_path, "w"), indent=2)
    print(f"✅ Few-shot intent category '{intent}' saved to: {output_path}")

if __name__ == "__main__":
    main()
