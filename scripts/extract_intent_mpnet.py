import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# Load the NLI-based MPNet model
model = SentenceTransformer("sentence-transformers/nli-mpnet-base-v2")

# Few-shot intent category examples
intent_examples = {
    "faq": "How can I reset my password or get help with logging in?",
    "procedure": "Follow these instructions to complete the installation process.",
    "legal": "This document outlines the legal guidelines and compliance requirements.",
    "contact_information": "You can reach customer support via email or phone contact details.",
    "general_info": "This document contains general product or service information.",
    "policy": "The following rules define our privacy, refund, or security policies.",
    "guide": "This is a step-by-step instructional guide for users to follow."
}

def get_intent_for_text(text: str) -> str:
    """Find the closest matching intent based on cosine similarity."""
    doc_embedding = model.encode(text, convert_to_tensor=True)
    intent_texts = list(intent_examples.values())
    intent_embeddings = model.encode(intent_texts, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(doc_embedding, intent_embeddings)
    best_idx = scores.argmax().item()
    return list(intent_examples.keys())[best_idx]

def main():
    input_dir = Path("../data/pdfs")              # Input PDFs
    text_dir = Path("../data/chunks")             # Pre-extracted text files
    output_dir = Path("../data/intent/mpnet")     # Intent output
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_file in input_dir.glob("*.pdf"):
        text_file = text_dir / f"{pdf_file.stem}.txt"
        if not text_file.exists():
            print(f"⚠️ Skipping {pdf_file.name} — Missing extracted text at {text_file}")
            continue

        # Load extracted plain text
        text = text_file.read_text(encoding="utf-8")
        predicted_intent = get_intent_for_text(text)

        # Save the intent result
        output_path = output_dir / f"{pdf_file.stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"intent": predicted_intent}, f, indent=2)

        print(f"✅ {pdf_file.name} ➜ intent: {predicted_intent}")

if __name__ == "__main__":
    main()
