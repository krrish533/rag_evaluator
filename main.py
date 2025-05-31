import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from openai import AzureOpenAI
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 🔐 Load API Keys
load_dotenv(dotenv_path="config/api_keys.env")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# 🤖 Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

# 🧠 Embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# 🏷️ NER model
ner_tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
ner_model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")


class MetadataExtractionPipeline:
    def __init__(self, chunks_dir, output_dir, intent_example_path):
        self.chunks_dir = Path(chunks_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.intent_examples = self.load_intent_examples(intent_example_path)
        self.keyword_model = KeyBERT()

    @staticmethod
    def load_intent_examples(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def clean_text(text):
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'[^\w\s\-]', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip().lower()

    def extract_keywords(self, text):
        cleaned = self.clean_text(text)
        keywords = [kw for kw, _ in self.keyword_model.extract_keywords(
            cleaned,
            top_n=10,
            stop_words='english',
            keyphrase_ngram_range=(1, 1)
        )]
        return keywords

    def predict_intent(self, text):
        labels = list(self.intent_examples.keys())
        examples = list(self.intent_examples.values())

        doc_emb = embedding_model.encode(text, convert_to_tensor=True)
        example_emb = embedding_model.encode(examples, convert_to_tensor=True)
        cosine_scores = util.cos_sim(doc_emb, example_emb)[0]

        best_idx = int(cosine_scores.argmax())
        return labels[best_idx], float(cosine_scores[best_idx])

    def extract_entities(self, text):
        return [{"text": ent["word"], "label": ent["entity_group"]} for ent in ner_pipeline(text)]

    def summarize(self, text):
        prompt = f"Summarize the following text in 2–3 lines:\n\n{text[:3000]}"
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()

    def embed_text(self, text):
        return embedding_model.encode(text).tolist()

    def process_all(self):
        for txt_file in self.chunks_dir.glob("*.txt"):
            text = txt_file.read_text(encoding="utf-8").strip()
            if not text:
                print(f"⚠️ Skipping {txt_file.name} — empty text.")
                continue

            print(f"🔍 Processing: {txt_file.name}")
            intent, confidence = self.predict_intent(text)
            result = {
                "filename": txt_file.name,
                "context": text,
                "keywords": self.extract_keywords(text),
                "intent_category": intent,
                # "confidence_score": round(confidence, 4),
                "named_entities": self.extract_entities(text),
                "summary": self.summarize(text),
                "embedding": self.embed_text(text)
            }

            with open(self.output_dir / f"{txt_file.stem}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"✅ Saved metadata: {txt_file.stem}.json")


# 🚀 Run the pipeline
if __name__ == "__main__":
    pipeline = MetadataExtractionPipeline(
        chunks_dir="data/chunks",
        output_dir="data/metadata/final",
        intent_example_path="data/intent_examples.json"
    )
    pipeline.process_all()
