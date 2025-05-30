# import os
# import json
# from pathlib import Path
# import re
# from PyPDF2 import PdfReader
# from sentence_transformers import SentenceTransformer, util
# import spacy
# from keybert import KeyBERT
# from openai import AzureOpenAI
# from dotenv import load_dotenv

# # Load environment variables
# #load_dotenv(dotenv_path="../config/api_keys.env")
# load_dotenv(dotenv_path="config/api_keys.env")

# # Azure OpenAI credentials
# AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
# AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# # Initialize Azure OpenAI client
# client = AzureOpenAI(
#     api_key=AZURE_API_KEY,
#     api_version=AZURE_API_VERSION,
#     azure_endpoint=AZURE_ENDPOINT
# )

# # Define all processor classes
# class KeywordExtractor:
#     def __init__(self):
#         self.model = KeyBERT()

#     def extract(self, text):
#         cleaned = self.clean_text(text)
#         keywords = [kw for kw, _ in self.model.extract_keywords(
#             cleaned,
#             top_n=10,
#             stop_words='english',
#             keyphrase_ngram_range=(1, 1)
#         )]
#         return keywords

#     @staticmethod
#     def clean_text(text):
#         text = re.sub(r'\n+', ' ', text)
#         text = re.sub(r'[^\w\s\-]', '', text)
#         text = re.sub(r'\s{2,}', ' ', text)
#         return text.strip().lower()


# class IntentPredictor:
#     def __init__(self, example_path):
#         # self.model = SentenceTransformer("sentence-transformers/nli-mpnet-base-v2")
#         self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
#         with open(example_path, "r", encoding="utf-8") as f:
#             self.intent_examples = json.load(f)

#     def predict(self, text):
#         labels = list(self.intent_examples.keys())
#         examples = list(self.intent_examples.values())

#         doc_emb = self.model.encode(text, convert_to_tensor=True)
#         example_emb = self.model.encode(examples, convert_to_tensor=True)
#         cosine_scores = util.cos_sim(doc_emb, example_emb)[0]

#         best_idx = int(cosine_scores.argmax())
#         return labels[best_idx], float(cosine_scores[best_idx])


# class NERExtractor:
#     def __init__(self):
#         self.model = spacy.load("en_core_web_sm")

#     def extract(self, text):
#         doc = self.model(text)
#         return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]


# class Summarizer:
#     def __init__(self):
#         self.client = client

#     def summarize(self, text):
#         prompt = f"Summarize the following text in 2–3 lines:\n\n{text[:3000]}"
#         response = self.client.chat.completions.create(
#             model=AZURE_DEPLOYMENT,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.5,
#             max_tokens=256
#         )
#         return response.choices[0].message.content.strip()


# class DocumentProcessor:
#     def __init__(self, chunks_dir, output_dir, intent_example_path):
#         self.chunks_dir = Path(chunks_dir)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         self.keyword_extractor = KeywordExtractor()
#         self.intent_predictor = IntentPredictor(intent_example_path)
#         self.ner_extractor = NERExtractor()
#         self.summarizer = Summarizer()

#     def process_all(self):
#         for txt_file in self.chunks_dir.glob("*.txt"):
#             text = txt_file.read_text(encoding="utf-8").strip()
#             if not text:
#                 print(f"⚠️ Skipping {txt_file.name} — empty text.")
#                 continue

#             print(f"🔍 Processing: {txt_file.name}")
#             result = {
#                 "filename": txt_file.name,
#                 "keywords": self.keyword_extractor.extract(text),
#                 "predicted_intent": {},
#                 "named_entities": self.ner_extractor.extract(text),
#                 "summary": self.summarizer.summarize(text)
#             }

#             intent, confidence = self.intent_predictor.predict(text)
#             result["predicted_intent"] = {
#                 "intent": intent,
#                 "confidence_score": round(confidence, 4)
#             }

#             with open(self.output_dir / f"{txt_file.stem}.json", "w", encoding="utf-8") as f:
#                 json.dump(result, f, indent=2)

#             print(f"✅ Saved combined metadata: {txt_file.stem}.json")


# # Instantiate and run
# processor = DocumentProcessor(
#     chunks_dir="data/chunks",
#     # output_dir="data/metadata/mpnet",
#     output_dir="data/metadata/all_mpnet",
#     intent_example_path="data/intent_examples.json"
# )
# processor.process_all()



import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from openai import AzureOpenAI

# Load environment variables
load_dotenv(dotenv_path="config/api_keys.env")

# Azure OpenAI credentials
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

class MetadataExtractionPipeline:
    def __init__(self, chunks_dir, output_dir, intent_example_path):
        self.chunks_dir = Path(chunks_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.keyword_model = KeyBERT()
        self.intent_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.ner_model = spacy.load("en_core_web_sm")

        with open(intent_example_path, "r", encoding="utf-8") as f:
            self.intent_examples = json.load(f)

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'[^\w\s\-]', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip().lower()

    def extract_keywords(self, text):
        cleaned = self.clean_text(text)
        return [kw for kw, _ in self.keyword_model.extract_keywords(
            cleaned,
            top_n=10,
            stop_words='english',
            keyphrase_ngram_range=(1, 1)
        )]

    def predict_intent(self, text):
        labels = list(self.intent_examples.keys())
        examples = list(self.intent_examples.values())

        doc_emb = self.intent_model.encode(text, convert_to_tensor=True)
        example_emb = self.intent_model.encode(examples, convert_to_tensor=True)
        cosine_scores = util.cos_sim(doc_emb, example_emb)[0]

        best_idx = int(cosine_scores.argmax())
        return labels[best_idx], float(cosine_scores[best_idx])

    def extract_named_entities(self, text):
        doc = self.ner_model(text)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    def generate_summary(self, text):
        prompt = f"Summarize the following text in 2–3 lines:\n\n{text[:3000]}"
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()

    def encode_text(self, text):
        return self.intent_model.encode(text).tolist()

    def process_all_documents(self):
        for txt_file in self.chunks_dir.glob("*.txt"):
            text = txt_file.read_text(encoding="utf-8").strip()
            if not text:
                print(f"⚠️ Skipping {txt_file.name} — empty text.")
                continue

            print(f"🔍 Processing: {txt_file.name}")
            keywords = self.extract_keywords(text)
            intent, confidence = self.predict_intent(text)
            entities = self.extract_named_entities(text)
            summary = self.generate_summary(text)
            embedding = self.encode_text(text)

            metadata = {
                "filename": txt_file.name,
                "length": len(text),
                "context": text,
                "keywords": keywords,
                "intent_category": intent,
                # "confidence_score": round(confidence, 4),
                "named_entities": entities,
                "summary": summary,
                "embedding": embedding
            }

            with open(self.output_dir / f"{txt_file.stem}.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            print(f"✅ Saved metadata: {txt_file.stem}.json")

# Execute the pipeline
pipeline = MetadataExtractionPipeline(
    chunks_dir="data/chunks",
    output_dir="data/metadata/final_allmpnet",
    intent_example_path="data/intent_examples.json"
)
pipeline.process_all_documents()
