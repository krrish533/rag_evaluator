# import json
# from pathlib import Path
# from sentence_transformers import SentenceTransformer, util

# def load_intent_examples(json_path):
#     with open(json_path, "r", encoding="utf-8") as f:
#         return json.load(f)

# def load_document_text(txt_path):
#     try:
#         return Path(txt_path).read_text(encoding="utf-8").strip()
#     except Exception as e:
#         print(f"❌ Failed to read {txt_path.name}: {e}")
#         return None

# def predict_intent(doc_text, intent_examples, model):
#     # Prepare data
#     intent_labels = list(intent_examples.keys())
#     example_texts = list(intent_examples.values())

#     # Embed document and example intents
#     doc_embedding = model.encode(doc_text, convert_to_tensor=True)
#     example_embeddings = model.encode(example_texts, convert_to_tensor=True)

#     # Compute cosine similarities
#     cosine_scores = util.cos_sim(doc_embedding, example_embeddings)[0]

#     # Select best intent
#     best_idx = int(cosine_scores.argmax())
#     best_score = float(cosine_scores[best_idx])
#     best_intent = intent_labels[best_idx]

#     # Debug output
#     print(f"\n📄 Document intent prediction:")
#     for idx, label in enumerate(intent_labels):
#         print(f"  - {label:25s}: Score = {cosine_scores[idx]:.4f}")
#     print(f"✅ Predicted Intent: '{best_intent}' (Confidence: {best_score:.4f})")

#     return best_intent, best_score

# def main():
#     chunks_dir = Path("../data/chunks")
#     intent_json_path = Path("../data/intent_examples.json")
#     model = SentenceTransformer("sentence-transformers/nli-mpnet-base-v2")

#     intent_examples = load_intent_examples(intent_json_path)

#     for txt_file in chunks_dir.glob("*.txt"):
#         doc_text = load_document_text(txt_file)
#         if not doc_text:
#             print(f"⚠️ Skipping {txt_file.name} — could not load text.")
#             continue

#         print(f"\n🔍 Processing {txt_file.name}...")
#         predict_intent(doc_text, intent_examples, model)

# if __name__ == "__main__":
#     main()

# ----------------------------------------------------------------------------

# import json
# from pathlib import Path
# from sentence_transformers import SentenceTransformer, util

# def load_intent_examples(json_path):
#     with open(json_path, "r", encoding="utf-8") as f:
#         return json.load(f)

# def load_document_text(txt_path):
#     try:
#         return Path(txt_path).read_text(encoding="utf-8").strip()
#     except Exception as e:
#         print(f"❌ Failed to read {txt_path.name}: {e}")
#         return None

# def predict_intent(doc_text, intent_examples, model):
#     # Prepare data
#     intent_labels = list(intent_examples.keys())
#     example_texts = list(intent_examples.values())

#     # Embed document and example intents
#     doc_embedding = model.encode(doc_text, convert_to_tensor=True)
#     example_embeddings = model.encode(example_texts, convert_to_tensor=True)

#     # Compute cosine similarities
#     cosine_scores = util.cos_sim(doc_embedding, example_embeddings)[0]

#     # Select best intent
#     best_idx = int(cosine_scores.argmax())
#     best_score = float(cosine_scores[best_idx])
#     best_intent = intent_labels[best_idx]

#     # Detailed score output
#     score_dict = {label: float(cosine_scores[i]) for i, label in enumerate(intent_labels)}

#     return best_intent, best_score, score_dict

# def main():
#     chunks_dir = Path("../data/chunks")
#     intent_json_path = Path("../data/intent_examples.json")
#     output_dir = Path("../data/intent/mpnet")
#     output_dir.mkdir(parents=True, exist_ok=True)

#     model = SentenceTransformer("sentence-transformers/nli-mpnet-base-v2")
#     intent_examples = load_intent_examples(intent_json_path)

#     for txt_file in chunks_dir.glob("*.txt"):
#         doc_text = load_document_text(txt_file)
#         if not doc_text:
#             print(f"⚠️ Skipping {txt_file.name} — could not load text.")
#             continue

#         print(f"\n🔍 Processing {txt_file.name}...")
#         best_intent, best_score, score_details = predict_intent(doc_text, intent_examples, model)

#         # Print results
#         print(f"📄 Intent prediction for {txt_file.name}:")
#         for label, score in score_details.items():
#             print(f"  - {label:25s}: Score = {score:.4f}")
#         print(f"✅ Predicted Intent: '{best_intent}' (Confidence: {best_score:.4f})")

#         # Save result
#         result = {
#             "filename": txt_file.name,
#             "predicted_intent": best_intent,
#             "confidence": best_score,
#             "all_scores": score_details
#         }
#         output_path = output_dir / f"{txt_file.stem}.json"
#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(result, f, indent=2)

# if __name__ == "__main__":
#     main()


# ----------------------------------------------------------------------

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

def load_intent_examples(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_document_text(txt_path):
    try:
        return Path(txt_path).read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"❌ Failed to read {txt_path.name}: {e}")
        return None

def predict_intent(doc_text, intent_examples, model):
    intent_labels = list(intent_examples.keys())
    example_texts = list(intent_examples.values())

    doc_embedding = model.encode(doc_text, convert_to_tensor=True)
    example_embeddings = model.encode(example_texts, convert_to_tensor=True)

    cosine_scores = util.cos_sim(doc_embedding, example_embeddings)[0]

    best_idx = int(cosine_scores.argmax())
    best_score = float(cosine_scores[best_idx])
    best_intent = intent_labels[best_idx]

    print(f"\n📄 Document intent prediction:")
    for idx, label in enumerate(intent_labels):
        print(f"  - {label:25s}: Score = {cosine_scores[idx]:.4f}")
    print(f"✅ Predicted Intent: '{best_intent}' (Confidence: {best_score:.4f})")

    return best_intent, best_score

def main():
    chunks_dir = Path("../data/chunks")
    output_dir = Path("../data/intent/mpnet")
    intent_json_path = Path("../data/intent_examples.json")
    model = SentenceTransformer("sentence-transformers/nli-mpnet-base-v2")

    output_dir.mkdir(parents=True, exist_ok=True)
    intent_examples = load_intent_examples(intent_json_path)

    for txt_file in chunks_dir.glob("*.txt"):
        doc_text = load_document_text(txt_file)
        if not doc_text:
            print(f"⚠️ Skipping {txt_file.name} — could not load text.")
            continue

        print(f"\n🔍 Processing {txt_file.name}...")
        predicted_intent, confidence = predict_intent(doc_text, intent_examples, model)

        result = {
            "filename": txt_file.name,
            "predicted_intent": predicted_intent,
            "confidence_score": round(confidence, 4)
        }

        output_path = output_dir / f"{txt_file.stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()

