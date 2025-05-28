import json
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_keywords(path):
    data = json.load(open(path))
    if "keywords" in data:
        return data["keywords"]
    elif "ground_truth" in data:
        return data["ground_truth"]
    else:
        raise KeyError("Expected 'keywords' or 'ground_truth' in JSON.")

def evaluate(gt, predicted):
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform([gt])
    y_pred = mlb.transform([predicted])

    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return p, r, f1

def semantic_similarity(gt, pred):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    gt_vecs = model.encode(gt)
    pred_vecs = model.encode(pred)
    return np.mean([max(cosine_similarity([v], pred_vecs)[0]) for v in gt_vecs])

def main():
    # pdf_name = "sample.pdf"
    pdf_name = "ieee papers 2.pdf"
    gt = load_keywords(f"../data/ground_truth/{pdf_name.replace('.pdf', '.json')}")
    ml_pred = load_keywords(f"../data/keywords/ml/{pdf_name.replace('.pdf', '.json')}")
    llm_pred = load_keywords(f"../data/keywords/llm/{pdf_name.replace('.pdf', '.json')}")

    for label, pred in [("ML", ml_pred), ("LLM", llm_pred)]:
        p, r, f1 = evaluate(gt, pred)
        sim = semantic_similarity(gt, pred)
        print(f"\n📊 {label} Evaluation")
        print(f"Precision: {p:.2f}, Recall: {r:.2f}, F1 Score: {f1:.2f}")
        print(f"Semantic Similarity: {sim:.2f}")

if __name__ == "__main__":
    main()
