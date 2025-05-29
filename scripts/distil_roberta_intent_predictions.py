import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_predictions(pred_dir):
    preds = {}
    for file in Path(pred_dir).glob("*.json"):
        data = json.load(open(file, "r", encoding="utf-8"))
        preds[data["filename"]] = data["predicted_intent"]
    return preds

def main():
    ground_truth_path = Path("../data/ground_truth/ground_truth_intent.json")
    prediction_dir = Path("../data/intent/distil_roberta")
    output_path = Path("../data/evaluations/distil_roberta_intent_classification_report.json")
    

    ground_truth = load_ground_truth(ground_truth_path)
    predictions = load_predictions(prediction_dir)

    y_true, y_pred = [], []
    unmatched_files = []

    for doc_name, true_intent in ground_truth.items():
        pred_intent = predictions.get(doc_name)
        if pred_intent:
            y_true.append(true_intent)
            y_pred.append(pred_intent)
        else:
            unmatched_files.append(doc_name)

    accuracy = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()

    # 📄 Save evaluation report
    report = {
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "missing_predictions": unmatched_files
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 🖨️ Print summary
    print("🎯 Accuracy:", round(accuracy, 4))
    print("\n📊 Classification Report (macro avg):")
    print(json.dumps(class_report.get("macro avg", {}), indent=2))

    if unmatched_files:
        print("\n⚠️ Missing predictions for:", unmatched_files)

if __name__ == "__main__":
    main()