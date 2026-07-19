# Evaluates the trained model on the held-out test set and saves reports/charts to outputs/.

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

MODELS_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load model artifacts
print("Loading model artifacts...")

best_model = joblib.load(MODELS_DIR / "best_model.pkl")
vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
X_test, y_test = joblib.load(MODELS_DIR / "test_data.pkl")

print(f"  Model type  : {type(best_model).__name__}")
print(f"  Test samples: {len(X_test):,}")
print(f"  Classes     : {list(best_model.classes_)}")

# Transform test data (never fit on test set)
print("\nTransforming test data...")
X_test_tfidf = vectorizer.transform(X_test)
print(f"  X_test_tfidf shape: {X_test_tfidf.shape}")

# Compute metrics for visualization
print("\nEvaluation Results (Test Set)")

y_pred    = best_model.predict(X_test_tfidf)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")

# Classification report
print("\nClassification Report:")
report = classification_report(y_test, y_pred, zero_division=0)
print(report)
# Save classification report
report_path = OUTPUT_DIR / "classification_report.txt"
with open(report_path, "w") as f:
    f.write("AI Resume Screening — Classification Report\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1 Score  : {f1:.4f}\n\n")
    f.write("Per-Class Report:\n")
    f.write(report)
print(f"Saved: {report_path}")

# Confusion matrix for visualization
print("\nGenerating confusion matrix...")

cm   = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap="Blues", ax=ax, colorbar=False)
plt.title("Confusion Matrix — Test Set", fontsize=14, pad=15)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()
print(f"Saved: outputs/confusion_matrix.png")

# Accuracy comparison chart (reads model_comparison.csv from train_model.py)
comparison_csv = OUTPUT_DIR / "model_comparison.csv"
if comparison_csv.exists():
    print("\nGenerating accuracy comparison chart...")
    comparison_df = pd.read_csv(comparison_csv)

    colors = ["#4C72B0", "#55A868", "#C44E52"]  # matches model order: LR, DT, RF
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(comparison_df["Model"], comparison_df["Accuracy"], color=colors, width=0.5)

    # Add accuracy value labels above each bar
    for bar, val in zip(bars, comparison_df["Accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title("Model Accuracy Comparison (Validation Set)", fontsize=14, pad=15)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Model")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=150)
    plt.close()
    print("Saved: outputs/accuracy_comparison.png")
else:
    print(f"Warning: {comparison_csv} not found. Run train_model.py first.")

# Feature importance (tree models only)
if hasattr(best_model, "feature_importances_"):
    print("\nGenerating feature importance chart...")

    feature_names = vectorizer.get_feature_names_out()
    importances   = best_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances,
    })

    top_20 = importance_df.sort_values("Importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_20["Feature"], top_20["Importance"], color="steelblue")
    ax.invert_yaxis()  # highest importance at top
    ax.set_title("Top 20 Important Features (Random Forest)", fontsize=14, pad=15)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature (Word)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print("Saved: outputs/feature_importance.png")
else:
    print("Model does not expose feature_importances_. Skipping feature importance chart.")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value":  [accuracy,   precision,   recall,   f1],
})
metrics_df.to_csv(OUTPUT_DIR / "evaluation_metrics.csv", index=False)
print("Saved: outputs/evaluation_metrics.csv")

print("\nEvaluation completed successfully!")

