import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Load training pairs
print("Loading training pairs...")

pairs_df = pd.read_csv(DATA_DIR / "training_pairs.csv")
print(f"Training pairs loaded: {len(pairs_df):,} rows")
print("\nLabel distribution:")
print(pairs_df["Suitability"].value_counts())

X = pairs_df["Combined_Text"]  # combined resume + job text
y = pairs_df["Suitability"]    # label: Highly Suitable / Suitable / Not Suitable

# Train / Validation / Test split (60 / 20 / 20), stratified by class
print("\nSplitting dataset: 60% train / 20% val / 20% test ...")

# First split: 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

# Second split: 50/50 temp into val and test (20% each)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"  Train      : {len(X_train):,} samples")
print(f"  Validation : {len(X_val):,} samples")
print(f"  Test       : {len(X_test):,} samples")

# Fit TF-IDF on training data only (transform only on val/test to prevent leakage)
print("\nFitting TF-IDF vectorizer on training data...")

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"  TF-IDF vocabulary size : {len(vectorizer.vocabulary_):,} features")
print(f"  X_train_tfidf shape    : {X_train_tfidf.shape}")
print(f"  X_val_tfidf shape      : {X_val_tfidf.shape}")
print(f"  X_test_tfidf shape     : {X_test_tfidf.shape}")


def evaluate_on_validation(model, X_val_tfidf, y_val):
    """Evaluates a model on the validation set and returns weighted metrics."""
    y_pred    = model.predict(X_val_tfidf)
    accuracy  = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_val, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    return {
        "Accuracy":  accuracy,
        "Precision": precision,
        "Recall":    recall,
        "F1 Score":  f1,
    }


# Train Logistic Regression
print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_metrics = evaluate_on_validation(lr_model, X_val_tfidf, y_val)
print(f"  Accuracy  : {lr_metrics['Accuracy']:.4f}")
print(f"  F1 Score  : {lr_metrics['F1 Score']:.4f}")

# Train Decision Tree
print("\nTraining Decision Tree...")

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_tfidf, y_train)
dt_metrics = evaluate_on_validation(dt_model, X_val_tfidf, y_val)
print(f"  Accuracy  : {dt_metrics['Accuracy']:.4f}")
print(f"  F1 Score  : {dt_metrics['F1 Score']:.4f}")

# Train Random Forest (baseline)
print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_tfidf, y_train)
rf_metrics = evaluate_on_validation(rf_model, X_val_tfidf, y_val)
print(f"  Accuracy  : {rf_metrics['Accuracy']:.4f}")
print(f"  F1 Score  : {rf_metrics['F1 Score']:.4f}")

# Save baseline comparison table
comparison_df = pd.DataFrame({
    "Model":     ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy":  [lr_metrics["Accuracy"],  dt_metrics["Accuracy"],  rf_metrics["Accuracy"]],
    "Precision": [lr_metrics["Precision"], dt_metrics["Precision"], rf_metrics["Precision"]],
    "Recall":    [lr_metrics["Recall"],    dt_metrics["Recall"],    rf_metrics["Recall"]],
    "F1 Score":  [lr_metrics["F1 Score"],  dt_metrics["F1 Score"],  rf_metrics["F1 Score"]],
})

comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
print("\nModel comparison (validation set):")
print(comparison_df.to_string(index=False))
print("Saved: outputs/model_comparison.csv")

# Hyperparameter tuning with GridSearchCV on Random Forest (5-fold CV, all CPU cores)
print("\nPerforming GridSearchCV on Random Forest (this may take a few minutes)...")

# Search space for Random Forest hyperparameters
param_grid = {
    "n_estimators":      [100, 200],
    "max_depth":         [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,    # use all CPU cores
    verbose=1,
)

grid_search.fit(X_train_tfidf, y_train)

best_model  = grid_search.best_estimator_
best_params = grid_search.best_params_

best_val_metrics = evaluate_on_validation(best_model, X_val_tfidf, y_val)

print(f"\nBest hyperparameters found:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print(f"\nTuned model validation accuracy : {best_val_metrics['Accuracy']:.4f}")
print(f"Tuned model validation F1 Score : {best_val_metrics['F1 Score']:.4f}")


# Save all artifacts
print("\nSaving model artifacts...")

joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
print("Saved: models/best_model.pkl")

joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
print("Saved: models/tfidf_vectorizer.pkl")

# Raw text splits are saved so evaluate.py can vectorize them with the fitted TF-IDF
joblib.dump((X_train, y_train), MODELS_DIR / "train_data.pkl")
joblib.dump((X_val,   y_val),   MODELS_DIR / "validation_data.pkl")
joblib.dump((X_test,  y_test),  MODELS_DIR / "test_data.pkl")
print("Saved: models/train_data.pkl")
print("Saved: models/validation_data.pkl")
print("Saved: models/test_data.pkl")

print("\nTraining completed successfully!")
print("Next step: python evaluate.py")
