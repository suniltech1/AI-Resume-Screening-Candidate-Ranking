
#import libraries
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

# Paths
DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Load Training Pairs
print("=" * 60)
print("Loading training pairs...")
print("=" * 60)

pairs_df = pd.read_csv(DATA_DIR / "training_pairs.csv")
print(f"Training pairs loaded: {len(pairs_df):,} rows")
print("\nLabel distribution:")
print(pairs_df["Suitability"].value_counts())

# X: the combined resume + job description text
# y: the suitability label (Highly Suitable / Suitable / Not Suitable)
X = pairs_df["Combined_Text"]
y = pairs_df["Suitability"]

# Train / Validation / Test Split (60 / 20 / 20)
# stratify=y ensures each split has the same class proportions
print("\n" + "=" * 60)
print("Splitting dataset: 60% train / 20% val / 20% test ...")
print("=" * 60)

# First split: 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

# Second split: split temp 50/50 into validation and test (20% each)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"  Train      : {len(X_train):,} samples")
print(f"  Validation : {len(X_val):,} samples")
print(f"  Test       : {len(X_test):,} samples")

# TF-IDF Vectorization
#
# IMPORTANT: fit_transform() is called ONLY on X_train.
# Calling fit() on validation or test data would cause data leakage —
# the model would indirectly "see" test words during training.
# For validation and test, we use transform() only.
print("\n" + "=" * 60)
print("Fitting TF-IDF vectorizer on training data only...")
print("=" * 60)

vectorizer = TfidfVectorizer(max_features=5000)

# fit_transform on training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# transform only on validation and test data
X_val_tfidf  = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print(f"  TF-IDF vocabulary size : {len(vectorizer.vocabulary_):,} features")
print(f"  X_train_tfidf shape    : {X_train_tfidf.shape}")
print(f"  X_val_tfidf shape      : {X_val_tfidf.shape}")
print(f"  X_test_tfidf shape     : {X_test_tfidf.shape}")


# Helper: Evaluate a model on the validation set
def evaluate_on_validation(model, X_val_tfidf, y_val):
    """
    Predict on the validation set and return a dictionary of metrics.
    Uses weighted averaging so all class sizes are accounted for.
    """
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
print("\n" + "=" * 60)
print("Training Logistic Regression...")
print("=" * 60)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_metrics = evaluate_on_validation(lr_model, X_val_tfidf, y_val)
print(f"  Accuracy  : {lr_metrics['Accuracy']:.4f}")
print(f"  F1 Score  : {lr_metrics['F1 Score']:.4f}")

# Train Decision Tree
print("\n" + "=" * 60)
print("Training Decision Tree...")
print("=" * 60)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_tfidf, y_train)
dt_metrics = evaluate_on_validation(dt_model, X_val_tfidf, y_val)
print(f"  Accuracy  : {dt_metrics['Accuracy']:.4f}")
print(f"  F1 Score  : {dt_metrics['F1 Score']:.4f}")

# Train Random Forest (baseline, before tuning)
print("\n" + "=" * 60)
print("Training Random Forest...")
print("=" * 60)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_tfidf, y_train)
rf_metrics = evaluate_on_validation(rf_model, X_val_tfidf, y_val)
print(f"  Accuracy  : {rf_metrics['Accuracy']:.4f}")
print(f"  F1 Score  : {rf_metrics['F1 Score']:.4f}")

# Save Model Comparison Table (validation results for all 3 baseline models)
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

# Hyperparameter Tuning: GridSearchCV on Random Forest
#
# GridSearchCV tries every combination of parameters in param_grid and
# uses 5-fold cross-validation to find the best configuration.
# n_jobs=-1 uses all available CPU cores to speed up the search.
print("\n" + "=" * 60)
print("Performing GridSearchCV on Random Forest (this may take a few minutes)...")
print("=" * 60)

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
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train_tfidf, y_train)

best_model  = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate the tuned model on the validation set
best_val_metrics = evaluate_on_validation(best_model, X_val_tfidf, y_val)

print(f"\nBest hyperparameters found:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print(f"\nTuned model validation accuracy : {best_val_metrics['Accuracy']:.4f}")
print(f"Tuned model validation F1 Score : {best_val_metrics['F1 Score']:.4f}")


# Save All Artifacts
print("\n" + "=" * 60)
print("Saving model artifacts...")
print("=" * 60)

# Save the best trained model
joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
print("Saved: models/best_model.pkl")

# Save the fitted TF-IDF vectorizer (required for inference in app.py)
joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
print("Saved: models/tfidf_vectorizer.pkl")

# Save the data splits for use in evaluate.py
joblib.dump((X_train, y_train), MODELS_DIR / "train_data.pkl")
joblib.dump((X_val,   y_val),   MODELS_DIR / "validation_data.pkl")
joblib.dump((X_test,  y_test),  MODELS_DIR / "test_data.pkl")
print("Saved: models/train_data.pkl")
print("Saved: models/validation_data.pkl")
print("Saved: models/test_data.pkl")

print("\nTraining completed successfully!")
print("Next step: python evaluate.py")
