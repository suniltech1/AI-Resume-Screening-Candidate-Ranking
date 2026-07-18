# 🤖 AI Resume Screening and Candidate Ranking System

> An intelligent resume screening system that uses Machine Learning to  evaluate and rank candidates based on their fit for a selected job posting.

**Student:** Sunil Kumar Khadka | **Student ID:** 250123

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Technologies Used](#technologies-used)

---

## Overview

The **AI Resume Screening and Candidate Ranking System** automates the initial stage of hiring by analyzing uploaded resumes against a chosen job posting. It leverages TF-IDF vectorization and a tuned Random Forest classifier to predict whether a candidate is:

| Label | Cosine Similarity Threshold |
|---|---|
| ✅ Highly Suitable | ≥ 70% |
| 🟡 Suitable | 40% – 69% |
| ❌ Not Suitable | < 40% |

The system provides both a **model confidence score** and a **cosine similarity score** so recruiters can make informed, data-backed decisions.

---

## Features

- 📄 **PDF & TXT Resume Upload** — Supports both file formats for maximum flexibility.
- 🔍 **Job Posting Selection** — Search and select any job role from the dataset.
- 🧠 **ML-Powered Prediction** — Tuned Random Forest classifier predicts candidate suitability.
- 📊 **Dual Scoring** — Reports both model confidence (%) and cosine similarity (%) for transparency.
- 📥 **Downloadable Report** — Export evaluation results as a CSV file.
- 🌐 **Interactive Web App** — Clean Streamlit UI accessible in any browser.

---

## Project Structure

`
AI-Resume-Screening-Candidate-Ranking/
│
├── data/
│   ├── UpdatedResumeDataSet.csv        # Raw resume dataset
│   ├── job_postings.csv                # Raw job postings dataset
│   ├── clean_resume_dataset.csv        # Cleaned resumes (generated)
│   └── clean_job_dataset.csv           # Cleaned job postings (generated)
│
├── models/
│   ├── best_model.pkl                  # Trained & tuned Random Forest model
│   ├── tfidf_vectorizer.pkl            # Fitted TF-IDF vectorizer
│   ├── train_data.pkl                  # Training split
│   ├── validation_data.pkl             # Validation split
│   └── test_data.pkl                   # Test split
│
├── outputs/
│   ├── model_comparison.csv            # Accuracy comparison of all models
│   ├── evaluation_metrics.csv          # Final test set metrics
│   ├── classification_report.txt       # Detailed per-class report
│   ├── confusion_matrix.png            # Confusion matrix plot
│   ├── accuracy_comparison.png         # Bar chart of model accuracies
│   └── feature_importance.png          # Top 20 TF-IDF features chart
│
├── app.py                              # Streamlit web application
├── preprocessing.py                    # Data cleaning & preparation
├── train_model.py                      # Model training & hyperparameter tuning
├── evaluate.py                         # Model evaluation & report generation
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
`

---

## How It Works

### 1. Preprocessing (preprocessing.py)
- Loads raw resume and job posting CSV datasets.
- Cleans text: lowercasing, removing punctuation, numbers, and NLTK stopwords.
- Combines job title, description, skills, and experience level into a single text field.
- Saves cleaned datasets to data/.

### 2. Training (	rain_model.py)
- Generates **cosine-similarity-based suitability labels** using a temporary TF-IDF vectorizer.
- Splits data into **60% train / 20% validation / 20% test**.
- Fits a final TF-IDF vectorizer *only* on training data to prevent data leakage.
- Trains three candidate models: Logistic Regression, Decision Tree, and Random Forest.
- Runs **GridSearchCV** (5-fold cross-validation) to tune the best Random Forest.
- Saves the best model and all artifacts to models/.

### 3. Evaluation (evaluate.py)
- Loads the saved model and held-out test data.
- Generates accuracy, precision, recall, F1-score, and a full classification report.
- Saves a confusion matrix, accuracy comparison chart, and feature importance plot to outputs/.

### 4. Web App (pp.py)
- Provides an interactive Streamlit interface for end-users.
- User selects a job role → uploads a resume (PDF/TXT) → receives an instant suitability prediction with scores.

---

## Dataset

| Dataset | Description |
|---|---|
| UpdatedResumeDataSet.csv | Labelled resumes across multiple job categories |
| job_postings.csv | Real-world job postings with titles, descriptions, skills & experience levels |

---

## Model Performance

### Validation Set Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 57.58% | 47.66% | 57.58% | 48.87% |
| Decision Tree | 51.52% | 45.89% | 51.52% | 48.48% |
| **Random Forest** | **60.61%** | **52.04%** | **60.61%** | **50.87%** |

> ✅ **Random Forest** achieved the best overall performance and was selected as the final model. It was further tuned using GridSearchCV with 5-fold cross-validation before deployment.

### GridSearchCV Hyperparameter Grid

`python
param_grid = {
    n_estimators:      [100, 200],
    max_depth:         [10, 20, None],
    min_samples_split: [2, 5],
    min_samples_leaf:  [1, 2]
}
`

### Suitability Label Thresholds (Cosine Similarity)

`python
def create_suitability_label(score):
    if score >= 0.70:
        return Highly Suitable
    elif score >= 0.40:
        return Suitable
    else:
        return Not Suitable
`

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Steps

**1. Clone the repository**
`ash
git clone https://github.com/your-username/AI-Resume-Screening-Candidate-Ranking.git
cd AI-Resume-Screening-Candidate-Ranking
`

**2. Create and activate a virtual environment**
`ash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
`

**3. Install dependencies**
`ash
pip install -r requirements.txt
`

---

## Usage

Run the scripts **in order** if starting from scratch. Skip steps 1–3 if the models/ folder already contains trained artifacts.

### Step 1 — Preprocess Data
`ash
python preprocessing.py
`
*Outputs: data/clean_resume_dataset.csv, data/clean_job_dataset.csv*

### Step 2 — Train the Model
`ash
python train_model.py
`
*Outputs: models/best_model.pkl, models/tfidf_vectorizer.pkl, outputs/model_comparison.csv*

### Step 3 — Evaluate the Model
`ash
python evaluate.py
`
*Outputs: Confusion matrix, accuracy chart, feature importance chart, classification report in outputs/*

### Step 4 — Launch the Web App
`ash
streamlit run app.py
`
Open **http://localhost:8501** in your browser.

---

## Pipeline

`
Raw Data
   │
   ▼
preprocessing.py  ──►  clean_resume_dataset.csv
                  ──►  clean_job_dataset.csv
   │
   ▼
train_model.py    ──►  best_model.pkl
                  ──►  tfidf_vectorizer.pkl
                  ──►  model_comparison.csv
   │
   ▼
evaluate.py       ──►  confusion_matrix.png
                  ──►  accuracy_comparison.png
                  ──►  feature_importance.png
                  ──►  classification_report.txt
   │
   ▼
app.py (Streamlit) ──►  Interactive Web Interface
`

---

## Technologies Used

| Library | Purpose |
|---|---|
| streamlit | Web application framework |
| scikit-learn | ML models, TF-IDF, GridSearchCV, metrics |
| pandas | Data loading and manipulation |
| 
umpy | Numerical operations |
| 
ltk | Stopword removal during preprocessing |
| pdfplumber | PDF text extraction in the web app |
| joblib | Model serialization (save/load .pkl files) |
| matplotlib | Evaluation charts and plots |

---

## License

This project was developed as an academic submission. All rights reserved by the author.

---

*Built with by Sunil Kumar Khadka (Student ID: 250123)*
