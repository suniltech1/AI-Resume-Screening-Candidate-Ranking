
#import necessary libraries

import re
import string
import streamlit as st
import pandas as pd
import joblib
import pdfplumber
import nltk
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# NLTK Setup
# Download stopwords silently on first run.

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")
DATA_DIR   = Path("data")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Project Information")
st.sidebar.markdown("---")
st.sidebar.write("**Project Title:**")
st.sidebar.write("AI Resume Screening and Candidate Ranking System")
st.sidebar.markdown("---")
st.sidebar.write("**Student Name:**")
st.sidebar.write("Sunil Kumar Khadka")
st.sidebar.write("**Student ID:**")
st.sidebar.write("250123")
st.sidebar.markdown("---")
st.sidebar.write("**Model:**")
st.sidebar.write("Random Forest (GridSearchCV tuned)")
st.sidebar.write("**Features:**")
st.sidebar.write("TF-IDF on Resume + Job Description")


# ---------------------------------------------------------------------------
# Text Cleaning Function
#
# CRITICAL: This function must be IDENTICAL to clean_text() in preprocessing.py.
#
# The TF-IDF vectorizer was fitted on text produced by that function.
# Any difference here shifts the feature space and breaks predictions.
#
# Cleaning steps (in the same order as preprocessing.py):
#   1. Handle NaN / empty string
#   2. Lowercase
#   3. Remove numbers (delete entirely, NOT replaced with a space)
#   4. Remove punctuation using string.punctuation
#   5. Collapse whitespace
#   6. Remove NLTK English stopwords
# ---------------------------------------------------------------------------
def clean_text(text):
    """
    Clean input text to match the training pipeline exactly.
    Returns a cleaned, space-separated string of meaningful words.
    """
    if pd.isna(text) or str(text).strip() == "":
        return ""

    # Step 1: Lowercase
    text = str(text).lower()

    # Step 2: Remove numbers entirely
    text = re.sub(r"\d+", "", text)

    # Step 3: Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 4: Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Step 5: Remove stopwords
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]

    return " ".join(words)


# ---------------------------------------------------------------------------
# PDF / TXT Text Extraction
# ---------------------------------------------------------------------------
def extract_text_from_file(uploaded_file):
    """
    Extract raw text from a PDF or TXT file.
    Returns (text, error_message).
    On success, error_message is None.
    """
    resume_text = ""
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        resume_text += page_text + "\n"
        elif uploaded_file.name.lower().endswith(".txt"):
            resume_text = uploaded_file.getvalue().decode("utf-8")
        else:
            return "", "Unsupported file type. Please upload a PDF or TXT file."
    except Exception as e:
        return "", f"Could not read file: {e}"

    return resume_text, None


# ---------------------------------------------------------------------------
# Load Model Resources (cached)
#
# @st.cache_resource prevents reloading large .pkl files on every
# Streamlit interaction. Resources are loaded once per session.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    """
    Load the trained model, TF-IDF vectorizer, and cleaned job dataset.
    Returns (model, vectorizer, job_dataframe, error_message).
    error_message is None on success.
    """
    model_path      = MODELS_DIR / "best_model.pkl"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    job_data_path   = DATA_DIR   / "clean_job_dataset.csv"

    # Check all required files exist before loading
    missing = []
    for path in [model_path, vectorizer_path, job_data_path]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        return None, None, None, "Missing files: " + ", ".join(missing)

    try:
        model      = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        job_df     = pd.read_csv(job_data_path)
        return model, vectorizer, job_df, None
    except Exception as e:
        return None, None, None, f"Failed to load resources: {e}"


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
st.title("AI Resume Screening and Candidate Ranking System")
st.write(
    "Select a job posting, upload a resume, and the system will predict "
    "whether the candidate is **Highly Suitable**, **Suitable**, or "
    "**Not Suitable** for that specific role."
)
st.markdown("---")

# Load all required resources
best_model, tfidf_vectorizer, job_df, load_error = load_resources()

# Stop early if any file is missing or failed to load
if load_error:
    st.error(f"Startup Error: {load_error}")
    st.info(
        "Run the full pipeline first:\n\n"
        "```\n"
        "python preprocessing.py\n"
        "python train_model.py\n"
        "python evaluate.py\n"
        "```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Step 1: Job Selection
#
# The job_df is loaded from clean_job_dataset.csv, which contains the
# pre-cleaned job descriptions produced by preprocessing.py.
# We use Clean_Job directly — no additional cleaning is applied.
# ---------------------------------------------------------------------------
st.subheader("1. Select a Job Posting")

job_titles = job_df["Display_Name"].tolist()

selected_title = st.selectbox(
    "Choose a job role (type to search):",
    options=job_titles,
    index=None,
    placeholder="Search for a job role...",
)

if not selected_title:
    st.info("Please select a job role from the dropdown to continue.")
    st.stop()

# Retrieve the pre-cleaned job description (already processed by preprocessing.py)
selected_job_row  = job_df[job_df["Display_Name"] == selected_title].iloc[0]
clean_job_text    = str(selected_job_row["Clean_Job"])

# Guard against empty job descriptions in the CSV
if not clean_job_text.strip():
    st.error("The selected job posting has an empty description. Please choose another role.")
    st.stop()

with st.expander("View Job Description (cleaned text)"):
    st.write(clean_job_text)

st.markdown("---")

# ---------------------------------------------------------------------------
# Step 2: Resume Upload
# ---------------------------------------------------------------------------
st.subheader("2. Upload Resume")

uploaded_file = st.file_uploader(
    "Upload a candidate resume (PDF or TXT):",
    type=["pdf", "txt"],
)

if uploaded_file is None:
    st.info("Please upload a resume to see the evaluation.")
    st.stop()

# Extract raw text from the uploaded file
raw_resume_text, extract_error = extract_text_from_file(uploaded_file)

if extract_error:
    st.error(f"File Error: {extract_error}")
    st.stop()

if not raw_resume_text.strip():
    st.error(
        "The uploaded file appears to be empty or contains no readable text. "
        "If this is a scanned PDF, please use a text-layer PDF instead."
    )
    st.stop()

with st.expander("Preview Uploaded Resume (raw text)"):
    preview = raw_resume_text[:3000]
    st.text(preview + ("..." if len(raw_resume_text) > 3000 else ""))

# ---------------------------------------------------------------------------
# Step 3: Text Preprocessing
#
# Apply the SAME clean_text() used during training (defined above).
# This ensures the resume tokens match the TF-IDF vocabulary.
# ---------------------------------------------------------------------------
cleaned_resume = clean_text(raw_resume_text)

if not cleaned_resume.strip():
    st.error(
        "After cleaning, the resume text is empty. "
        "The file may contain only numbers, punctuation, or common stopwords."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Step 4: Combine Resume + Job Description
#
# This is the critical step that enables multi-job generalisation.
# The model was trained on Combined_Text = cleaned_resume + " " + clean_job.
# We replicate that EXACT format here.
# ---------------------------------------------------------------------------
combined_text = cleaned_resume + " " + clean_job_text

# ---------------------------------------------------------------------------
# Step 5: TF-IDF Vectorization
#
# ONLY transform() is called — never fit() or fit_transform().
# The vectorizer vocabulary was fixed during training in train_model.py.
# ---------------------------------------------------------------------------
combined_vector = tfidf_vectorizer.transform([combined_text])

# Also transform resume and job separately for cosine similarity display
resume_vector = tfidf_vectorizer.transform([cleaned_resume])
job_vector    = tfidf_vectorizer.transform([clean_job_text])

# ---------------------------------------------------------------------------
# Step 6: Prediction
# ---------------------------------------------------------------------------
prediction    = best_model.predict(combined_vector)[0]
probabilities = best_model.predict_proba(combined_vector)[0]
class_labels  = list(best_model.classes_)
prob_dict     = dict(zip(class_labels, probabilities))
confidence    = prob_dict[prediction]

# ---------------------------------------------------------------------------
# Step 7: Cosine Similarity (informational display only)
#
# Computed between the resume vector and job vector using the trained
# TF-IDF vocabulary. NOT used for prediction — the model decides that.
# ---------------------------------------------------------------------------
similarity_score = cosine_similarity(resume_vector, job_vector)[0][0]

# ---------------------------------------------------------------------------
# Step 8: Display Results
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("3. Evaluation Results")

# Colour-coded prediction banner
if prediction == "Highly Suitable":
    st.success(f"Prediction:  {prediction}")
elif prediction == "Suitable":
    st.warning(f"Prediction:  {prediction}")
else:
    st.error(f"Prediction:  {prediction}")

# Recommendation text
recommendation_map = {
    "Highly Suitable": (
        "This candidate strongly matches the job requirements. "
        "Recommend proceeding to the interview stage."
    ),
    "Suitable": (
        "This candidate meets the basic requirements. "
        "Consider reviewing their profile for further assessment."
    ),
    "Not Suitable": (
        "This candidate does not closely match the job requirements. "
        "Consider other candidates or rescreen with a different role."
    ),
}
st.info(f"Recommendation: {recommendation_map[prediction]}")

# Metric columns
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Model Confidence",
        value=f"{confidence * 100:.2f}%",
        help="Probability the Random Forest assigns to the predicted class.",
    )

with col2:
    st.metric(
        label="Resume-Job Similarity",
        value=f"{similarity_score * 100:.2f}%",
        help="Cosine similarity between resume and job description (informational only).",
    )
    st.progress(float(min(similarity_score, 1.0)))

# ---------------------------------------------------------------------------
# Debug Panel (expandable)
# ---------------------------------------------------------------------------
with st.expander("Debug Information"):
    st.write("**Raw resume word count:**", len(raw_resume_text.split()))
    st.write("**Cleaned resume word count:**", len(cleaned_resume.split()))
    st.write("**Cleaned resume preview (first 300 chars):**")
    st.code(cleaned_resume[:300])
    st.write("**Combined text word count:**", len(combined_text.split()))
    st.write("**Combined vector shape:**", combined_vector.shape)
    st.write("**Non-zero TF-IDF features:**", combined_vector.nnz)
    st.write("**Prediction probabilities per class:**")
    prob_df = pd.DataFrame({
        "Class":          class_labels,
        "Probability (%)": [f"{prob_dict[c] * 100:.2f}" for c in class_labels],
    })
    st.table(prob_df)
    st.write(f"**Cosine similarity score:** {similarity_score:.6f}")

# ---------------------------------------------------------------------------
# Step 9: Export Report
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("4. Export Report")

report_data = {
    "Candidate File":         [uploaded_file.name],
    "Selected Job":           [selected_title],
    "Prediction":             [prediction],
    "Confidence (%)":         [f"{confidence * 100:.2f}"],
    "Cosine Similarity (%)":  [f"{similarity_score * 100:.2f}"],
}

# Add per-class probability columns
for label in class_labels:
    report_data[f"P({label}) (%)"] = [f"{prob_dict[label] * 100:.2f}"]

report_df  = pd.DataFrame(report_data)
csv_report = report_df.to_csv(index=False)

st.download_button(
    label="Download Results as CSV",
    data=csv_report,
    file_name="candidate_evaluation_report.csv",
    mime="text/csv",
)
