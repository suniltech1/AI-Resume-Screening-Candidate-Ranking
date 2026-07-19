import re
import string
import streamlit as st
import pandas as pd
import joblib
import pdfplumber
import nltk
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords on first run

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))  # set for O(1) lookup

st.set_page_config(
    page_title="AI Resume Screening System",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = Path("models")
DATA_DIR   = Path("data")

# Sidebar — project info
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
st.sidebar.write("Random Forest")
st.sidebar.write("**Features:**")
st.sidebar.write("TF-IDF on Resume + Job Description")


def clean_text(text):
    """Clean text — must stay identical to preprocessing.py to avoid feature space drift."""
    if pd.isna(text) or str(text).strip() == "":
        return ""

    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


def extract_text_from_file(uploaded_file):
    """Extracts text from a PDF or TXT upload. Returns (text, error)."""
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


@st.cache_resource
def load_resources():
    """Loads model, vectorizer, and job dataset once per session."""
    model_path      = MODELS_DIR / "best_model.pkl"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    job_data_path   = DATA_DIR   / "clean_job_dataset.csv"

    missing = [str(p) for p in [model_path, vectorizer_path, job_data_path] if not p.exists()]
    if missing:
        return None, None, None, "Missing files: " + ", ".join(missing)

    try:
        model      = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        job_df     = pd.read_csv(job_data_path)
        return model, vectorizer, job_df, None
    except Exception as e:
        return None, None, None, f"Failed to load resources: {e}"

# Heading of the app
st.title("Resume Screening and Candidate Ranking System")
st.write(
    "Select a job posting, upload a resume, and the system will predict "
    "whether the candidate is **Highly Suitable**, **Suitable**, or "
    "**Not Suitable** for that specific role."
)
st.markdown("---")
# Load resources
best_model, tfidf_vectorizer, job_df, load_error = load_resources()

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

# --- Job Selection ---
st.subheader("1. Select a Job Posting")

job_titles = job_df["Display_Name"].tolist()

selected_title = st.selectbox(
    "Choose a job role (type to search):",
    options=job_titles,
    index=None,
    placeholder="Search for a job role...",
)
# If no job is selected, show a message and stop
if not selected_title:
    st.info("Please select a job role from the dropdown to continue.")
    st.stop()
#  Select the job posting row and get the clean job text
selected_job_row = job_df[job_df["Display_Name"] == selected_title].iloc[0]
clean_job_text   = str(selected_job_row["Clean_Job"])
# If the job description is empty, show an error and stop
if not clean_job_text.strip():
    st.error("The selected job posting has an empty description. Please choose another role.")
    st.stop()

with st.expander("View Job Description (cleaned text)"):
    st.write(clean_job_text)

st.markdown("---")

# --- Resume Upload ---
st.subheader("2. Upload Resume")

uploaded_file = st.file_uploader(
    "Upload a candidate resume (PDF or TXT):",
    type=["pdf", "txt"],
)
# If no file is uploaded, show a message and stop
if uploaded_file is None:
    st.info("Please upload a resume to see the evaluation.")
    st.stop()
#  Extract text from the uploaded file
raw_resume_text, extract_error = extract_text_from_file(uploaded_file)

# If the file cannot be read, show an error and stop
if extract_error:
    st.error(f"File Error: {extract_error}")
    st.stop()
# If the extracted text is empty, show an error and stop
if not raw_resume_text.strip():
    st.error(
        "The uploaded file appears to be empty or contains no readable text. "
        "If this is a scanned PDF, please use a text-layer PDF instead."
    )
    st.stop()

with st.expander("Preview Uploaded Resume (raw text)"):
    preview = raw_resume_text[:3000]
    st.text(preview + ("..." if len(raw_resume_text) > 3000 else ""))

cleaned_resume = clean_text(raw_resume_text)

if not cleaned_resume.strip():
    st.error(
        "After cleaning, the resume text is empty. "
        "The file may contain only numbers, punctuation, or common stopwords."
    )
    st.stop()

# Combine text (same format used during training) then vectorize
# Only transform() — vocabulary is frozen from training, never re-fit
combined_text   = cleaned_resume + " " + clean_job_text
combined_vector = tfidf_vectorizer.transform([combined_text])
resume_vector   = tfidf_vectorizer.transform([cleaned_resume])
job_vector      = tfidf_vectorizer.transform([clean_job_text])

# Predict suitability
prediction    = best_model.predict(combined_vector)[0]
probabilities = best_model.predict_proba(combined_vector)[0]
class_labels  = list(best_model.classes_)
prob_dict     = dict(zip(class_labels, probabilities))
confidence    = prob_dict[prediction]

# Cosine similarity — display only, not used for prediction
similarity_score = cosine_similarity(resume_vector, job_vector)[0][0]

# --- Results ---
st.markdown("---")
st.subheader("3. Evaluation Results")

if prediction == "Highly Suitable":
    st.success(f"Prediction:  {prediction}")
elif prediction == "Suitable":
    st.warning(f"Prediction:  {prediction}")
else:
    st.error(f"Prediction:  {prediction}")
# Map the prediction to a recommendation
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
# Display model confidence and resume-job similarity
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
# Debug information
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

# --- Export Report ---
st.markdown("---")
st.subheader("4. Export Report")

report_data = {
    "Candidate File":         [uploaded_file.name],
    "Selected Job":           [selected_title],
    "Prediction":             [prediction],
    "Confidence (%)":         [f"{confidence * 100:.2f}"],
    "Cosine Similarity (%)":  [f"{similarity_score * 100:.2f}"],
}

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
