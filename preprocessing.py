# Generates Resume-Job training pairs and saves cleaned datasets.

import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Max jobs sampled for training pairs (all jobs still saved for the UI)
N_SAMPLE_JOBS_FOR_TRAINING = 60

RANDOM_SEED = 42
DATA_DIR    = Path("data")

# Download stopwords silently; STOP_WORDS used in every clean_text() call
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    """Clean text — must stay identical across preprocessing.py, train_model.py, and app.py."""
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


# Step 1: Load and clean resume dataset
print("Loading and cleaning resume dataset...")

resume_df = pd.read_csv(DATA_DIR / "UpdatedResumeDataSet.csv")
print(f"Raw resume dataset shape: {resume_df.shape}")

resume_df = resume_df[["Category", "Resume"]].copy()
resume_df.dropna(inplace=True)
resume_df.drop_duplicates(inplace=True)
resume_df["Clean_Resume"] = resume_df["Resume"].apply(clean_text)
resume_df = resume_df[resume_df["Clean_Resume"].str.strip() != ""].reset_index(drop=True)

print(f"Cleaned resume dataset shape: {resume_df.shape}")
resume_df.to_csv(DATA_DIR / "clean_resume_dataset.csv", index=False)
print("Saved: data/clean_resume_dataset.csv")

# Step 2: Load and clean job dataset
print("\nLoading and cleaning job dataset...")

job_df = pd.read_csv(DATA_DIR / "job_postings.csv")
print(f"Raw job dataset shape: {job_df.shape}")

required_columns = ["title", "description", "skills_desc", "formatted_experience_level"]

for col in required_columns:
    if col not in job_df.columns:
        job_df[col] = ""

job_df = job_df[required_columns].copy()
job_df.fillna("", inplace=True)

job_df["Job_Text"] = (
    job_df["title"] + " " +
    job_df["description"] + " " +
    job_df["skills_desc"] + " " +
    job_df["formatted_experience_level"]
)
job_df["Clean_Job"] = job_df["Job_Text"].apply(clean_text)
job_df = job_df[job_df["Clean_Job"].str.strip() != ""]
job_df.drop_duplicates(subset=["Clean_Job"], inplace=True)
job_df.reset_index(drop=True, inplace=True)

print(f"Unique cleaned job postings available: {len(job_df)}")

# Build display name for the Streamlit selectbox
def create_display_name(row):
    title = str(row['title']).strip()
    exp = str(row['formatted_experience_level']).strip()
    if exp:
        return f"{title} ({exp})"
    return title

job_df["Display_Name"] = job_df.apply(create_display_name, axis=1)
job_df[["title", "formatted_experience_level", "Display_Name", "Clean_Job"]].to_csv(DATA_DIR / "clean_job_dataset.csv", index=False)
print("Saved: data/clean_job_dataset.csv")


# Step 3: Generate Resume-Job training pairs
print("\nGenerating Resume-Job training pairs...")

if len(job_df) > N_SAMPLE_JOBS_FOR_TRAINING:
    sampled_job_df = job_df.sample(n=N_SAMPLE_JOBS_FOR_TRAINING, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Sampled {N_SAMPLE_JOBS_FOR_TRAINING} unique job postings for training pair generation.")
else:
    sampled_job_df = job_df.copy()
    print(f"Using all {len(job_df)} available job postings for training pair generation.")

n_resumes = len(resume_df)
n_jobs    = len(sampled_job_df)
n_pairs   = n_resumes * n_jobs
print(f"{n_resumes} resumes x {n_jobs} jobs = {n_pairs:,} training pairs")

print("\nFitting temporary TF-IDF vectorizer for label generation...")
all_texts = resume_df["Clean_Resume"].tolist() + sampled_job_df["Clean_Job"].tolist()

temp_vectorizer = TfidfVectorizer(max_features=5000)
temp_matrix     = temp_vectorizer.fit_transform(all_texts)

resume_matrix = temp_matrix[:n_resumes]
job_matrix    = temp_matrix[n_resumes:]

print("Computing cosine similarity matrix...")
similarity_matrix = cosine_similarity(resume_matrix, job_matrix)

# Cosine similarity thresholds for labeling each resume-job pair
p_high     = 0.10   # >= 0.10 → Highly Suitable
p_suitable = 0.05   # >= 0.05 → Suitable, else Not Suitable

def create_label(score, threshold_high, threshold_suitable):
    """Maps a cosine similarity score to a suitability label."""
    if score >= threshold_high:
        return "Highly Suitable"
    elif score >= threshold_suitable:
        return "Suitable"
    else:
        return "Not Suitable"

print("\nBuilding training pairs dataframe...")

resume_indices = np.repeat(np.arange(n_resumes), n_jobs)
job_indices    = np.tile(np.arange(n_jobs), n_resumes)
similarities   = similarity_matrix.flatten()

clean_resumes_arr = resume_df["Clean_Resume"].values
clean_jobs_arr    = sampled_job_df["Clean_Job"].values

pairs_df = pd.DataFrame({
    "Clean_Resume": clean_resumes_arr[resume_indices],
    "Clean_Job":    clean_jobs_arr[job_indices],
    "Similarity":   similarities,
})

pairs_df["Combined_Text"] = pairs_df["Clean_Resume"] + " " + pairs_df["Clean_Job"]
pairs_df["Suitability"] = pairs_df["Similarity"].apply(
    lambda s: create_label(s, p_high, p_suitable)
)

print(f"\nTotal training pairs: {len(pairs_df):,}")
print("\nLabel distribution:")
print(pairs_df["Suitability"].value_counts())

pairs_df.to_csv(DATA_DIR / "training_pairs.csv", index=False)
print("\nSaved: data/training_pairs.csv")
print("\nPreprocessing completed successfully!")
print("Next step: python train_model.py")
