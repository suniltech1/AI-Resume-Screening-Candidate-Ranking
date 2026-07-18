"""
AI Resume Screening and Candidate Ranking System
-------------------------------------------------
preprocessing.py

Generates Resume-Job training pairs for multi-job generalization.

Steps:
1. Load raw resume and job datasets.
2. Clean all text using the shared clean_text() function.
3. Keep all unique jobs for the Streamlit UI, but sample jobs for pair generation to avoid excessive memory and training time.
4. Generate Resume x Job training pairs using a sampled subset of jobs.
5. Label each pair with a suitability score using cosine similarity
   and absolute thresholds.
6. Save:
     data/clean_resume_dataset.csv
     data/clean_job_dataset.csv (contains ALL jobs for the UI)
     data/training_pairs.csv (subset of pairs for fast model training)

NOTE: clean_text() is the single source of truth for text cleaning.
      It must remain IDENTICAL in preprocessing.py, train_model.py,
      and app.py. Any difference will corrupt the TF-IDF feature space.
"""

import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration

# Number of unique job postings to sample FOR PAIR GENERATION ONLY.
# We save ALL jobs to clean_job_dataset.csv, but generating pairs for all
# 15k jobs would result in ~15 million pairs, crashing most systems.
# We limit training pairs to a subset of jobs.
N_SAMPLE_JOBS_FOR_TRAINING = 60

RANDOM_SEED = 42
DATA_DIR    = Path("data")

# NLTK Setup
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# Text Cleaning Function
def clean_text(text):
    """
    Clean input text to match the training pipeline.
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

    # Step 4: Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Step 5: Remove stopwords
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]

    return " ".join(words)


# Step 1: Load and Clean Resume Dataset
print("=" * 60)
print("STEP 1: Loading and cleaning resume dataset...")
print("=" * 60)

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

# Step 2: Load and Clean Job Dataset
print("\n" + "=" * 60)
print("STEP 2: Loading and cleaning job dataset...")
print("=" * 60)

job_df = pd.read_csv(DATA_DIR / "job_postings.csv")
print(f"Raw job dataset shape: {job_df.shape}")

# Keep only columns relevant to describing the job
required_columns = [
    "title",
    "description",
    "skills_desc",
    "formatted_experience_level"
]

# Ensure all columns exist, handle missing ones if necessary
for col in required_columns:
    if col not in job_df.columns:
        job_df[col] = ""

job_df = job_df[required_columns].copy()

# Replace missing values with empty strings before combining
job_df.fillna("", inplace=True)

# Combine all job fields into a single text column
job_df["Job_Text"] = (
    job_df["title"] + " " +
    job_df["description"] + " " +
    job_df["skills_desc"] + " " +
    job_df["formatted_experience_level"]
)

# Apply text cleaning
job_df["Clean_Job"] = job_df["Job_Text"].apply(clean_text)

# Remove jobs that produced empty text after cleaning
job_df = job_df[job_df["Clean_Job"].str.strip() != ""]

# Remove duplicate cleaned job descriptions
job_df.drop_duplicates(subset=["Clean_Job"], inplace=True)
job_df.reset_index(drop=True, inplace=True)

print(f"Unique cleaned job postings available: {len(job_df)}")

# Generate Display_Name for Streamlit Selectbox
def create_display_name(row):
    title = str(row['title']).strip()
    exp = str(row['formatted_experience_level']).strip()
    if exp:
        return f"{title} ({exp})"
    return title

job_df["Display_Name"] = job_df.apply(create_display_name, axis=1)

# Save the ENTIRE cleaned job dataset for Streamlit
job_df[["title", "formatted_experience_level", "Display_Name", "Clean_Job"]].to_csv(DATA_DIR / "clean_job_dataset.csv", index=False)
print("Saved: data/clean_job_dataset.csv (Contains all jobs for the Streamlit UI)")


# Step 3: Generate Resume-Job Training Pairs
print("\n" + "=" * 60)
print("STEP 3: Generating Resume-Job training pairs...")
print("=" * 60)

# Sample jobs for training pair generation to prevent memory/time explosion
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

p_high     = 0.10
p_suitable = 0.05

def create_label(score, threshold_high, threshold_suitable):
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
