"""
preprocess.py — Text cleaning and normalization for candidates and JDs.

Handles all text preprocessing: cleaning, normalization, and building
composite text representations for embedding generation.
"""

import re
import string


def clean_text(text):
    """
    Basic text cleaning: lowercase, remove extra whitespace, strip punctuation noise.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text.

    Example:
        >>> clean_text("  Senior ML   Engineer @ Google!!! ")
        'senior ml engineer google'
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Remove excessive punctuation (keep basic ones)
    text = re.sub(r"[^\w\s\-\+\#\.]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_skills(skills_str, delimiter=","):
    """
    Parse and normalize a skills string into a clean list.

    Args:
        skills_str (str): Comma-separated (or other delimiter) skills string.
        delimiter (str): Delimiter used to separate skills.

    Returns:
        list[str]: List of cleaned, lowercased skill strings.

    Example:
        >>> normalize_skills("Python, Machine Learning , NLP,  TensorFlow")
        ['python', 'machine learning', 'nlp', 'tensorflow']
    """
    if not isinstance(skills_str, str) or not skills_str.strip():
        return []

    skills = [s.strip().lower() for s in skills_str.split(delimiter)]
    skills = [s for s in skills if s]  # Remove empty strings
    return skills


def build_candidate_text(row, config):
    """
    Build a single composite text string from a candidate's profile.

    This combines multiple fields into one text block for embedding.
    Adapt the fields based on your actual dataset columns.

    Args:
        row (pd.Series): A single candidate row from the dataframe.
        config (dict): Configuration dictionary.

    Returns:
        str: Combined text representation of the candidate.
    """
    cols = config["dataset"]["candidate_columns"]
    parts = []

    # Add each available field
    for field_key in ["job_title", "resume_text", "skills", "education"]:
        if field_key in cols:
            col_name = cols[field_key]
            value = row.get(col_name, "")
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())

    combined = " . ".join(parts)
    return clean_text(combined)


def build_job_text(row, config):
    """
    Build a single composite text string from a job description.

    This combines JD fields into one text block for embedding.

    Args:
        row (pd.Series): A single job row from the dataframe.
        config (dict): Configuration dictionary.

    Returns:
        str: Combined text representation of the job.
    """
    cols = config["dataset"]["job_columns"]
    parts = []

    for field_key in ["title", "description", "required_skills"]:
        if field_key in cols:
            col_name = cols[field_key]
            value = row.get(col_name, "")
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())

    combined = " . ".join(parts)
    return clean_text(combined)


def preprocess_dataframe(df, text_column, clean_column_name="clean_text"):
    """
    Add a cleaned text column to a dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        text_column (str): Name of the column to clean.
        clean_column_name (str): Name for the new cleaned column.

    Returns:
        pd.DataFrame: Dataframe with added clean text column.
    """
    df = df.copy()
    df[clean_column_name] = df[text_column].apply(clean_text)
    return df
