"""
features.py — Structured feature engineering for recruiter-like scoring.

These features capture things that pure text embeddings might miss:
skill overlap, experience matching, education alignment, etc.
"""

from src.preprocess import normalize_skills


def compute_skills_overlap(candidate_skills_str, job_skills_str, delimiter=","):
    """
    Compute the fraction of required job skills that the candidate has.

    Args:
        candidate_skills_str (str): Candidate's skills (comma-separated).
        job_skills_str (str): Job's required skills (comma-separated).
        delimiter (str): Delimiter for skills strings.

    Returns:
        float: Overlap score between 0.0 and 1.0.
               1.0 means candidate has ALL required skills.

    Example:
        >>> compute_skills_overlap("python, ml, nlp", "python, ml, java")
        0.6667
    """
    candidate_skills = set(normalize_skills(candidate_skills_str, delimiter))
    job_skills = set(normalize_skills(job_skills_str, delimiter))

    if not job_skills:
        return 0.0  # No required skills specified

    overlap = candidate_skills & job_skills
    return len(overlap) / len(job_skills)


def compute_experience_score(candidate_years, required_years):
    """
    Score how well a candidate's experience matches the job requirement.

    Logic:
    - Meets or exceeds requirement → 1.0
    - Below requirement → proportional score (capped at 0.0)

    Args:
        candidate_years (float): Candidate's years of experience.
        required_years (float): Job's minimum experience requirement.

    Returns:
        float: Experience match score between 0.0 and 1.0.

    Example:
        >>> compute_experience_score(5, 3)
        1.0
        >>> compute_experience_score(2, 5)
        0.4
    """
    try:
        candidate_years = float(candidate_years)
        required_years = float(required_years)
    except (ValueError, TypeError):
        return 0.5  # Unknown → neutral score

    if required_years <= 0:
        return 1.0  # No requirement

    score = min(candidate_years / required_years, 1.0)
    return max(score, 0.0)


def compute_education_score(candidate_education, job_education=None):
    """
    Simple education level scoring based on common hierarchy.

    This is a basic heuristic — customize for your dataset.

    Args:
        candidate_education (str): Candidate's education level/description.
        job_education (str, optional): Job's education requirement.

    Returns:
        float: Education score between 0.0 and 1.0.
    """
    # Simple hierarchy mapping (customize as needed)
    education_levels = {
        "phd": 1.0,
        "doctorate": 1.0,
        "master": 0.85,
        "masters": 0.85,
        "mba": 0.85,
        "bachelor": 0.7,
        "bachelors": 0.7,
        "associate": 0.5,
        "diploma": 0.4,
        "high school": 0.3,
        "self-taught": 0.3,
    }

    if not isinstance(candidate_education, str):
        return 0.5  # Unknown → neutral

    edu_lower = candidate_education.lower().strip()

    for key, score in education_levels.items():
        if key in edu_lower:
            return score

    return 0.5  # Default neutral score


def build_feature_vector(candidate_row, job_row, config):
    """
    Build a dictionary of structured features for one candidate-job pair.

    Args:
        candidate_row (pd.Series): One candidate's data.
        job_row (pd.Series): One job's data.
        config (dict): Configuration dictionary.

    Returns:
        dict: Feature name → value mapping.

    Example:
        >>> features = build_feature_vector(candidate, job, config)
        >>> print(features)
        {'skills_overlap': 0.75, 'experience_score': 1.0, 'education_score': 0.85}
    """
    cc = config["dataset"]["candidate_columns"]
    jc = config["dataset"]["job_columns"]

    features = {}

    # --- Skills overlap ---
    cand_skills = candidate_row.get(cc.get("skills", ""), "")
    job_skills = job_row.get(jc.get("required_skills", ""), "")
    features["skills_overlap"] = compute_skills_overlap(cand_skills, job_skills)

    # --- Experience match ---
    cand_exp = candidate_row.get(cc.get("experience_years", ""), 0)
    job_exp = job_row.get(jc.get("min_experience", ""), 0)
    features["experience_score"] = compute_experience_score(cand_exp, job_exp)

    # --- Education score ---
    cand_edu = candidate_row.get(cc.get("education", ""), "")
    features["education_score"] = compute_education_score(cand_edu)

    # TODO: Add more features as needed:
    # - Title similarity (fuzzy match between current title and job title)
    # - Location match (same city/remote compatibility)
    # - Recency score (how recent is relevant experience)
    # - Certification match

    return features
