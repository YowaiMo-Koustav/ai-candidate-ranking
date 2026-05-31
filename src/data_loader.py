"""
data_loader.py — Load candidate and job data from various formats.

This module handles all data I/O. When the actual dataset arrives,
update the column names in configs/config.yaml and the loading logic here.
"""

import pandas as pd
import yaml
import os


def load_config(config_path="configs/config.yaml"):
    """
    Load the central YAML configuration file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_candidates(config):
    """
    Load candidate data from the raw data directory.

    Args:
        config (dict): Configuration dictionary (from load_config).

    Returns:
        pd.DataFrame: Candidate profiles dataframe.

    Example:
        >>> config = load_config()
        >>> candidates = load_candidates(config)
        >>> print(candidates.shape)
    """
    filepath = os.path.join(
        config["paths"]["raw_data"],
        config["dataset"]["candidates_file"]
    )

    # Auto-detect format by extension
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith(".json"):
        df = pd.read_json(filepath)
    elif filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    print(f"✅ Loaded {len(df)} candidates from {filepath}")
    return df


def load_jobs(config):
    """
    Load job descriptions from the raw data directory.

    Args:
        config (dict): Configuration dictionary (from load_config).

    Returns:
        pd.DataFrame: Job descriptions dataframe.

    Example:
        >>> config = load_config()
        >>> jobs = load_jobs(config)
        >>> print(jobs.shape)
    """
    filepath = os.path.join(
        config["paths"]["raw_data"],
        config["dataset"]["jobs_file"]
    )

    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith(".json"):
        df = pd.read_json(filepath)
    elif filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    print(f"✅ Loaded {len(df)} job descriptions from {filepath}")
    return df


def load_single_jd(jd_text):
    """
    Create a job dataframe from a single JD string (for quick testing).

    Args:
        jd_text (str): Raw job description text.

    Returns:
        pd.DataFrame: Single-row dataframe with the JD.

    Example:
        >>> jd = "Looking for a senior ML engineer with 5+ years experience..."
        >>> jobs = load_single_jd(jd)
    """
    return pd.DataFrame([{
        "job_id": "manual_001",
        "job_title": "Manual Entry",
        "job_description": jd_text,
        "required_skills": "",
        "min_experience": 0
    }])


def get_column(config, section, key):
    """
    Safely get a column name from config, with a helpful error if missing.

    Args:
        config (dict): Configuration dictionary.
        section (str): Either "candidate_columns" or "job_columns".
        key (str): The logical column name (e.g., "resume_text").

    Returns:
        str: The actual column name in the dataset.
    """
    try:
        return config["dataset"][section][key]
    except KeyError:
        raise KeyError(
            f"Column '{key}' not found in config['dataset']['{section}']. "
            f"Please update configs/config.yaml with your dataset's column names."
        )
