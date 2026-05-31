"""
utils.py — Shared utility functions used across the pipeline.

Helpers for I/O, formatting, timing, and common operations.
"""

import os
import time
import json
import functools
import pandas as pd
import numpy as np


# ==============================================================================
# Timer / Profiling
# ==============================================================================

def timer(func):
    """
    Decorator that prints how long a function took to run.

    Usage:
        @timer
        def my_slow_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"⏱️  {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


# ==============================================================================
# Data Helpers
# ==============================================================================

def safe_str(value, default=""):
    """
    Safely convert a value to string, handling NaN and None.

    Args:
        value: Any value.
        default (str): Default if value is NaN/None.

    Returns:
        str: String representation.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return str(value).strip()


def flatten_list(nested_list):
    """
    Flatten a list of lists into a single list.

    Args:
        nested_list (list): List of lists.

    Returns:
        list: Flattened list.
    """
    return [item for sublist in nested_list for item in sublist]


# ==============================================================================
# File I/O
# ==============================================================================

def ensure_dir(path):
    """
    Create directory if it doesn't exist.

    Args:
        path (str): Directory path.
    """
    os.makedirs(path, exist_ok=True)


def save_json(data, filepath):
    """
    Save a dictionary or list to a JSON file.

    Args:
        data: JSON-serializable data.
        filepath (str): Output file path.
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"💾 Saved JSON: {filepath}")


def load_json(filepath):
    """
    Load a JSON file.

    Args:
        filepath (str): Path to JSON file.

    Returns:
        dict or list: Parsed JSON content.
    """
    with open(filepath, "r") as f:
        return json.load(f)


def save_embeddings(embeddings, filepath):
    """
    Save embeddings as a NumPy file for reuse (avoids re-computing).

    Args:
        embeddings (np.ndarray): Embedding matrix.
        filepath (str): Output path (should end in .npy).
    """
    ensure_dir(os.path.dirname(filepath))
    np.save(filepath, embeddings)
    print(f"💾 Saved embeddings: {filepath} (shape: {embeddings.shape})")


def load_embeddings(filepath):
    """
    Load pre-computed embeddings from a NumPy file.

    Args:
        filepath (str): Path to .npy file.

    Returns:
        np.ndarray: Embedding matrix.
    """
    embeddings = np.load(filepath)
    print(f"✅ Loaded embeddings: {filepath} (shape: {embeddings.shape})")
    return embeddings


# ==============================================================================
# Display / Formatting
# ==============================================================================

def print_ranking(ranked_df, score_col="hybrid_score", max_rows=10):
    """
    Pretty-print a ranking table to the console.

    Args:
        ranked_df (pd.DataFrame): Ranked candidates.
        score_col (str): Name of the score column.
        max_rows (int): Maximum rows to display.
    """
    print("\n" + "=" * 50)
    print("🏆 CANDIDATE RANKING")
    print("=" * 50)

    display_df = ranked_df.head(max_rows)
    for _, row in display_df.iterrows():
        rank = row.get("rank", "?")
        score = row.get(score_col, 0)
        name = row.get("candidate_name", row.get("candidate_id", "Unknown"))
        print(f"  #{rank:<3}  Score: {score:.4f}  |  {name}")

    print("=" * 50)
    print(f"Showing top {min(max_rows, len(ranked_df))} of {len(ranked_df)} candidates\n")


# ==============================================================================
# Colab Helpers
# ==============================================================================

def setup_colab():
    """
    Set up the environment for Google Colab.
    Call this at the top of any Colab notebook.

    Returns:
        str: Project root path.
    """
    import sys

    # Common Colab project paths
    possible_paths = [
        "/content/ai-candidate-ranking",
        "/content/drive/MyDrive/ai-candidate-ranking",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            if path not in sys.path:
                sys.path.insert(0, path)
            os.chdir(path)
            print(f"✅ Working directory set to: {path}")
            return path

    print("⚠️ Project directory not found. Please set path manually.")
    return os.getcwd()
