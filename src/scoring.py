"""
scoring.py — Heuristic and hybrid scoring logic.

Combines semantic similarity scores with structured feature scores
to produce a final candidate ranking score.
"""

import numpy as np


def compute_hybrid_score(semantic_score, feature_scores, weights=None):
    """
    Combine semantic similarity and structured features into one final score.

    Args:
        semantic_score (float): Cosine similarity score (0 to 1).
        feature_scores (dict): Feature name → value mapping from features.py.
        weights (dict, optional): Scoring weights from config.
            Expected keys: "semantic_weight", "skills_weight",
                           "experience_weight", "education_weight"

    Returns:
        float: Final hybrid score between 0.0 and 1.0.

    Example:
        >>> score = compute_hybrid_score(
        ...     semantic_score=0.82,
        ...     feature_scores={"skills_overlap": 0.75, "experience_score": 1.0, "education_score": 0.85},
        ...     weights={"semantic_weight": 0.5, "skills_weight": 0.25,
        ...              "experience_weight": 0.15, "education_weight": 0.10}
        ... )
        >>> print(f"{score:.4f}")
        0.7475
    """
    # Default weights if none provided
    if weights is None:
        weights = {
            "semantic_weight": 0.50,
            "skills_weight": 0.25,
            "experience_weight": 0.15,
            "education_weight": 0.10,
        }

    # Normalize weights to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    # Compute weighted sum
    score = (
        weights.get("semantic_weight", 0) * semantic_score
        + weights.get("skills_weight", 0) * feature_scores.get("skills_overlap", 0)
        + weights.get("experience_weight", 0) * feature_scores.get("experience_score", 0)
        + weights.get("education_weight", 0) * feature_scores.get("education_score", 0)
    )

    return float(np.clip(score, 0.0, 1.0))


def score_all_candidates(semantic_scores, all_feature_scores, weights=None):
    """
    Compute hybrid scores for all candidates against a single job.

    Args:
        semantic_scores (np.ndarray): Array of semantic similarity scores, shape (n,).
        all_feature_scores (list[dict]): List of feature dicts, one per candidate.
        weights (dict, optional): Scoring weights from config.

    Returns:
        list[float]: Final hybrid scores for each candidate.

    Example:
        >>> scores = score_all_candidates(sim_scores, feature_list, config["scoring"])
        >>> print(len(scores))
        100
    """
    hybrid_scores = []

    for i, (sem_score, feat_scores) in enumerate(zip(semantic_scores, all_feature_scores)):
        hybrid = compute_hybrid_score(sem_score, feat_scores, weights)
        hybrid_scores.append(hybrid)

    return hybrid_scores


def explain_score(semantic_score, feature_scores, weights=None):
    """
    Generate a human-readable explanation of why a candidate scored the way they did.

    Useful for recruiter trust — shows which signals contributed most.

    Args:
        semantic_score (float): Cosine similarity score.
        feature_scores (dict): Structured feature scores.
        weights (dict, optional): Scoring weights.

    Returns:
        dict: Breakdown of score contributions.

    Example:
        >>> breakdown = explain_score(0.82, features, weights)
        >>> for k, v in breakdown.items():
        ...     print(f"{k}: {v:.2f}")
    """
    if weights is None:
        weights = {
            "semantic_weight": 0.50,
            "skills_weight": 0.25,
            "experience_weight": 0.15,
            "education_weight": 0.10,
        }

    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    breakdown = {
        "semantic_contribution": weights.get("semantic_weight", 0) * semantic_score,
        "skills_contribution": weights.get("skills_weight", 0) * feature_scores.get("skills_overlap", 0),
        "experience_contribution": weights.get("experience_weight", 0) * feature_scores.get("experience_score", 0),
        "education_contribution": weights.get("education_weight", 0) * feature_scores.get("education_score", 0),
    }

    breakdown["total_score"] = sum(breakdown.values())
    return breakdown
