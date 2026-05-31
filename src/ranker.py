"""
ranker.py — Final ranking logic: sort, filter, and return top-K candidates.

This module takes hybrid scores and produces the final ranked shortlist.
Supports optional supervised re-ranking if labeled data becomes available.
"""

import pandas as pd
import numpy as np


def rank_candidates(candidates_df, scores, top_k=10, score_threshold=0.0,
                    id_column="candidate_id", score_column="hybrid_score"):
    """
    Rank candidates by their hybrid scores and return top-K.

    Args:
        candidates_df (pd.DataFrame): Original candidate dataframe.
        scores (list[float]): Hybrid scores (one per candidate).
        top_k (int): Number of top candidates to return.
        score_threshold (float): Minimum score to include (0 = include all).
        id_column (str): Name of the candidate ID column.
        score_column (str): Name for the score column in output.

    Returns:
        pd.DataFrame: Ranked candidates with scores, sorted best-first.

    Example:
        >>> ranked = rank_candidates(candidates, hybrid_scores, top_k=5)
        >>> print(ranked[["candidate_id", "hybrid_score"]].head())
    """
    results = candidates_df.copy()
    results[score_column] = scores
    results["rank"] = 0  # Will be filled after sorting

    # Filter by threshold
    if score_threshold > 0:
        results = results[results[score_column] >= score_threshold]

    # Sort by score descending
    results = results.sort_values(score_column, ascending=False).reset_index(drop=True)

    # Assign ranks (1-indexed)
    results["rank"] = range(1, len(results) + 1)

    # Return top-K
    if top_k > 0:
        results = results.head(top_k)

    return results


def format_shortlist(ranked_df, config, include_explanation=False):
    """
    Format the ranked results into a clean shortlist for output.

    Args:
        ranked_df (pd.DataFrame): Ranked candidates from rank_candidates().
        config (dict): Configuration dictionary.
        include_explanation (bool): If True, include score breakdown columns.

    Returns:
        pd.DataFrame: Clean shortlist with relevant columns.
    """
    cc = config["dataset"]["candidate_columns"]

    # Select columns to include in output
    output_columns = ["rank", "hybrid_score"]

    # Add candidate columns that exist
    for key in ["id", "name", "job_title", "skills", "experience_years", "education"]:
        col = cc.get(key, "")
        if col and col in ranked_df.columns:
            output_columns.append(col)

    # Add explanation columns if requested
    if include_explanation:
        explanation_cols = [c for c in ranked_df.columns if c.endswith("_contribution")]
        output_columns.extend(explanation_cols)

    # Filter to existing columns only
    output_columns = [c for c in output_columns if c in ranked_df.columns]

    return ranked_df[output_columns]


# ==============================================================================
# OPTIONAL: Supervised Re-ranker (activate when labeled data is available)
# ==============================================================================

def train_reranker(features_df, labels, model_type="xgboost"):
    """
    Train a supervised re-ranking model on recruiter-labeled data.

    ⚠️ OPTIONAL — Only use this when you have labeled data (e.g., recruiter
    shortlist decisions). For the initial hackathon version, the heuristic
    scoring in scoring.py is sufficient.

    Args:
        features_df (pd.DataFrame): Feature matrix (from features.py).
        labels (pd.Series): Binary labels (1=shortlisted, 0=rejected).
        model_type (str): "xgboost" or "lightgbm".

    Returns:
        model: Trained model object.
    """
    # TODO: Implement when labeled data is available
    #
    # Rough plan:
    # 1. Split features_df + labels into train/val
    # 2. Train XGBoost/LightGBM classifier or ranker
    # 3. Return trained model
    #
    # from sklearn.model_selection import train_test_split
    # if model_type == "xgboost":
    #     import xgboost as xgb
    #     model = xgb.XGBClassifier(...)
    # elif model_type == "lightgbm":
    #     import lightgbm as lgb
    #     model = lgb.LGBMClassifier(...)
    #
    # model.fit(X_train, y_train)
    # return model

    raise NotImplementedError(
        "Supervised re-ranker is not yet implemented. "
        "Use heuristic scoring (scoring.py) for now."
    )


def predict_reranker(model, features_df):
    """
    Use a trained re-ranker to predict scores for candidates.

    Args:
        model: Trained model from train_reranker().
        features_df (pd.DataFrame): Feature matrix for prediction.

    Returns:
        np.ndarray: Predicted scores / probabilities.
    """
    # TODO: Implement when train_reranker is ready
    # return model.predict_proba(features_df)[:, 1]

    raise NotImplementedError(
        "Supervised re-ranker is not yet implemented."
    )
