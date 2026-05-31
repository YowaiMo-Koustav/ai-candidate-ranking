"""
inference.py — End-to-end inference pipeline.

Ties together all modules: load data → preprocess → embed → score → rank → save.
Call run_pipeline() to execute the full pipeline in one shot.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_loader import load_config, load_candidates, load_jobs, get_column
from src.preprocess import build_candidate_text, build_job_text
from src.embeddings import load_embedding_model, generate_embeddings, compute_similarity
from src.features import build_feature_vector
from src.scoring import score_all_candidates, explain_score
from src.ranker import rank_candidates, format_shortlist


def run_pipeline(config_path="configs/config.yaml", job_index=0, top_k=None):
    """
    Run the full candidate ranking pipeline end-to-end.

    Steps:
        1. Load config and data
        2. Preprocess texts
        3. Generate embeddings
        4. Compute semantic similarity
        5. Engineer structured features
        6. Compute hybrid scores
        7. Rank and return top-K

    Args:
        config_path (str): Path to the configuration file.
        job_index (int): Index of the job to rank candidates for.
                         Use 0 for single-JD datasets.
        top_k (int, optional): Override for number of top candidates.
                               If None, uses config value.

    Returns:
        pd.DataFrame: Ranked shortlist of candidates.

    Example:
        >>> from src.inference import run_pipeline
        >>> results = run_pipeline()
        >>> results.to_csv("outputs/submissions/ranked_candidates.csv", index=False)
    """
    # --- Step 1: Load config and data ---
    print("=" * 60)
    print("🚀 AI Candidate Ranking Pipeline")
    print("=" * 60)

    config = load_config(config_path)
    candidates = load_candidates(config)
    jobs = load_jobs(config)

    if top_k is None:
        top_k = config["ranking"]["top_k"]

    # Select the target job
    job_row = jobs.iloc[job_index]
    job_id = job_row.get(get_column(config, "job_columns", "id"), f"job_{job_index}")
    print(f"\n📋 Ranking candidates for: {job_id}")

    # --- Step 2: Preprocess texts ---
    print("\n📝 Preprocessing texts...")
    candidate_texts = [
        build_candidate_text(candidates.iloc[i], config)
        for i in range(len(candidates))
    ]
    job_text = build_job_text(job_row, config)

    print(f"   Candidates: {len(candidate_texts)} texts built")
    print(f"   Job text preview: {job_text[:100]}...")

    # --- Step 3: Generate embeddings ---
    print("\n🧠 Generating embeddings...")
    model_name = config["embeddings"]["model_name"]
    batch_size = config["embeddings"]["batch_size"]

    model = load_embedding_model(model_name)
    candidate_embeddings = generate_embeddings(candidate_texts, model, batch_size)
    job_embedding = generate_embeddings([job_text], model, batch_size)

    print(f"   Candidate embeddings: {candidate_embeddings.shape}")
    print(f"   Job embedding: {job_embedding.shape}")

    # --- Step 4: Compute semantic similarity ---
    print("\n🔗 Computing semantic similarity...")
    semantic_scores = compute_similarity(job_embedding, candidate_embeddings)
    print(f"   Score range: [{semantic_scores.min():.4f}, {semantic_scores.max():.4f}]")

    # --- Step 5: Engineer structured features ---
    print("\n⚙️ Engineering structured features...")
    all_features = []
    for i in range(len(candidates)):
        features = build_feature_vector(candidates.iloc[i], job_row, config)
        all_features.append(features)

    print(f"   Features per candidate: {list(all_features[0].keys()) if all_features else 'none'}")

    # --- Step 6: Compute hybrid scores ---
    print("\n📊 Computing hybrid scores...")
    weights = config.get("scoring", {})
    hybrid_scores = score_all_candidates(semantic_scores, all_features, weights)
    print(f"   Hybrid score range: [{min(hybrid_scores):.4f}, {max(hybrid_scores):.4f}]")

    # --- Step 7: Rank and format ---
    print(f"\n🏆 Ranking top {top_k} candidates...")
    cc = config["dataset"]["candidate_columns"]
    id_col = cc.get("id", "candidate_id")
    threshold = config["ranking"].get("score_threshold", 0.0)

    ranked = rank_candidates(candidates, hybrid_scores, top_k=top_k,
                             score_threshold=threshold, id_column=id_col)

    shortlist = format_shortlist(ranked, config, include_explanation=True)

    print(f"\n✅ Pipeline complete! Top {len(shortlist)} candidates ranked.")
    print("=" * 60)

    return shortlist


def save_results(results_df, output_dir="outputs/submissions/", prefix="ranked"):
    """
    Save ranked results to CSV with a timestamp.

    Args:
        results_df (pd.DataFrame): Ranked candidates dataframe.
        output_dir (str): Directory to save to.
        prefix (str): Filename prefix.

    Returns:
        str: Path to saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    results_df.to_csv(filepath, index=False)
    print(f"💾 Results saved to: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Quick CLI entry point (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_pipeline()
    save_results(results)
    print(results.to_string(index=False))
