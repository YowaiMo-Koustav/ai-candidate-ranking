import argparse
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import build_jd_text, load_candidates_for_ids
from src.embeddings import load_encoder_model
from src.scoring import compute_final_score, generate_reasoning


def validate_submission(submission_df, candidates_path):
    """
    Run spec guardrail assertions on the submission DataFrame.
    Raises AssertionError if any check fails.
    """
    # Work with core columns only (ignore job_id if present)
    check_df = submission_df
    # 1. Exactly 100 rows
    assert len(check_df) == 100, (
        f"Expected 100 rows, got {len(check_df)}"
    )

    # 2. Ranks 1..100 with no gaps or duplicates
    ranks = check_df["rank"].tolist()
    assert sorted(set(ranks)) == list(range(1, 101)), (
        "Ranks must be 1..100 with no gaps or duplicates"
    )

    # 3. No duplicate candidate IDs
    ids = check_df["candidate_id"].tolist()
    assert len(set(ids)) == len(ids), (
        "Duplicate candidate_id values found in submission"
    )

    # 4. All IDs exist in the original candidates file
    cand_ids = set()
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            if cid:
                cand_ids.add(cid)

    missing = [cid for cid in ids if cid not in cand_ids]
    assert not missing, (
        f"{len(missing)} candidate_ids not found in candidates.jsonl: {missing[:5]}"
    )

    # 5. Scores monotonically non-increasing
    scores = check_df["score"].tolist()
    mono = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    assert mono, "Scores must be monotonically non-increasing with rank"

    # 6. Scores not all identical
    assert len(set(scores)) > 1, (
        "All scores are identical – this is discouraged by the spec"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Candidate Ranking Inference (Offline)"
    )
    parser.add_argument(
        "--candidates", type=str, default="data/raw/candidates.jsonl",
        help="Path to raw candidates JSONL (used for reasoning enrichment).",
    )
    parser.add_argument(
        "--job-desc", type=str, default="data/raw/job_description.docx",
        help="Path to the Job Description Word document.",
    )
    parser.add_argument(
        "--embeddings", type=str, default="data/processed/embeddings.npy",
        help="Path to precomputed candidate embeddings.",
    )
    parser.add_argument(
        "--candidate-ids", type=str, default="data/processed/candidate_ids.npy",
        help="Path to precomputed candidate ID array.",
    )
    parser.add_argument(
        "--features", type=str, default="data/processed/candidates_feather.parquet",
        help="Path to precomputed tabular features parquet file.",
    )
    parser.add_argument(
        "--out", type=str, default="outputs/submissions/ranked_candidates.csv",
        help="Path to save the final submission CSV.",
    )
    parser.add_argument(
        "--top-k", type=int, default=100,
        help="Number of top candidates to include in the output.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  AI Candidate Ranking — Inference Pipeline")
    print("=" * 60)
    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # 1. Load Precomputed Artifacts
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n[1/7] Loading precomputed artifacts...")

    candidate_ids = np.load(args.candidate_ids, allow_pickle=True)
    cand_embeddings = np.load(args.embeddings)
    features_df = pd.read_parquet(args.features)

    print(f"  Loaded {len(candidate_ids)} candidate IDs")
    print(f"  Embeddings shape: {cand_embeddings.shape}")
    print(f"  Features shape: {features_df.shape}")
    print(f"  ⏱  {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 2. Build and Encode JD
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n[2/7] Encoding Job Description...")

    jd_text = build_jd_text(args.job_desc)
    model = load_encoder_model()
    jd_embedding = model.encode([jd_text]) if jd_text else np.array([])

    print(f"  JD text length: {len(jd_text)} chars")
    print(f"  ⏱  {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 3. Compute Semantic Similarity
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n[3/7] Computing cosine similarities...")

    if cand_embeddings.shape[0] > 0 and jd_embedding.shape[0] > 0:
        similarities = cosine_similarity(jd_embedding, cand_embeddings)[0]
    else:
        similarities = np.zeros(len(candidate_ids))

    sim_df = pd.DataFrame({
        "candidate_id": candidate_ids,
        "semantic_similarity": similarities,
    })
    full_df = pd.merge(features_df, sim_df, on="candidate_id", how="inner")
    print(f"  Merged DataFrame: {full_df.shape}")
    print(f"  ⏱  {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 4. Final Scoring
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n[4/7] Computing final scores...")

    max_ml = full_df["ml_years_estimate"].max() if "ml_years_estimate" in full_df.columns else 1.0
    full_df["final_score"] = full_df.apply(
        lambda row: compute_final_score(row, max_ml_years=max_ml), axis=1
    )

    print(f"  Score range: {full_df['final_score'].min():.4f} – {full_df['final_score'].max():.4f}")
    print(f"  ⏱  {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 5. Filter and Rank
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n[5/7] Filtering honeypots and ranking top {args.top_k}...")

    filtered_df = full_df[full_df["honeypot_risk_score"] < 0.6].copy()
    ranked_df = filtered_df.sort_values(by="final_score", ascending=False).head(args.top_k).copy()

    honeypot_count = len(full_df) - len(filtered_df)
    print(f"  Filtered out {honeypot_count} high-risk honeypot candidates")
    print(f"  Top {args.top_k} selected from {len(filtered_df)} remaining")
    print(f"  ⏱  {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 6. Generate JD-Aware Reasoning
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n[6/7] Generating context-aware reasoning...")

    top_ids = set(ranked_df["candidate_id"].values)
    cand_details = load_candidates_for_ids(args.candidates, top_ids)
    print(f"  Loaded raw details for {len(cand_details)}/{len(top_ids)} candidates")

    def safe_reasoning(row):
        cid = row["candidate_id"]
        return generate_reasoning(row, cand_raw=cand_details.get(cid))

    ranked_df["reasoning"] = ranked_df.apply(safe_reasoning, axis=1)
    unique_reasons = ranked_df["reasoning"].nunique()
    print(f"  Unique reasonings: {unique_reasons}/{len(ranked_df)}")
    print(f"  ⏱  {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 7. Format and Export
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n[7/7] Formatting final submission...")

    ranked_df["score"] = ranked_df["final_score"].round(3)
    ranked_df["rank"] = range(1, len(ranked_df) + 1)

    # Enforce monotonically non-increasing scores
    prev_score = ranked_df.iloc[0]["score"]
    adjusted_scores = []
    for s in ranked_df["score"]:
        if s > prev_score:
            adjusted_scores.append(prev_score)
        else:
            adjusted_scores.append(s)
            prev_score = s
    ranked_df["score"] = adjusted_scores

    # Build submission with job_id for validator compatibility
    submission_df = ranked_df[["candidate_id", "rank", "score", "reasoning"]].copy()
    submission_df.insert(1, "job_id", "JD_SENIOR_AI_ENGINEER")

    # Run spec guardrails
    print("\n  Running spec guardrails...")
    try:
        validate_submission(submission_df, args.candidates)
        print("  ✅ All spec guardrails passed")
    except AssertionError as e:
        print(f"  ❌ Spec guardrail FAILED: {e}")
        print("  Submission CSV was NOT written.")
        return

    # Write CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    submission_df.to_csv(args.out, index=False, encoding="utf-8")

    # Also write validated copy
    validated_path = os.path.join(
        os.path.dirname(args.out), "ranked_candidates_validated.csv"
    )
    submission_df.to_csv(validated_path, index=False, encoding="utf-8")

    elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"  ✅ Inference Complete in {elapsed:.2f}s")
    print(f"  Saved: {args.out}")
    print(f"  Saved: {validated_path}")
    print(f"  Score range: {submission_df['score'].min():.4f} – {submission_df['score'].max():.4f}")
    print(f"  Unique scores: {submission_df['score'].nunique()}/{len(submission_df)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
