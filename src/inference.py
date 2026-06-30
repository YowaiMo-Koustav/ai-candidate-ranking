"""
inference.py — CLI entry point for the full candidate ranking pipeline.

Scores ALL 100,000 candidates, then outputs the TOP 100 ranked.
Outputs a spec-compliant CSV: candidate_id, rank, score, reasoning.
Per submission_spec.docx: exactly 100 rows, ranks 1–100.
"""

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


TOP_K = 100  # Per submission spec: exactly 100 candidates


def validate_submission(submission_df, candidates_path):
    """
    Run spec guardrail assertions on the submission DataFrame.
    Matches submission_spec.docx requirements exactly.
    """
    import re

    # 1. Exactly 100 rows
    assert len(submission_df) == TOP_K, (
        f"Expected {TOP_K} rows, got {len(submission_df)}"
    )

    # 2. Exactly 4 columns in correct order
    expected_cols = ["candidate_id", "rank", "score", "reasoning"]
    assert list(submission_df.columns) == expected_cols, (
        f"Expected columns {expected_cols}, got {list(submission_df.columns)}"
    )

    # 3. Ranks 1..100 with no gaps or duplicates
    ranks = submission_df["rank"].tolist()
    assert sorted(set(ranks)) == list(range(1, TOP_K + 1)), (
        f"Ranks must be 1..{TOP_K} with no gaps or duplicates"
    )

    # 4. No duplicate candidate IDs
    ids = submission_df["candidate_id"].tolist()
    assert len(set(ids)) == len(ids), (
        "Duplicate candidate_id values found in submission"
    )

    # 5. All IDs match CAND_XXXXXXX format
    pattern = re.compile(r"^CAND_\d{7}$")
    bad_ids = [cid for cid in ids if not pattern.match(cid)]
    assert not bad_ids, (
        f"IDs not matching CAND_XXXXXXX format: {bad_ids[:5]}"
    )

    # 6. All IDs exist in candidates.jsonl
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
        f"{len(missing)} candidate_ids not in candidates.jsonl: {missing[:5]}"
    )

    # 7. Scores in [0, 1]
    scores = submission_df["score"].tolist()
    assert all(0.0 <= s <= 1.0 for s in scores), (
        "All scores must be between 0.0 and 1.0"
    )

    # 8. Scores monotonically non-increasing (spec requirement)
    mono = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    assert mono, "Scores must be monotonically non-increasing with rank"

    # 9. Scores not all identical
    assert len(set(round(s, 6) for s in scores)) > 1, (
        "All scores are identical — scoring has a degeneration bug"
    )

    # 10. No NaN or empty values
    assert submission_df.isna().sum().sum() == 0, "NaN values found"
    assert (submission_df["reasoning"].str.strip() != "").all(), (
        "Empty reasoning strings found"
    )


def main():
    parser = argparse.ArgumentParser(
        description="AI Candidate Ranking — Inference Pipeline (Top-100)"
    )
    parser.add_argument(
        "--candidates", type=str, default="data/raw/candidates.jsonl",
        help="Path to raw candidates JSONL.",
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

    args = parser.parse_args()

    print("=" * 60)
    print("  AI Candidate Ranking — Inference (Top-100)")
    print("  Team: NPCsWithWifi")
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

    n_candidates = len(candidate_ids)
    print(f"  Loaded {n_candidates} candidate IDs")
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
    # 3. Compute Semantic Similarity (all 100k)
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n[3/7] Computing cosine similarities for {n_candidates} candidates...")

    if cand_embeddings.shape[0] > 0 and jd_embedding.shape[0] > 0:
        similarities = cosine_similarity(jd_embedding, cand_embeddings)[0]
    else:
        similarities = np.zeros(n_candidates)

    sim_df = pd.DataFrame({
        "candidate_id": candidate_ids,
        "semantic_similarity": similarities,
    })
    full_df = pd.merge(features_df, sim_df, on="candidate_id", how="inner")
    print(f"  Merged DataFrame: {full_df.shape}")
    print(f"  Similarity range: {similarities.min():.4f} – {similarities.max():.4f}")
    print(f"  ⏱  {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 4. Final Scoring (all 100k)
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n[4/7] Computing final scores for ALL {n_candidates} candidates...")

    max_ml = full_df["ml_years_estimate"].max()
    if max_ml == 0:
        max_ml = 1.0

    full_df["raw_score"] = full_df.apply(
        lambda row: compute_final_score(row, max_ml_years=max_ml), axis=1
    )

    # Rescale scores to use the full [0, 1] range
    # The raw scoring formula produces compressed values (~0.2–0.7).
    # Spec example shows 0.412–0.987, so we min-max rescale to spread scores.
    raw_min = full_df["raw_score"].min()
    raw_max = full_df["raw_score"].max()
    if raw_max > raw_min:
        full_df["final_score"] = (
            (full_df["raw_score"] - raw_min) / (raw_max - raw_min)
        )
    else:
        full_df["final_score"] = 0.5

    print(f"  Raw score range: {raw_min:.4f} – {raw_max:.4f}")
    print(f"  Rescaled range:  {full_df['final_score'].min():.4f} – {full_df['final_score'].max():.4f}")
    print(f"  Unique scores: {full_df['final_score'].nunique()}")
    print(f"  ⏱  {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 5. Select Top-100
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n[5/7] Selecting top {TOP_K} candidates from {n_candidates}...")

    # Filter high-risk honeypots before ranking
    filtered_df = full_df[full_df["honeypot_risk_score"] < 0.6].copy()
    honeypot_count = n_candidates - len(filtered_df)

    filtered_df["score"] = filtered_df["final_score"].round(4)

    # Sort by score DESC, then candidate_id ASC for deterministic tie-breaking
    # Per spec §3: "Break score ties deterministically using a secondary signal
    # from your model, or by candidate_id ascending."
    ranked_df = filtered_df.sort_values(
        by=["score", "candidate_id"],
        ascending=[False, True],
    ).head(TOP_K).copy()

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

    print(f"  Filtered out {honeypot_count} honeypot candidates")
    print(f"  Selected top {TOP_K} from {len(filtered_df)} remaining")
    print(f"  Score range (top {TOP_K}): {ranked_df['score'].min():.4f} – {ranked_df['score'].max():.4f}")
    print(f"  ⏱  {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 6. Generate JD-Aware Reasoning (top 100 only)
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n[6/7] Generating context-aware reasoning for top {TOP_K}...")

    top_ids = set(ranked_df["candidate_id"].values)
    cand_details = load_candidates_for_ids(args.candidates, top_ids)
    print(f"  Loaded raw details for {len(cand_details)}/{len(top_ids)} candidates")

    reasonings = []
    for _, row in ranked_df.iterrows():
        cid = row["candidate_id"]
        cand_raw = cand_details.get(cid)
        reasonings.append(generate_reasoning(row, cand_raw=cand_raw))
    ranked_df["reasoning"] = reasonings

    unique_reasons = ranked_df["reasoning"].nunique()
    print(f"  Unique reasonings: {unique_reasons}/{TOP_K}")
    print(f"  ⏱  {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 7. Format and Export — exactly 4 columns per spec
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n[7/7] Formatting and exporting submission...")

    submission_df = ranked_df[["candidate_id", "rank", "score", "reasoning"]].copy()

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
    print(f"  ✅ Inference Complete in {elapsed:.1f}s")
    print(f"  Saved: {args.out}")
    print(f"  Saved: {validated_path}")
    print(f"  Rows: {len(submission_df)} | Columns: {list(submission_df.columns)}")
    print(f"  Score range: {submission_df['score'].min():.4f} – {submission_df['score'].max():.4f}")
    print(f"  Unique scores: {submission_df['score'].nunique()}/{len(submission_df)}")
    print(f"  Unique reasonings: {submission_df['reasoning'].nunique()}/{len(submission_df)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
