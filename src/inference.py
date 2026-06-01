import argparse
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import build_jd_text
from src.embeddings import load_encoder_model
from src.scoring import compute_final_score, generate_reasoning

def main():
    parser = argparse.ArgumentParser(description="Run Candidate Ranking Inference (Offline)")
    parser.add_argument("--candidates", type=str, default="data/raw/candidates.jsonl",
                        help="Path to raw candidates (used for reasoning enrichment).")
    parser.add_argument("--job-desc", type=str, default="data/raw/job_description.docx",
                        help="Path to the Job Description word document.")
    parser.add_argument("--embeddings", type=str, default="data/processed/embeddings.npy",
                        help="Path to precomputed candidate embeddings.")
    parser.add_argument("--candidate-ids", type=str, default="data/processed/candidate_ids.npy",
                        help="Path to precomputed candidate ID array.")
    parser.add_argument("--features", type=str, default="data/processed/candidates_feather.parquet",
                        help="Path to precomputed tabular features parquet/feather file.")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to central configuration.")
    parser.add_argument("--out", type=str, default="outputs/submissions/ranked_candidates.csv",
                        help="Path to save the final submission CSV.")
    
    args = parser.parse_args()
    
    print("=== Starting Inference Pipeline ===")
    start_time = time.time()
    
    # 1. Load Precomputed Artifacts
    print(f"Loading precomputed artifacts from {args.candidate_ids} and {args.embeddings}...")
    candidate_ids = np.load(args.candidate_ids, allow_pickle=True)
    cand_embeddings = np.load(args.embeddings)
    
    print(f"Loading features from {args.features}...")
    features_df = pd.read_parquet(args.features)
    
    # 2. Build and Encode JD
    print(f"Loading Job Description from {args.job_desc}...")
    jd_text = build_jd_text(args.job_desc)
    
    print("Loading Sentence-Transformers Model (offline cached)...")
    # Will not download weights if already cached by huggingface/sentence-transformers
    model = load_encoder_model()
    
    print("Encoding JD text...")
    jd_embedding = model.encode([jd_text]) if jd_text else np.array([])
    
    # 3. Compute Semantic Similarity
    print("Computing cosine similarities...")
    if cand_embeddings.shape[0] > 0 and jd_embedding.shape[0] > 0:
        # Cosine similarity returns shape (1, N) here
        similarities = cosine_similarity(jd_embedding, cand_embeddings)[0]
    else:
        similarities = np.zeros(len(candidate_ids))
        
    sim_df = pd.DataFrame({"candidate_id": candidate_ids, "semantic_similarity": similarities})
    
    # Join the precomputed features with the fresh semantic similarity scores
    full_df = pd.merge(features_df, sim_df, on="candidate_id", how="inner")
    
    # 4. Final Scoring
    print("Applying heuristic scoring rules...")
    max_ml = full_df["ml_years_estimate"].max() if "ml_years_estimate" in full_df.columns else 1.0
    full_df["final_score"] = full_df.apply(lambda row: compute_final_score(row, max_ml_years=max_ml), axis=1)
    
    # 5. Filtering and Ranking
    # Safely filter out high-risk honeypot targets before generating the Top 100
    filtered_df = full_df[full_df["honeypot_risk_score"] < 0.6].copy()
    ranked_df = filtered_df.sort_values(by="final_score", ascending=False).head(100).copy()
    
    # 6. Extract Raw Details for Reasoning Generation
    # We load just the raw records for the top 100 to populate exact skill lists in the output
    print("Extracting raw candidate details for reasoning generation...")
    top_100_ids = set(ranked_df["candidate_id"].values)
    cand_details = {}
    
    cands_path = args.candidates
    if not os.path.exists(cands_path):
        # Fallback to sample data if full jsonl isn't present
        alt_path = "data/raw/sample_candidates.json"
        if os.path.exists(alt_path):
            cands_path = alt_path
            
    try:
        with open(cands_path, 'r', encoding="utf-8") as f:
            first_char = f.read(1)
            
        if first_char == '[':
            with open(cands_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for cand in data:
                    cid = cand.get("candidate_id")
                    if cid in top_100_ids:
                        cand_details[cid] = cand
        else:
            with open(cands_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    cand = json.loads(line)
                    cid = cand.get("candidate_id")
                    if cid in top_100_ids:
                        cand_details[cid] = cand
    except Exception as e:
        print(f"Warning: Could not load raw data for reasoning enrichment: {e}")
        
    # Generate explanatory strings
    print("Generating context-aware reasoning...")
    def safe_reasoning(row):
        cid = row["candidate_id"]
        return generate_reasoning(row, cand_raw=cand_details.get(cid))
        
    ranked_df["reasoning"] = ranked_df.apply(safe_reasoning, axis=1)
    
    # 7. Format Final Submission DataFrame
    print("Formatting final submission...")
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
    
    # Select columns exactly in the requested order
    submission_df = ranked_df[["candidate_id", "rank", "score", "reasoning"]].copy()
    
    # 8. Export CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    submission_df.to_csv(args.out, index=False, encoding="utf-8")
    
    elapsed = time.time() - start_time
    print(f"=== Inference Complete in {elapsed:.2f}s ===")
    print(f"Saved ranked candidates to {args.out}")

if __name__ == "__main__":
    main()
