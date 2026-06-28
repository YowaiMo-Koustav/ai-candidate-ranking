import argparse
import os
import time
try:
    import psutil
except ImportError:
    psutil = None
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from src.data_loader import load_candidates
from src.embeddings import load_encoder_model, build_candidate_text
from src.features import compute_all_features

def main():
    parser = argparse.ArgumentParser(description="Offline Precomputation Pipeline for Candidates")
    parser.add_argument("--candidates", type=str, default="data/raw/candidates.jsonl",
                        help="Path to raw candidates.jsonl.")
    parser.add_argument("--embeddings-out", type=str, default="data/processed/embeddings.npy",
                        help="Output path for embeddings array.")
    parser.add_argument("--ids-out", type=str, default="data/processed/candidate_ids.npy",
                        help="Output path for candidate IDs array.")
    parser.add_argument("--features-out", type=str, default="data/processed/candidates_feather.parquet",
                        help="Output path for features parquet/feather.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of candidates processed for debugging.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for generating embeddings.")
    
    args = parser.parse_args()
    
    print("=== Starting Full Dataset Precomputation ===")
    start_time = time.time()
    
    cands_path = args.candidates
    if not os.path.exists(cands_path):
        print(f"Error: Candidates file not found at {cands_path}")
        print("Please place candidates.jsonl in data/raw/ before running precompute.")
        return
            
    candidate_texts_full = []
    candidate_ids_full = []
    features_full = []
    
    print(f"Streaming and feature engineering from {cands_path}...")
    
    # Read stream using load_candidates
    cand_stream = load_candidates(cands_path, limit=args.limit)
    
    # Using a list comprehension or just a loop to process
    for cand in tqdm(cand_stream, desc="Processing candidates"):
        cid = cand.get("candidate_id")
        if not cid: continue
        
        # Build raw candidate string representation
        text = build_candidate_text(cand)
        
        # Calculate structured features
        feats = compute_all_features(cand)
        
        candidate_ids_full.append(cid)
        candidate_texts_full.append(text)
        features_full.append(feats)
        
    print(f"Processed {len(candidate_ids_full)} candidates.")
    time_parsing = time.time() - start_time
    print(f"Parsing & Feature Eng took: {time_parsing:.2f} seconds.")
    
    # 2. Extract Embeddings
    print("Loading Sentence-Transformers Model...")
    model = load_encoder_model()
    
    print("Encoding all candidates (this may take a while)...")
    start_emb = time.time()
    if candidate_texts_full:
        full_embeddings = model.encode(candidate_texts_full, batch_size=args.batch_size, show_progress_bar=True)
    else:
        full_embeddings = np.array([])
    time_emb = time.time() - start_emb
    print(f"Encoding took: {time_emb:.2f} seconds.")
    
    # 3. Save artifacts
    print("Saving processed artifacts...")
    os.makedirs(os.path.dirname(args.embeddings_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.ids_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.features_out), exist_ok=True)
    
    np.save(args.embeddings_out, full_embeddings)
    np.save(args.ids_out, np.array(candidate_ids_full))
    
    full_features_df = pd.DataFrame(features_full)
    full_features_df.to_parquet(args.features_out, index=False)
    
    total_time = time.time() - start_time
    print("=== Precomputation Complete ===")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Candidates processed: {len(candidate_ids_full)}")
    if psutil:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Peak Memory Usage: {mem_mb:.2f} MB")

if __name__ == "__main__":
    main()
