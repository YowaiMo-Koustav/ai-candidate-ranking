import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Add root project dir to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_candidates, build_jd_text
from src.embeddings import load_encoder_model, build_candidate_text
from src.features import compute_all_features
from src.scoring import compute_final_score, generate_reasoning

st.set_page_config(page_title="AI Candidate Ranker", layout="wide")
st.title("🎯 Intelligent Candidate Discovery & Ranking")
st.caption("Sandbox demo — ranks a small candidate subset using the full pipeline")

@st.cache_resource
def load_model():
    return load_encoder_model()

@st.cache_data
def load_sample_dataset():
    """
    Loads small_candidates.jsonl, computes features, and embeds them.
    This runs exactly once at startup.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Try small_candidates.jsonl first, then sample_candidates.json
    cands_path = os.path.join(base_dir, "data", "sample", "small_candidates.jsonl")
    if not os.path.exists(cands_path) or os.path.getsize(cands_path) == 0:
        cands_path = os.path.join(base_dir, "data", "raw", "sample_candidates.json")
    if not os.path.exists(cands_path) or os.path.getsize(cands_path) == 0:
        return [], pd.DataFrame(), np.array([])

    cands_list = list(load_candidates(cands_path, limit=500))
    if not cands_list:
        return [], pd.DataFrame(), np.array([])

    features = []
    texts = []
    for c in cands_list:
        feat = compute_all_features(c)
        features.append(feat)
        texts.append(build_candidate_text(c))

    df = pd.DataFrame(features)

    # Load model and encode
    model = load_model()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False) if texts else np.array([])

    return cands_list, df, embeddings

# Initialize resources
model = load_model()
with st.spinner("Loading and precomputing small candidate dataset..."):
    cands_raw, features_df, cand_embeddings = load_sample_dataset()

if features_df.empty:
    st.error(
        "No candidate data found. Please ensure `data/sample/small_candidates.jsonl` "
        "contains at least one candidate record."
    )
    st.stop()

st.success(f"Loaded {len(cands_raw)} candidates with {features_df.shape[1]} features each.")

# UI Layout
st.header("1. Job Description")
default_jd_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "data", "raw", "job_description.docx",
)
default_jd_text = (
    build_jd_text(default_jd_path)
    if os.path.exists(default_jd_path)
    else "Looking for an AI Engineer with PyTorch experience..."
)

jd_input = st.text_area("Edit or paste JD text below:", value=default_jd_text, height=200)

# Sidebar filters
st.sidebar.header("🔧 Filters")
min_ml_years = st.sidebar.slider(
    "Minimum ML Years", min_value=0.0, max_value=15.0, value=0.0, step=0.5
)
min_product_score = st.sidebar.slider(
    "Minimum Product-Company Score", min_value=0.0, max_value=1.0, value=0.0, step=0.1
)
top_k = st.sidebar.slider(
    "Top K Candidates", min_value=5, max_value=50, value=20, step=5
)

if st.button("Rank Candidates", type="primary"):
    with st.spinner("Ranking candidates..."):
        # 1. Encode JD
        jd_emb = model.encode([jd_input])

        # 2. Compute similarity
        if cand_embeddings.shape[0] > 0:
            similarities = cosine_similarity(jd_emb, cand_embeddings)[0]
        else:
            similarities = np.zeros(len(features_df))

        # 3. Join and score
        work_df = features_df.copy()
        work_df["semantic_similarity"] = similarities

        max_ml = work_df["ml_years_estimate"].max()
        work_df["final_score"] = work_df.apply(
            lambda r: compute_final_score(r, max_ml_years=max_ml), axis=1
        )

        # 4. Apply filters
        work_df = work_df[work_df["honeypot_risk_score"] < 0.6]
        work_df = work_df[work_df["ml_years_estimate"] >= min_ml_years]
        work_df = work_df[work_df["product_company_score"] >= min_product_score]

        if work_df.empty:
            st.warning("No candidates match the current filters. Try relaxing the constraints.")
        else:
            top_df = work_df.sort_values(by="final_score", ascending=False).head(top_k).copy()

            # 5. Reasoning and formatting
            cand_dict = {c["candidate_id"]: c for c in cands_raw}

            def safe_reasoning(row):
                cid = row["candidate_id"]
                return generate_reasoning(row, cand_raw=cand_dict.get(cid))

            top_df["reasoning"] = top_df.apply(safe_reasoning, axis=1)
            top_df["score"] = top_df["final_score"].round(3)
            top_df["rank"] = range(1, len(top_df) + 1)

            # Pull original title for display
            top_df["current_title"] = top_df["candidate_id"].apply(
                lambda cid: cand_dict[cid].get("profile", {}).get("current_title", "Unknown")
                if cid in cand_dict else "Unknown"
            )

            out_cols = ["rank", "candidate_id", "current_title", "score", "reasoning"]
            final_display_df = top_df[out_cols]

            st.success(f"Ranking complete! Showing top {len(final_display_df)} candidates.")

            # Score distribution
            col1, col2, col3 = st.columns(3)
            col1.metric("Score Range", f"{top_df['score'].min():.3f} – {top_df['score'].max():.3f}")
            col2.metric("Candidates Shown", len(final_display_df))
            col3.metric("Unique Scores", top_df["score"].nunique())

            st.dataframe(final_display_df, use_container_width=True)

            # Download button
            csv = final_display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download Top {len(final_display_df)} as CSV",
                data=csv,
                file_name="ranked_candidates.csv",
                mime="text/csv",
            )
