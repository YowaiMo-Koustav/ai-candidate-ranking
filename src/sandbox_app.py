"""
sandbox_app.py — Streamlit sandbox for the AI Candidate Ranking pipeline.

Spec requirement (Section 10.5): accepts ≤100 candidates via upload or
pre-loaded, runs full pipeline end-to-end, produces ranked CSV.
Works on Streamlit Cloud without local data files.
"""

import os
import sys
import json
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.embeddings import load_encoder_model, build_candidate_text
from src.features import compute_all_features
from src.scoring import compute_final_score, generate_reasoning

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="AI Candidate Ranker — NPCsWithWifi", layout="wide")
st.title("🎯 Intelligent Candidate Discovery & Ranking")
st.caption("Team NPCsWithWifi — Sandbox Demo (≤100 candidates)")

# ── Default JD text (fallback when .docx is unavailable) ─────────────
DEFAULT_JD = """
Senior AI Engineer — Semantic Search & Ranking

We are looking for an experienced AI Engineer to design, build, and optimise
semantic search and candidate-ranking systems.

Key responsibilities:
- Build and maintain embedding-based retrieval pipelines using dense vector
  representations (Sentence Transformers, FAISS, Milvus, pgvector).
- Design ranking models that combine semantic similarity with structured
  candidate signals (experience, skills, behavioural data).
- Implement evaluation frameworks (NDCG, MAP, P@K) and run offline/online
  experiments to improve ranking quality.
- Collaborate with product and engineering to integrate search into the
  hiring platform.

Requirements:
- 5-8 years of industry experience with at least 2 years in ML/AI roles.
- Hands-on experience with PyTorch or TensorFlow, NLP models, and
  embedding/vector search systems.
- Familiarity with MLOps practices (model versioning, CI/CD for ML,
  experiment tracking).
- Strong software engineering fundamentals (Python, APIs, data pipelines).

Nice to have:
- Experience with recommendation systems or learning-to-rank.
- Knowledge of OpenSearch / Elasticsearch.
- Published research or open-source contributions in NLP/IR.
""".strip()


def load_jd_text():
    """Try to load JD from .docx, fall back to embedded default."""
    docx_path = os.path.join(ROOT_DIR, "data", "raw", "job_description.docx")
    if os.path.exists(docx_path):
        try:
            from docx import Document
            doc = Document(docx_path)
            text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
            if text:
                return text
        except Exception:
            pass
    return DEFAULT_JD


def find_sample_candidates():
    """Find pre-loaded sample candidates from various locations."""
    paths = [
        os.path.join(ROOT_DIR, "data", "sample", "small_candidates.jsonl"),
        os.path.join(ROOT_DIR, "data", "raw", "sample_candidates.json"),
    ]
    for p in paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def parse_candidates_file(file_content: str):
    """Parse uploaded or loaded candidate data (JSONL or JSON array)."""
    content = file_content.strip()
    candidates = []
    if content.startswith("["):
        # JSON array
        candidates = json.loads(content)
    else:
        # JSONL
        for line in content.split("\n"):
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


@st.cache_resource
def get_model():
    """Load the sentence-transformer model (cached)."""
    return load_encoder_model()


def run_pipeline(candidates_list, jd_text, top_k=20):
    """Run the full ranking pipeline on a list of candidate dicts."""
    model = get_model()

    # 1. Feature engineering
    features = []
    texts = []
    for c in candidates_list:
        features.append(compute_all_features(c))
        texts.append(build_candidate_text(c))

    features_df = pd.DataFrame(features)

    # 2. Encode JD and candidates
    jd_emb = model.encode([jd_text])
    cand_embs = model.encode(texts, batch_size=32, show_progress_bar=False)

    # 3. Semantic similarity
    if cand_embs.shape[0] > 0:
        similarities = cosine_similarity(jd_emb, cand_embs)[0]
    else:
        similarities = np.zeros(len(features_df))

    features_df["semantic_similarity"] = similarities

    # 4. Final scoring
    max_ml = features_df["ml_years_estimate"].max()
    if max_ml == 0:
        max_ml = 1.0
    features_df["final_score"] = features_df.apply(
        lambda r: compute_final_score(r, max_ml_years=max_ml), axis=1
    )

    # 5. Filter and rank
    work_df = features_df[features_df["honeypot_risk_score"] < 0.6].copy()
    if work_df.empty:
        work_df = features_df.copy()  # Don't filter if everything is flagged

    top_df = work_df.sort_values("final_score", ascending=False).head(top_k).copy()
    top_df["rank"] = range(1, len(top_df) + 1)
    top_df["score"] = top_df["final_score"].round(4)

    # Enforce monotonically non-increasing
    prev = top_df.iloc[0]["score"]
    adj = []
    for s in top_df["score"]:
        if s > prev:
            adj.append(prev)
        else:
            adj.append(s)
            prev = s
    top_df["score"] = adj

    # 6. Generate reasoning
    cand_dict = {c["candidate_id"]: c for c in candidates_list}
    reasonings = []
    for _, row in top_df.iterrows():
        cid = row["candidate_id"]
        reasonings.append(generate_reasoning(row, cand_raw=cand_dict.get(cid)))
    top_df["reasoning"] = reasonings

    # 7. Add display columns
    top_df["current_title"] = top_df["candidate_id"].apply(
        lambda cid: cand_dict.get(cid, {}).get("profile", {}).get("current_title", "Unknown")
    )

    return top_df


# ══════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════

# Sidebar
st.sidebar.header("📋 Data Source")
data_source = st.sidebar.radio(
    "Choose candidate source:",
    ["Pre-loaded sample", "Upload JSONL/JSON file"],
    index=0,
)

st.sidebar.header("🔧 Settings")
top_k = st.sidebar.slider("Top K Candidates", 5, 100, 20, 5)

# Load candidates based on source
candidates_list = []

if data_source == "Upload JSONL/JSON file":
    uploaded = st.sidebar.file_uploader(
        "Upload candidates (.jsonl or .json)",
        type=["jsonl", "json"],
    )
    if uploaded is not None:
        raw = uploaded.read().decode("utf-8")
        try:
            candidates_list = parse_candidates_file(raw)
            st.sidebar.success(f"Loaded {len(candidates_list)} candidates from upload.")
        except Exception as e:
            st.sidebar.error(f"Failed to parse file: {e}")
    else:
        st.info("👈 Upload a candidates JSONL or JSON file in the sidebar to get started.")
else:
    sample_path = find_sample_candidates()
    if sample_path:
        with open(sample_path, "r", encoding="utf-8") as f:
            candidates_list = parse_candidates_file(f.read())
        st.sidebar.success(f"Loaded {len(candidates_list)} pre-loaded candidates.")
    else:
        st.warning(
            "No pre-loaded sample found. Upload a file instead, or add "
            "`data/sample/small_candidates.jsonl` to the repo."
        )

# JD Section
st.header("1. Job Description")
jd_text = st.text_area("Edit or paste JD text:", value=load_jd_text(), height=200)

# Run pipeline
if candidates_list:
    if st.button("🚀 Rank Candidates", type="primary"):
        with st.spinner(f"Ranking {len(candidates_list)} candidates..."):
            top_df = run_pipeline(candidates_list, jd_text, top_k=top_k)

        st.success(f"✅ Ranking complete! Showing top {len(top_df)} candidates.")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Score Range", f"{top_df['score'].min():.3f}–{top_df['score'].max():.3f}")
        col2.metric("Candidates", len(top_df))
        col3.metric("Unique Scores", top_df["score"].nunique())
        col4.metric("Unique Reasonings", top_df["reasoning"].nunique())

        # Results table
        st.header("2. Ranked Results")
        display_cols = ["rank", "candidate_id", "current_title", "score", "reasoning"]
        st.dataframe(top_df[display_cols], use_container_width=True, height=600)

        # Download — spec-compliant 4-column CSV
        submission_cols = ["candidate_id", "rank", "score", "reasoning"]
        csv_data = top_df[submission_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download Top {len(top_df)} as CSV",
            data=csv_data,
            file_name="ranked_candidates.csv",
            mime="text/csv",
        )

        # Expandable reasoning samples
        st.header("3. Reasoning Samples")
        for _, row in top_df.head(5).iterrows():
            with st.expander(
                f"Rank {int(row['rank'])} — {row['candidate_id']} "
                f"({row['current_title']}) — Score: {row['score']:.4f}"
            ):
                st.write(row["reasoning"])

elif data_source == "Pre-loaded sample":
    st.info("No candidates available. Upload a file or add sample data to the repository.")

# Footer
st.markdown("---")
st.caption(
    "Team **NPCsWithWifi** — ANSHU RAJ · KOUSTAV MALLICK · SOUMYADIP MANDAL  \n"
    "India Runs Data & AI Hiring Challenge"
)
