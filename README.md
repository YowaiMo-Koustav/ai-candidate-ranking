# 🎯 Redrob Intelligent Candidate Discovery & Ranking

> **Hackathon Project** — An AI system that ranks candidates the way a great recruiter would — not by matching keywords, but by actually understanding who fits the role, predicting reliability, and penalizing keyword-stuffing honeypots.

---

## 📌 Problem Overview

Recruiters go through hundreds of profiles and still often miss the right person. Not because the talent isn't there — but because pure keyword filters can't see what actually matters, and LLM evaluations are often too slow or prone to hallucinations.

**Goal:** Build a robust, scalable AI system that:
1. Understands semantic meaning behind a Job Description (JD).
2. Analyzes candidate profiles holistically (skills, experience, trajectory, redrob behavioral signals).
3. Weeds out spam/honeypot candidates with impossible timelines.
4. Produces a ranked shortlist with human-readable explanations—all within a strict 5-minute CPU-only compute budget constraint.

---

## 🧠 Hybrid Candidate Ranking Pipeline

We combine state-of-the-art NLP with hard-coded recruiter heuristics:

| Layer | What it does | How |
|-------|-------------|-----|
| **Semantic Matching** | Understands meaning beyond keywords | `sentence-transformers/all-MiniLM-L6-v2` embeddings + cosine similarity |
| **Structured Scoring** | Captures recruiter-like heuristics | Feature engineering on experience, skills overlap, geo-fit, and behavioral signals |
| **Honeypot Filtering** | Catches spam/fake profiles | Heuristics targeting impossible experience timelines and skill keyword-stuffing |
| **Explainability** | Builds trust with the recruiter | Dynamic text generation combining metrics to explain *why* a candidate ranked highly |

### Why Hybrid?
- Pure keyword matching misses semantically equivalent skills.
- Pure embedding similarity ignores hard requirements (years of experience, redrob responsiveness scores).
- A Hybrid approach mimics how a skilled recruiter actually thinks while remaining fully transparent.

---

## 🏗️ Project Architecture

```
ai-candidate-ranking/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── submission_metadata.yaml           # Hackathon metadata
├── validate_submission.py             # Hackathon format validator script
├── configs/
│   └── config.yaml                    # Central configuration
├── data/
│   ├── raw/                           # Original unmodified data (e.g., job_description.docx, candidates.jsonl)
│   ├── processed/                     # Precomputed artifacts (embeddings.npy, candidates_feather.parquet)
│   └── sample/                        # Small sample for quick local testing (small_candidates.jsonl)
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis & Honeypot Identification
│   ├── 02_pipeline_dev.ipynb          # Pipeline development & Offline 100k Batch Precomputation
│   └── 03_inference_submission.ipynb  # Final 5-minute CPU Inference & CSV generation
├── src/
│   ├── __init__.py                    
│   ├── data_loader.py                 # Lazy-loading JSON streams & JD parsing
│   ├── embeddings.py                  # Text aggregration & sentence-transformers logic
│   ├── features.py                    # 15+ Recruiter feature engineering heuristics
│   ├── scoring.py                     # Weighted scoring fusion & explanation generator
│   └── sandbox_app.py                 # Interactive Streamlit testing sandbox
└── outputs/
    └── submissions/                   # Final ranked_candidates.csv output
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.9+
- `pip`

### Installation

```bash
# Clone the repository
git clone https://github.com/YowaiMo-Koustav/ai-candidate-ranking.git
cd ai-candidate-ranking

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Interactive Streamlit Sandbox

Want to test the pipeline on a small subset of candidates instantly? We built a Streamlit app to interact with the pipeline.

```bash
streamlit run src/sandbox_app.py
```

- **Dynamic Job Description**: Edit the JD text on the fly.
- **Sub-Second Inference**: Ranks the cached candidates instantly against your custom JD.
- **Explainability**: View clear, human-readable reasonings for the top 20 candidates.

---

## 🔄 Execution Workflow for the Hackathon

The processing is split into two phases to adhere to the hackathon's compute constraints:

### 1. Offline Batch Precomputation (`02_pipeline_dev.ipynb`)
- Streams all 100,000 candidates using lazy loading to prevent out-of-memory errors.
- Computes all structured features (experience bands, skill depth, notice penalties).
- Embeds all candidate text using `sentence-transformers`.
- Saves cached artifacts to `data/processed/` (`.npy` and `.parquet`).

### 2. Fast CPU Inference (`03_inference_submission.ipynb`)
- **Strict 5-minute constraint**.
- Runs completely offline without external network calls (model is loaded from cache).
- Loads cached `.parquet` and `.npy` artifacts instantly.
- Generates JD embedding and runs lightning-fast matrix multiplication for cosine similarities.
- Applies final weighting, filters honeypots, dynamically generates text reasonings for the Top 100, and exports `ranked_candidates.csv`.
- Runs `validate_submission.py` to ensure perfect format compliance.

---

## 📊 Feature Engineering Highlights

We built 15+ complex features directly reflecting actual recruitment parameters:
- **Role Title Score**: Prioritizes explicit ML/AI/Search titles.
- **Product Company Affinity**: Penalizes candidates deeply rooted in pure IT services unless they have explicit product ML roles.
- **Experience Band Match**: Peaks around 6-8 years to target senior talent without favoring massive unrelated histories.
- **Behavioral Signals**: Leverages Redrob data like `recruiter_response_rate` and `last_active_date`.
- **Honeypot Penalty**: Strictly penalizes candidates with 15+ AI skills but less than 1 year of estimated ML working experience.

---

## 📝 License

Built for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

---

*Built with ❤️ for better hiring.*
