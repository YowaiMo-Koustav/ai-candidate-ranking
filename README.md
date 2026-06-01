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

## ⚡ Quick Start (Reproduce Submission)

Follow these steps to reproduce the final submission exactly according to the hackathon specifications.

### 1. Environment Setup

Create a Python 3.9+ virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Place Official Bundle Files

Ensure the official hackathon data bundle is extracted directly into the `data/raw/` directory. Your `data/raw/` folder must contain these exact files:

- `candidates.jsonl`
- `job_description.docx`
- `candidate_schema.json`
- `redrob_signals_doc.docx`
- `submission_spec.docx`
- `sample_candidates.json`
- `validate_submission.py`
- `submission_metadata_template.yaml`
- `sample_submission.csv`

### 3. Step 1: Offline Precomputation

Before running the final inference, we must precompute the heavy feature engineering and `sentence-transformer` embeddings. 
*(Note: This step can take 20–40 minutes depending on hardware, but it runs **offline** and is exempt from the 5-minute inference budget).*

```bash
python -m src.precompute
```
This generates `embeddings.npy`, `candidate_ids.npy`, and `candidates_feather.parquet` in the `data/processed/` directory.

### 4. Step 2: Fast CPU Inference

Run the final ranking pipeline. This step respects the **strict 5-minute, CPU-only, 16GB RAM, no-network constraint** by loading the pre-downloaded model cache and the precomputed artifacts from `data/processed/`.

```bash
python -m src.inference \
  --candidates data/raw/candidates.jsonl \
  --job-desc data/raw/job_description.docx \
  --out outputs/submissions/TEAM_ID.csv
```

### 5. Validate Submission

Verify that the final output perfectly matches the hackathon CSV specifications and schema:

```bash
python data/raw/validate_submission.py outputs/submissions/TEAM_ID.csv
```

---

## 🚀 Interactive Streamlit Sandbox

Want to test the pipeline interactively on a small subset of candidates? We built a Streamlit app to visualize the ranking logic instantly.

```bash
streamlit run src/sandbox_app.py
```

- **Dynamic Job Description**: Edit the JD text on the fly.
- **Sub-Second Inference**: Ranks the cached candidates instantly against your custom JD.
- **Explainability**: View clear, human-readable reasonings for the top 20 candidates.

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

## 📊 Feature Engineering Highlights

We built 15+ complex features directly reflecting actual recruitment parameters:
- **Role Title Score**: Prioritizes explicit ML/AI/Search titles.
- **Product Company Affinity**: Penalizes candidates deeply rooted in pure IT services unless they have explicit product ML roles.
- **Experience Band Match**: Peaks around 6-8 years to target senior talent without favoring massive unrelated histories.
- **Behavioral Signals**: Leverages Redrob data like `recruiter_response_rate` and `last_active_date`.
- **Honeypot Penalty**: Strictly penalizes candidates with 15+ AI skills but less than 1 year of estimated ML working experience.

---

## 🏗️ Project Architecture

```
ai-candidate-ranking/
├── README.md                          
├── requirements.txt                   
├── submission_metadata.yaml           
├── validate_submission.py             
├── configs/
│   └── config.yaml                    
├── data/
│   ├── raw/                           # Official hackathon bundle files
│   ├── processed/                     # Precomputed artifacts
│   └── sample/                        # Small sample for local testing
├── notebooks/
│   ├── 01_eda.ipynb                   
│   ├── 02_pipeline_dev.ipynb          
│   └── 03_inference_submission.ipynb  
├── src/
│   ├── __init__.py                    
│   ├── data_loader.py                 # Lazy-loading JSON streams & JD parsing
│   ├── embeddings.py                  # Text aggregration & sentence-transformers logic
│   ├── features.py                    # Recruiter feature engineering heuristics
│   ├── scoring.py                     # Weighted scoring fusion & explanation generator
│   ├── precompute.py                  # CLI entrypoint for batch precomputation
│   ├── inference.py                   # CLI entrypoint for fast CPU inference
│   └── sandbox_app.py                 # Interactive Streamlit testing sandbox
└── outputs/
    └── submissions/                   # Final ranked CSV outputs
```

---

## 📝 License

Built for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

---

*Built with ❤️ for better hiring.*
