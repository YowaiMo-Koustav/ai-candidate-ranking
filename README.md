# 🎯 Redrob Intelligent Candidate Discovery & Ranking

> **Hackathon Submission** — An AI-powered candidate ranking system that thinks like a recruiter: semantic understanding of job fit, structured evaluation of experience & behavioral signals, and built-in honeypot detection — all within a strict **5-minute CPU-only compute budget**.

---

## 📌 Problem Statement

Given a single Job Description (Senior AI Engineer — Founding Team at Redrob AI) and **100,000 candidate profiles** in JSONL format, produce a **ranked top-100 CSV** scored on relevance. The system must:

- Run the final ranking step on **CPU-only, 16 GB RAM, no network access**, under **5 minutes**.
- Generate **unique, human-readable reasoning** for every ranked candidate.
- Detect and demote **~80 honeypot candidates** (subtly impossible profiles).
- Produce a submission matching the exact CSV spec: `candidate_id, rank, score, reasoning` — top 100 candidates.

---

## ⚡ Quick Start — Reproduce the Submission

### 1. Clone & Setup

```bash
git clone https://github.com/YowaiMo-Koustav/ai-candidate-ranking.git
cd ai-candidate-ranking

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Place Official Data Bundle

Extract the hackathon data into `data/raw/`:

```
data/raw/
├── candidates.jsonl                 # 100,000 candidate profiles
├── job_description.docx             # The target JD
├── validate_submission.py           # Official submission validator
```

### 3. Step 1 — Offline Precomputation (one-time)

This step streams all 100k candidates, engineers features, and encodes embeddings.  
**Not subject to the 5-minute budget** — runs once before inference.

```bash
python -m src.precompute
```

Generates three artifacts in `data/processed/`:
| File | Contents | Size |
|------|----------|------|
| `embeddings.npy` | 100k × 384 sentence-transformer vectors | ~146 MB |
| `candidate_ids.npy` | Ordered candidate ID array | ~4.6 MB |
| `candidates_feather.parquet` | 15 engineered features per candidate | ~1.6 MB |

### 4. Step 2 — Fast CPU Inference (under 5 min)

Loads precomputed artifacts, encodes the JD, computes similarity, scores, ranks, and writes the final CSV.

```bash
python -m src.inference \
  --candidates data/raw/candidates.jsonl \
  --job-desc data/raw/job_description.docx \
  --out outputs/submissions/team_NPCsWithWifi.csv
```

**CLI Arguments** (all have sensible defaults):

| Flag              | Default                                     | Description                  |
| ----------------- | ------------------------------------------- | ---------------------------- |
| `--candidates`    | `data/raw/candidates.jsonl`                 | Path to candidates JSONL     |
| `--job-desc`      | `data/raw/job_description.docx`             | Path to job description      |
| `--embeddings`    | `data/processed/embeddings.npy`             | Precomputed embeddings       |
| `--candidate-ids` | `data/processed/candidate_ids.npy`          | Candidate ID mapping         |
| `--features`      | `data/processed/candidates_feather.parquet` | Feature matrix               |
| `--out`           | `outputs/submissions/team_NPCsWithWifi.csv` | Output CSV path              |
| `--top-k`         | `100`                                       | Number of candidates to rank |

### 5. Validate

```bash
python data/raw/validate_submission.py \
  outputs/submissions/team_NPCsWithWifi.csv \
  submission_metadata.yaml
```

Expected output: `Validation successful!`

---

## 🚀 Interactive Sandbox (Streamlit)

A lightweight demo that lets you test the ranking pipeline interactively.

```bash
streamlit run src/sandbox_app.py
```

- **Edit the JD** on-the-fly via a text area
- **Instant re-ranking** of a small candidate subset
- **See reasoning** for the top 20 candidates
- **Download** results as CSV

---

## 🧠 System Architecture

### Two-Phase Pipeline

```
Phase 1: Offline Precomputation (no time limit)
┌──────────────┐    ┌──────────────────┐    ┌────────────────┐
│  candidates   │───▶│  Feature Engine   │───▶│  embeddings.npy│
│  .jsonl       │    │  15 heuristics   │    │  features.pqt  │
│  (100k)       │    │  + text builder  │    │  ids.npy       │
└──────────────┘    └──────────────────┘    └────────────────┘
                           ▲
                    sentence-transformers
                    all-MiniLM-L6-v2

Phase 2: Fast Inference (< 5 min, CPU-only)
┌────────────────┐    ┌────────────┐    ┌──────────────┐    ┌─────────┐
│ Load artifacts │───▶│ Encode JD  │───▶│ Score + Rank │───▶│ CSV out │
│ (< 3 sec)      │    │ + cosine   │    │ + reasoning  │    │ top 100 │
└────────────────┘    └────────────┘    └──────────────┘    └─────────┘
```

### Hybrid Scoring Model

We combine **semantic NLP** with **recruiter-domain heuristics** into a single weighted score:

| Component                    | Weight | Signal Source                                                |
| ---------------------------- | ------ | ------------------------------------------------------------ |
| **Semantic Similarity**      | 30%    | Cosine similarity between JD and candidate text embeddings   |
| **Role Title Fit**           | 10%    | Regex match against ML/AI/Search title patterns              |
| **Product Company Affinity** | 8%     | Penalizes pure IT services backgrounds (Infosys, TCS, Wipro) |
| **Experience Band Match**    | 10%    | Peaks at 6–8 years (JD requirement), tapers outside          |
| **ML Years Estimate**        | 10%    | Actual time spent in ML-titled roles from career history     |
| **AI Skill Depth**           | 5%     | Proficiency × duration weighted score across AI skills       |
| **Availability**             | 4%     | Open-to-work flag, recency, recruiter response rate          |
| **Reliability**              | 4%     | Interview completion rate, verification status               |
| **GitHub Activity**          | 3%     | Normalized GitHub activity score (0–100 → 0–1)               |
| **Geo Fit**                  | 2%     | Location preference alignment                                |
| **Work Mode Fit**            | 2%     | Hybrid/flexible preference alignment                         |

**Penalties applied after scoring:**

| Penalty       | Scale    | Trigger                              |
| ------------- | -------- | ------------------------------------ |
| Notice Period | α = 0.15 | > 30 days notice period              |
| Honeypot Risk | β = 0.30 | Impossible experience/skill patterns |

Final score = `clip(weighted_sum − penalties, 0, 1)`

---

## 🕵️ Honeypot Detection

The dataset contains ~80 synthetic candidates with subtly impossible profiles. Our detection flags three patterns:

1. **Timeline Mismatch** — Claimed career history years exceed stated YOE by 3+ years (or vice versa)
2. **Skill Stuffing** — "Expert"/"Advanced" proficiency in AI skills with ≤ 1 month of declared duration
3. **Skills vs Experience Gap** — 10+ AI/ML skills listed but < 0.5 years of actual ML role tenure

Candidates with `honeypot_risk_score ≥ 0.6` are **excluded** from the final top-100.

---

## 📊 Engineered Features (15 total)

| Feature                  | Range       | Description                                          |
| ------------------------ | ----------- | ---------------------------------------------------- |
| `role_title_score`       | [-0.5, 1.0] | ML/AI title match; negative for marketing/HR         |
| `product_company_score`  | [0, 0.8]    | Product/tech vs IT services background               |
| `total_years_experience` | continuous  | From `profile.years_of_experience`                   |
| `experience_band_match`  | [0.2, 1.0]  | Peak at 6–8 yrs per JD spec                          |
| `ml_years_estimate`      | continuous  | Calculated from career_history titles + dates        |
| `ai_core_skills_count`   | integer     | Count of AI-pattern-matching skills                  |
| `ai_skill_depth_score`   | continuous  | Weighted by proficiency × duration                   |
| `availability_score`     | [0, 1]      | Composite of open_to_work, recency, response rate    |
| `reliability_score`      | [0, 1]      | Interview completion, offer acceptance, verification |
| `github_fit_score`       | [0, 1]      | Normalized from 0–100 scale                          |
| `geo_fit_score`          | [0.2, 1.0]  | Location preference for Pune/Noida/India             |
| `work_mode_fit`          | {0.5, 1.0}  | Hybrid/flexible = 1.0                                |
| `notice_period_penalty`  | [0, 1]      | Escalates past 30 days                               |
| `honeypot_risk_score`    | [0, 1]      | Composite of 3 honeypot detectors                    |
| `semantic_similarity`    | [0, 1]      | Cosine sim (computed at inference time)              |

---

## 🏗️ Project Structure

```
ai-candidate-ranking/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── submission_metadata.yaml            # Hackathon submission metadata
│
├── configs/
│   └── config.yaml                     # Centralized configuration
│
├── data/
│   ├── raw/                            # Official hackathon data bundle
│   ├── processed/                      # Precomputed artifacts (generated)
│   │   ├── embeddings.npy              #   100k × 384 embeddings
│   │   ├── candidate_ids.npy           #   Ordered ID array
│   │   └── candidates_feather.parquet  #   15-column feature matrix
│   └── sample/
│       └── small_candidates.jsonl      # Small subset for sandbox testing
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_pipeline_dev.ipynb           # Feature engineering & dev pipeline
│   └── 03_inference_submission.ipynb   # Final inference & validated CSV
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # JSONL streaming & JD parsing
│   ├── embeddings.py                   # Text builder & sentence-transformers
│   ├── features.py                     # 15 recruiter-domain feature functions
│   ├── inference.py                    # CLI: fast CPU inference (Step 2)
│   ├── precompute.py                   # CLI: batch precompute (Step 1)
│   ├── sandbox_app.py                  # Streamlit interactive demo
│   ├── scoring.py                      # Weighted scoring fusion & reasoning
│   └── utils.py                        # Helper utilities
│
└── outputs/
    ├── models/                         # For custom trained weights (if any)
    ├── reports/                        # Generation logs and reports
    └── submissions/
        └── team_NPCsWithWifi.csv       # Final submission CSV
```

---

## 📓 Notebooks

| Notebook            | Purpose                                                                                                                          | Key Outputs                  |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| **01_eda**          | Explores the JD, schema, signals, and sample candidates. Identifies feature ideas and honeypot patterns.                         | Feature hypotheses           |
| **02_pipeline_dev** | Develops and tests the full feature engineering pipeline on a 5k dev subset. Runs full 100k precomputation.                      | `data/processed/*` artifacts |
| **03_inference**    | Loads precomputed artifacts, scores all candidates, generates reasoning, writes submission CSV, and runs the official validator. | `team_NPCsWithWifi.csv` ✅   |

---

## 🔧 Technical Decisions

| Decision                            | Rationale                                                                               |
| ----------------------------------- | --------------------------------------------------------------------------------------- |
| **all-MiniLM-L6-v2** (22M params)   | Fast enough for 100k candidates on CPU; 384-dim embeddings keep cosine sim fast         |
| **No LLM calls**                    | Budget constraint: 5 min for 100k candidates rules out per-candidate LLM evaluation     |
| **Precompute/inference split**      | Embedding 100k candidates takes ~15 min — do it offline, then inference is < 30 seconds |
| **Heuristic scoring (no ML model)** | No labeled training data (no recruiter decisions to learn from), so hand-tuned weights  |
| **Honeypot penalty, not filter**    | Soft penalty preserves borderline candidates while strongly demoting obvious fakes      |

---

## 📝 Submission Metadata

Update `submission_metadata.yaml` before submitting:

```yaml
team_name: "Your Team Name"
team_members:
  - "Member 1"
model_description: "Sentence-Transformers (all-MiniLM-L6-v2) + 15 hand-engineered recruiter-domain features with weighted heuristic scoring"
features_used:
  - "semantic_similarity"
  - "role_title_score"
  - "product_company_score"
  - "experience_band_match"
  - "ml_years_estimate"
  - "ai_skill_depth_score"
  - "availability_score"
  - "reliability_score"
  - "github_fit_score"
  - "geo_fit_score"
  - "work_mode_fit"
  - "notice_period_penalty"
  - "honeypot_risk_score"
external_data_used: false
```

---

## 📋 Submission Checklist

- [x] Exactly 100 rows (top-100 ranking)
- [x] Columns: `candidate_id`, `rank`, `score`, `reasoning`
- [x] Ranks 1–100, starting at 1
- [x] Scores monotonically non-increasing
- [x] No duplicate candidate IDs
- [x] All candidate IDs exist in `candidates.jsonl`
- [x] Unique, specific reasoning per candidate (no templates)
- [x] No honeypots in top 10
- [x] Inference completes in < 5 minutes on CPU
- [x] `validate_submission.py` passes ✅

---

_Built with ❤️ for better hiring — Redrob Intelligent Candidate Discovery & Ranking Challenge_
