<p align="center">
  <h1 align="center">🎯 AI Candidate Ranking System</h1>
  <p align="center">
    <strong>Team NPCsWithWifi</strong> — Redrob India Runs Data & AI Hiring Challenge
  </p>
  <p align="center">
    <a href="https://ai1989.streamlit.app/"><img src="https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge" alt="Live Demo"></a>
    <a href="https://github.com/YowaiMo-Koustav/ai-candidate-ranking"><img src="https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github" alt="GitHub"></a>
  </p>
</p>

---

> **TL;DR** — Given 100,000 candidate profiles and one Job Description, our system produces a **ranked top-100 CSV** in **~16 seconds on CPU** using semantic embeddings + 15 hand-engineered recruiter-domain features — with unique, evidence-backed reasoning for every candidate and built-in honeypot detection.

---

## 👥 Team

| Name | Role |
|------|------|
| **ANSHU RAJ** | Team Leader |
| **KOUSTAV MALLICK** | Member |
| **SOUMYADIP MANDAL** | Member |

---

## 📌 Problem Statement

Rank the **top 100 candidates** from a pool of 100,000 for a _Senior AI Engineer — Founding Team at Redrob AI_ position. The system must:

- Produce exactly **100 ranked rows** as a spec-compliant CSV
- Run on **CPU-only** hardware (≤ 5 min, ≤ 16 GB RAM, no network, no GPU)
- Generate **unique, specific reasoning** per candidate (no templates, no hallucination)
- Detect and filter **~80 honeypot candidates** with impossible profiles
- Pass the official `validate_submission.py` validator

---

## ⚡ Reproduce the Submission

> **Spec §10.3**: _"Your README must indicate a single command that produces the submission CSV from the candidates.jsonl and job description files."_

### Step 0 — Clone & Install

```bash
git clone https://github.com/YowaiMo-Koustav/ai-candidate-ranking.git
cd ai-candidate-ranking

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1 — Offline Precomputation _(one-time, not time-budgeted)_

> **Spec §10.3**: _"If your system requires pre-computation (e.g., generating embeddings), document this clearly — pre-computation may exceed the 5-minute budget."_

```bash
python -m src.precompute
```

This streams all 100k candidates and produces three artifacts:

| Artifact | Shape | Size | Contents |
|----------|-------|------|----------|
| `data/processed/embeddings.npy` | 100,000 × 384 | 146 MB | Dense sentence-transformer vectors |
| `data/processed/candidate_ids.npy` | 100,000 | 4.6 MB | Ordered candidate ID array |
| `data/processed/candidates_feather.parquet` | 100,000 × 15 | 1.6 MB | Engineered feature matrix |

### Step 2 — Fast CPU Inference _(the timed step — ~16 seconds)_

**Single command** to produce the submission CSV:

```bash
python -m src.inference
```

That's it. Output: `outputs/submissions/team_NPCsWithWifi.csv`

<details>
<summary><strong>All CLI arguments (with defaults)</strong></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--candidates` | `data/raw/candidates.jsonl` | Path to 100k candidates |
| `--job-desc` | `data/raw/job_description.docx` | Job description file |
| `--embeddings` | `data/processed/embeddings.npy` | Precomputed embeddings |
| `--candidate-ids` | `data/processed/candidate_ids.npy` | Candidate ID array |
| `--features` | `data/processed/candidates_feather.parquet` | Feature matrix |
| `--out` | `outputs/submissions/team_NPCsWithWifi.csv` | Output CSV |

</details>

### Step 3 — Validate

```bash
python data/raw/validate_submission.py outputs/submissions/team_NPCsWithWifi.csv submission_metadata.yaml
```

Expected: **`Validation successful!`**

---

## 🧠 System Architecture

### Two-Phase Design

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  PHASE 1: Offline Precomputation (no time limit)                           ║
║                                                                            ║
║  candidates.jsonl ──► Feature Engineering ──► embeddings.npy    (146 MB)   ║
║  (100,000 profiles)   (15 heuristics)        features.parquet  (1.6 MB)   ║
║                       + text builder          ids.npy          (4.6 MB)   ║
║                       + sentence-transformers                              ║
║                         (all-MiniLM-L6-v2)                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PHASE 2: Fast Inference (~16s, CPU-only, offline)                         ║
║                                                                            ║
║  Load artifacts ──► Encode JD ──► Cosine Sim ──► Score ──► Rank ──► CSV   ║
║  (< 0.2s)           (~ 9s)        (0.1s)         (1.6s)    + reasoning    ║
║                                                              (2.5s)       ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Why Two Phases?

Embedding 100,000 candidates with `all-MiniLM-L6-v2` takes ~15 minutes — well outside the 5-min budget. By precomputing embeddings once, the actual inference step only needs to:
1. Load three numpy/parquet files (< 0.2s)
2. Encode the single JD text (~9s — model load dominates)
3. Score and rank all 100k candidates (~2s)
4. Generate reasoning for the top 100 (~2.5s)

**Total: ~16 seconds** — well within the 5-minute CPU budget.

---

## 🔬 Scoring Model

We combine **semantic NLP** with **recruiter-domain heuristics** into a single weighted score.

### Signal Weights

| Signal | Weight | What it captures |
|--------|--------|------------------|
| 🧠 **Semantic Similarity** | 30% | Cosine similarity between JD & candidate text embeddings |
| 🏷️ **Role Title Fit** | 10% | Regex match: ML/AI/Search titles score highest |
| 🏢 **Product Co. Affinity** | 8% | Product companies > IT services (Infosys, TCS, etc.) |
| 📅 **Experience Band** | 10% | Peaks at 6–8 yrs (JD sweet spot), tapers outside |
| 🤖 **ML Years Estimate** | 10% | Actual ML-titled role tenure from career history |
| 🎯 **AI Skill Depth** | 5% | Proficiency × duration across AI-relevant skills |
| 📬 **Availability** | 4% | Open-to-work flag, recency, response rate |
| ✅ **Reliability** | 4% | Interview completion rate, verification status |
| 💻 **GitHub Activity** | 3% | Normalized GitHub activity score (0–100 → 0–1) |
| 📍 **Geo Fit** | 2% | Location preference alignment |
| 🏠 **Work Mode Fit** | 2% | Hybrid/flexible preference match |

**Penalties** (applied post-scoring):

| Penalty | Scale | Trigger |
|---------|-------|---------|
| ⏰ Notice Period | α = 0.15 | > 30 days notice |
| 🚩 Honeypot Risk | β = 0.30 | Impossible experience/skill patterns |

**Final score** = `clamp(weighted_sum − penalties, 0, 1)` → then min-max rescaled across all 100k.

---

## 🕵️ Honeypot Detection

> **Spec §7**: _"The dataset contains ~80 honeypot candidates with subtly impossible profiles. If your submission ranks honeypots in the top 10, this is a strong signal your system isn't reading profiles."_

Our system flags three honeypot patterns:

| Pattern | Signal | Risk Score |
|---------|--------|------------|
| **Timeline Mismatch** | Career history years exceed stated YOE by 3+ years | +0.4 |
| **Skill Stuffing** | Expert/Advanced proficiency claimed with ≤ 1 month duration | +0.5 |
| **Skills vs Tenure Gap** | 10+ AI skills listed but < 0.5 yrs actual ML role tenure | +0.3 |

Candidates with `honeypot_risk_score ≥ 0.6` are **excluded** from the top-100 before ranking. In our final submission: **0 honeypots in top 100**.

---

## 📝 Reasoning Generation

> **Spec §3 (Table 2)**: _"We sample 10 random rows and check reasoning against: specific facts, JD connection, honest concerns, no hallucination, variation, rank consistency."_

Our reasoning engine (in `src/scoring.py`) produces **100 unique, evidence-grounded sentences** per run:

- **Specific facts**: References exact YOE, ML tenure, company type, and named skills from the profile
- **JD connection**: Maps candidate skills to JD-relevant categories (search/retrieval, embeddings/vectors, ML tools, MLOps)
- **Honest concerns**: Surfaces IT-services background, long notice periods, skill-tenure mismatches
- **No hallucination**: Only mentions skills actually present in `candidate.skills[]`
- **Variation**: Each reasoning is assembled from conditional sentence templates based on 6+ data dimensions
- **Rank consistency**: Rank-1 candidates get strong positive language; lower ranks surface trade-offs

**Sample output** (Rank 1):
> _"AI Engineer with 8 years of experience in a product-company environment, including 5.2 years in ML-focused roles. Relevant to the JD through search/retrieval (Search Systems, Recommendation Systems) and embeddings/vectors (FAISS, Sentence Transformers). Platform: 87% recruiter response rate, open to work."_

---

## 📊 All 15 Engineered Features

<details>
<summary><strong>Click to expand feature details</strong></summary>

| # | Feature | Range | Source |
|---|---------|-------|--------|
| 1 | `role_title_score` | [-0.5, 1.0] | Regex against ML/AI/Search title patterns |
| 2 | `product_company_score` | [0, 1.0] | Product company vs IT services detection |
| 3 | `total_years_experience` | continuous | `profile.years_of_experience` |
| 4 | `experience_band_match` | [0.2, 1.0] | Peak at 6–8 yrs, taper outside |
| 5 | `ml_years_estimate` | continuous | Sum of ML-titled role durations from career_history |
| 6 | `ai_core_skills_count` | integer | Count of AI-pattern-matching skills |
| 7 | `ai_skill_depth_score` | [0, 1] | Weighted by proficiency × months of experience |
| 8 | `availability_score` | [0, 1] | open_to_work × 0.3 + response_rate × 0.4 + recency × 0.3 |
| 9 | `reliability_score` | [0, 1] | Interview completion rate (clamped) |
| 10 | `github_fit_score` | [0, 1] | Raw 0–100 score normalized |
| 11 | `geo_fit_score` | [0.2, 1.0] | Location preference alignment |
| 12 | `work_mode_fit` | {0.5, 1.0} | Hybrid/flexible/remote = 1.0 |
| 13 | `notice_period_penalty` | [0, 1] | Escalates past 30 days |
| 14 | `honeypot_risk_score` | [0, 1] | Composite of 3 honeypot detectors |
| 15 | `semantic_similarity` | [0, 1] | Cosine sim (computed at inference time) |

</details>

---

## 🏗️ Project Structure

```
ai-candidate-ranking/
├── README.md                              # You are here
├── requirements.txt                       # All Python dependencies (pinned)
├── submission_metadata.yaml               # Portal metadata (Table 5 fields)
│
├── src/                                   # Core pipeline source code
│   ├── __init__.py
│   ├── data_loader.py                     # JSONL streaming, JD parsing (.docx)
│   ├── embeddings.py                      # Sentence-transformer encoding + text builder
│   ├── features.py                        # 15 recruiter-domain feature functions
│   ├── scoring.py                         # Weighted scoring fusion + reasoning generator
│   ├── inference.py                       # CLI entry point: fast CPU inference (Step 2)
│   ├── precompute.py                      # CLI entry point: batch precompute (Step 1)
│   ├── sandbox_app.py                     # Streamlit interactive demo
│   └── utils.py                           # Helper utilities
│
├── configs/
│   └── config.yaml                        # Centralized configuration
│
├── data/
│   ├── raw/                               # Official hackathon data bundle
│   │   ├── candidates.jsonl               #   100,000 candidate profiles
│   │   ├── job_description.docx           #   Target JD
│   │   ├── sample_submission.csv          #   Reference format
│   │   ├── submission_spec.docx           #   Competition rules
│   │   └── validate_submission.py         #   Official validator
│   ├── processed/                         # Precomputed artifacts (Step 1 output)
│   │   ├── embeddings.npy                 #   100k × 384 vectors (146 MB)
│   │   ├── candidate_ids.npy              #   Ordered ID array (4.6 MB)
│   │   └── candidates_feather.parquet     #   15-col feature matrix (1.6 MB)
│   └── sample/                            # Small datasets for sandbox testing
│       ├── small_candidates.jsonl         #   Pre-loaded sample for Streamlit
│       └── test_candidates_100.jsonl      #   100 synthetic test candidates
│
├── notebooks/
│   ├── 01_eda.ipynb                       # Exploratory data analysis
│   ├── 02_pipeline_dev.ipynb              # Feature engineering & dev pipeline
│   └── 03_inference_submission.ipynb      # Final inference & validated CSV
│
└── outputs/
    └── submissions/
        └── team_NPCsWithWifi.csv          # ← Final submission (100 rows)
```

---

## 📓 Notebooks

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| **01_eda** | Explores JD, candidate schema, signal distributions, and honeypot patterns | Feature hypotheses, EDA report |
| **02_pipeline_dev** | Develops and tests the full feature engineering pipeline. Runs 100k precomputation | `data/processed/*` artifacts |
| **03_inference** | Loads precomputed artifacts, scores all candidates, generates reasoning, writes submission, runs official validator | `team_NPCsWithWifi.csv` ✅ |

---

## 🚀 Live Sandbox Demo

> **Spec §10.5**: _"Your sandbox needs to: accept a small candidate sample (≤100 candidates) as input, run your ranking system end-to-end, and produce a ranked CSV."_

**🔗 [https://ai1989.streamlit.app/](https://ai1989.streamlit.app/)**

The Streamlit sandbox supports:
- **Pre-loaded sample** — Ships with a small candidate dataset for instant demo
- **File upload** — Upload any `.jsonl` or `.json` file (≤ 100 candidates)
- **Editable JD** — Modify the job description text in real-time
- **Full pipeline** — Runs feature engineering → embedding → scoring → ranking → reasoning
- **CSV download** — Download the ranked output as a spec-compliant CSV

```bash
# Run locally
streamlit run src/sandbox_app.py
```

---

## 🔧 Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **all-MiniLM-L6-v2** (22M params, 384-dim) | Fast enough for 100k candidates on CPU; good quality for job-resume semantic matching |
| **No LLM calls** | 5 min budget for 100k candidates rules out per-candidate GPT/Claude calls |
| **Precompute/inference split** | Embedding 100k takes ~15 min — precompute once, inference in ~16s |
| **Heuristic scoring (no ML model)** | No labeled training data available (no recruiter decisions to learn from) |
| **Honeypot penalty, not hard filter** | Soft penalty preserves borderline candidates; hard filter at 0.6 catches the worst |
| **Min-max rescaling** | Raw scores compressed to ~0.2–0.7; rescale across full 100k for meaningful spread |
| **Deterministic tiebreak** | Per spec: ties broken by `candidate_id` ascending after rounding |

---

## ✅ Spec Compliance Checklist

> Cross-referenced against every section of `submission_spec.docx v4`.

### §1–§3: CSV Format & Rules

- [x] Exactly **100 data rows** + 1 header
- [x] Columns: `candidate_id, rank, score, reasoning` (in order)
- [x] Filename: `team_NPCsWithWifi.csv`
- [x] Encoding: UTF-8, no BOM
- [x] Ranks 1–100, each exactly once
- [x] Each `candidate_id` exactly once
- [x] All IDs exist in `candidates.jsonl`
- [x] Scores monotonically **non-increasing** with rank
- [x] Score ties broken deterministically by `candidate_id` ascending
- [x] **100/100 unique reasonings** — no templates, no hallucination
- [x] Runtime: **~16s** (limit: 5 min)
- [x] Memory: **< 2 GB** (limit: 16 GB)
- [x] CPU-only, no GPU, no network calls

### §6: Common Rejections — All Avoided

- [x] Not 99 or 101 rows
- [x] Ranks start at 1, not 0
- [x] No duplicate `candidate_id`
- [x] No typo IDs
- [x] Scores not all identical (98/100 unique)
- [x] Rank 1 has highest score
- [x] File is `.csv`, not `.xlsx` or `.json`

### §7: Honeypot Detection

- [x] Honeypot filter active (`risk ≥ 0.6` excluded)
- [x] **0 honeypots** in top 100

### §10: Full Submission Package

- [x] **10.1** CSV file — `team_NPCsWithWifi.csv`
- [x] **10.2** Portal metadata — All Table 5 fields in `submission_metadata.yaml`
- [x] **10.3** Code repository — `README.md` + `src/` + `requirements.txt` + single repro command
- [x] **10.4** AI tools declared — Gemini, Antigravity IDE
- [x] **10.5** Sandbox — [https://ai1989.streamlit.app/](https://ai1989.streamlit.app/)
- [x] Official validator: **`Validation successful!`** ✅

---

## 📋 Submission Metadata

```yaml
team_name: "NPCsWithWifi"
team_members:
  - name: "ANSHU RAJ"      # Leader
  - name: "KOUSTAV MALLICK"
  - name: "SOUMYADIP MANDAL"
compute_environment: "MacBook Air M1, 8GB RAM, Python 3.14"
ai_tools_declared: ["Gemini", "Antigravity IDE"]
sandbox_demo_link: "https://ai1989.streamlit.app/"
external_data_used: false
```

---

<p align="center">
  Built with ❤️ by <strong>Team NPCsWithWifi</strong> for the Redrob India Runs Data & AI Hiring Challenge
</p>
