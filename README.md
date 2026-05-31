# 🎯 AI Candidate Ranking

> **Hackathon Project** — An AI system that ranks candidates the way a great recruiter would — not by matching keywords, but by actually understanding who fits the role.

---

## 📌 Problem Overview

Recruiters go through hundreds of profiles and still often miss the right person. Not because the talent isn't there — but because keyword filters can't see what actually matters.

**Goal:** Build an AI system that:
1. Reads a job description and understands what the role truly needs.
2. Analyzes candidate profiles holistically (skills, experience, trajectory, context).
3. Produces a ranked shortlist that a recruiter can trust.

---

## 🧠 Proposed Approach

A **hybrid candidate ranking pipeline** combining:

| Layer | What it does | How |
|-------|-------------|-----|
| **Semantic Matching** | Understands meaning beyond keywords | Sentence-transformer embeddings + cosine similarity |
| **Structured Scoring** | Captures recruiter-like heuristics | Feature engineering on experience, skills overlap, seniority, etc. |
| **Hybrid Ranker** | Combines both signals | Weighted fusion (heuristic) or learned re-ranker (if labels available) |

### Why Hybrid?

- Pure keyword matching → misses semantically equivalent skills ("ML" vs "Machine Learning").
- Pure embedding similarity → ignores hard requirements (years of experience, certifications).
- Hybrid → best of both worlds, mimics how a skilled recruiter actually thinks.

---

## 🏗️ Planned Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT                                │
│  Job Description (JD)  +  Candidate Profiles (CSV/JSON) │
└─────────────┬───────────────────────┬───────────────────┘
              │                       │
              ▼                       ▼
      ┌──────────────┐       ┌──────────────────┐
      │  Preprocess   │       │   Preprocess      │
      │  JD Text      │       │   Candidate Text  │
      └──────┬───────┘       └──────┬───────────┘
             │                       │
             ▼                       ▼
      ┌──────────────────────────────────────┐
      │        Embedding Generation          │
      │   (sentence-transformers / OpenAI)   │
      └──────────────┬───────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   ┌─────────────┐     ┌──────────────────┐
   │  Semantic    │     │  Feature         │
   │  Similarity  │     │  Engineering     │
   │  Score       │     │  (structured)    │
   └──────┬──────┘     └──────┬───────────┘
          │                    │
          ▼                    ▼
      ┌────────────────────────────┐
      │   Hybrid Scoring / Fusion  │
      │   (weighted or learned)    │
      └─────────────┬──────────────┘
                    │
                    ▼
      ┌────────────────────────────┐
      │   Ranked Candidate List    │
      │   (CSV / JSON output)      │
      └────────────────────────────┘
```

---

## 📁 Project Structure

```
ai-candidate-ranking/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── configs/
│   └── config.yaml                    # Central configuration
├── data/
│   ├── raw/                           # Original unmodified data
│   ├── processed/                     # Cleaned / transformed data
│   └── sample/                        # Small sample for quick testing
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_pipeline_dev.ipynb          # Pipeline development & experiments
│   └── 03_inference_submission.ipynb  # Final inference & output generation
├── src/
│   ├── __init__.py                    # Package init
│   ├── data_loader.py                 # Load data from various formats
│   ├── preprocess.py                  # Text cleaning & normalization
│   ├── features.py                    # Structured feature engineering
│   ├── embeddings.py                  # Embedding generation & similarity
│   ├── scoring.py                     # Heuristic & hybrid scoring
│   ├── ranker.py                      # Final ranking logic
│   ├── inference.py                   # End-to-end inference pipeline
│   └── utils.py                       # Shared utilities
└── outputs/
    ├── models/                        # Saved models / weights
    ├── submissions/                   # Final submission files
    └── reports/                       # EDA reports, visualizations
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/ai-candidate-ranking.git
cd ai-candidate-ranking

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### For Google Colab

Upload this project to Google Drive or clone from GitHub, then in a Colab cell:

```python
# Mount Drive (if using Drive)
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
%cd /content/drive/MyDrive/ai-candidate-ranking

# Install dependencies
!pip install -r requirements.txt

# Add src to path
import sys
sys.path.insert(0, './src')
```

---

## 🔍 Environment Verification

Before starting, ensure your local environment is correctly set up. This helps avoid missing dependency errors later when working in Jupyter or Colab.

### 1. Verify Local Setup
Activate your virtual environment and run the verification script:
```bash
source venv/bin/activate  # macOS/Linux
python src/env_check.py
```
This script checks your Python version and ensures all critical libraries (pandas, sentence-transformers, etc.) are installed and importable.

### 2. Verify Jupyter/Colab Setup
When you open `notebooks/01_eda.ipynb` (either locally or in Colab), run the first two cells. The first cell sets up the Colab environment (installing requirements if needed), and the second verifies that imports are working correctly within the notebook kernel.

---

## 🔄 Development Workflow

| Tool | Purpose |
|------|---------|
| **Antigravity IDE (Mac)** | Code editing, project structure, module development |
| **Google Colab (Browser)** | Running notebooks, GPU-accelerated embedding generation, experiments |
| **GitHub** | Version control, collaboration, final submission |

### Recommended Flow

1. **Edit code** in Antigravity IDE on Mac.
2. **Push to GitHub** for version control.
3. **Pull in Colab** and run notebooks for heavy computation.
4. **Iterate** — update modules locally, test in Colab.

---

## 📦 Expected Deliverables

- [ ] Ranked candidate list for a given job description
- [ ] Modular, reusable pipeline code
- [ ] EDA notebook with data insights
- [ ] Pipeline development notebook with experiments
- [ ] Inference notebook for final submission generation
- [ ] Documentation (this README)

---

## 📊 Dataset

> **Note:** The dataset schema will be integrated once the dataset is available. The current pipeline is designed to be **dataset-agnostic** — all data loading and column mapping is centralized in `configs/config.yaml` and `src/data_loader.py`.

**Expected data might include:**
- Candidate profiles (resume text, skills, experience, education, etc.)
- Job descriptions (title, requirements, responsibilities, qualifications)
- Possibly: recruiter labels / shortlist decisions (for supervised training)

---

## 🗺️ TODO Roadmap

### Phase 1: Project Setup ✅
- [x] Create project skeleton
- [x] Set up module stubs
- [x] Write README
- [ ] Initialize git repository
- [ ] Push to GitHub

### Phase 2: Dataset Understanding
- [ ] Obtain and load dataset
- [ ] Update `config.yaml` with actual column names
- [ ] Run EDA notebook (`01_eda.ipynb`)
- [ ] Document data quality issues
- [ ] Create sample subset for quick iteration

### Phase 3: Baseline Semantic Ranker
- [ ] Implement text preprocessing pipeline
- [ ] Generate embeddings for JD and candidates
- [ ] Compute cosine similarity scores
- [ ] Create initial ranking based on semantic similarity alone
- [ ] Evaluate qualitatively (does the ranking make sense?)

### Phase 4: Feature-Based Hybrid Scorer
- [ ] Engineer structured features (skills overlap, experience match, etc.)
- [ ] Build heuristic scoring function
- [ ] Combine semantic + structured scores (weighted fusion)
- [ ] Tune weights manually or via simple grid search
- [ ] (Optional) Train supervised re-ranker if labels available

### Phase 5: Evaluation & Submission
- [ ] Finalize ranking pipeline
- [ ] Generate submission file
- [ ] Run inference notebook end-to-end
- [ ] Package results in `outputs/submissions/`
- [ ] Final README polish & submission

---

## 📝 License

This project was built for a hackathon. See specific hackathon rules for usage terms.

---

*Built with ❤️ for better hiring.*
