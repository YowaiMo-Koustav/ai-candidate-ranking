from .embeddings import ai_pattern

weights = {
    "semantic": 0.30,   # Semantic similarity (cosine)
    "role": 0.10,       # Role title fit
    "product": 0.08,    # Product company affinity
    "exp_band": 0.10,   # Experience band match
    "ml_years": 0.10,   # ML years estimate
    "ai_depth": 0.05,   # AI skill depth
    "avail": 0.04,      # Availability signals
    "rel": 0.04,        # Reliability signals
    "github": 0.03,     # GitHub activity
    "geo": 0.02,        # Geo fit
    "work_mode": 0.02,  # Work mode preference
}
# Total positive weights = 0.88 (deliberately < 0.9 to prevent saturation)

alpha = 0.15  # Notice period penalty scale
beta = 0.30   # Honeypot penalty scale

def compute_final_score(row, max_ml_years=1.0):
    if max_ml_years == 0: max_ml_years = 1.0
    
    score = (
        row.get("semantic_similarity", 0) * weights["semantic"] +
        row.get("role_title_score", 0) * weights["role"] +
        row.get("product_company_score", 0) * weights["product"] +
        row.get("experience_band_match", 0) * weights["exp_band"] +
        (row.get("ml_years_estimate", 0) / max_ml_years) * weights["ml_years"] +
        min(row.get("ai_skill_depth_score", 0), 1.0) * weights["ai_depth"] +
        row.get("availability_score", 0) * weights["avail"] +
        row.get("reliability_score", 0) * weights["rel"] +
        row.get("github_fit_score", 0) * weights["github"] +
        row.get("geo_fit_score", 0) * weights["geo"] +
        row.get("work_mode_fit", 0) * weights["work_mode"]
    )
    score -= (row.get("notice_period_penalty", 0) * alpha)
    score -= (row.get("honeypot_risk_score", 0) * beta)
    
    return max(0.0, min(1.0, score))

import re as _re

_JD_RETRIEVAL_KW = _re.compile(
    r"search|retrieval|ranking|recommendation|information retrieval|"
    r"relevance|re-?rank|query|candidate.?match",
    _re.IGNORECASE,
)
_JD_EMBEDDING_KW = _re.compile(
    r"embedding|vector|faiss|milvus|pgvector|annoy|"
    r"sentence.?transform|cosine|semantic.?similar",
    _re.IGNORECASE,
)
_JD_ML_KW = _re.compile(
    r"machine learning|deep learning|nlp|pytorch|tensorflow|"
    r"mlflow|llm|ai|neural|transformer|fine.?tun",
    _re.IGNORECASE,
)
_JD_EVAL_KW = _re.compile(
    r"ndcg|map@|precision@|recall@|eval|evaluation|a/b test",
    _re.IGNORECASE,
)
_JD_INFRA_KW = _re.compile(
    r"mlops|deploy|deployment|pipeline|docker|kubernetes|ci/cd|"
    r"airflow|production|serving|latency",
    _re.IGNORECASE,
)

def generate_reasoning(row, cand_raw=None):
    # row: engineered features + scores
    # cand_raw: optional raw candidate dict with 'profile', 'skills',
    #           'career_history', 'redrob_signals'

    cid = row.get("candidate_id")
    yoe = float(row.get("total_years_experience", 0) or 0)
    ai_count = int(row.get("ai_core_skills_count", 0) or 0)
    sim = float(row.get("semantic_similarity", 0) or 0)
    ml_years = float(row.get("ml_years_estimate", 0) or 0)
    prod_score = float(row.get("product_company_score", 0) or 0)
    honeypot = float(row.get("honeypot_risk_score", 0) or 0)

    cand = cand_raw or {}
    profile = cand.get("profile", {})
    title = profile.get("current_title", row.get("current_title", "Professional"))
    industry = profile.get("current_industry", "")
    signals = cand.get("redrob_signals", {})
    skills = cand.get("skills", [])

    skill_names = [
        s.get("name", str(s)) if isinstance(s, dict) else str(s)
        for s in skills
    ]
    retrieval_skills = [s for s in skill_names if _JD_RETRIEVAL_KW.search(s)][:2]
    embedding_skills = [s for s in skill_names if _JD_EMBEDDING_KW.search(s)][:2]
    ml_skills = [s for s in skill_names if _JD_ML_KW.search(s)][:3]
    eval_skills = [s for s in skill_names if _JD_EVAL_KW.search(s)][:2]
    infra_skills = [s for s in skill_names if _JD_INFRA_KW.search(s)][:2]

    try:
        resp_rate = float(signals.get("recruiter_response_rate", 0) or 0)
    except Exception:
        resp_rate = 0.0
    try:
        interview_rate = float(signals.get("interview_completion_rate", 0) or 0)
    except Exception:
        interview_rate = 0.0
    last_active = str(signals.get("last_active_date", "") or "")
    open_to_work = bool(signals.get("open_to_work_flag", False))
    try:
        notice_days = int(signals.get("notice_period_days", 0) or 0)
    except Exception:
        notice_days = 0

    is_services = prod_score < 0.3
    is_product = prod_score >= 0.6

    parts = []

    # Experience-tier opening
    if yoe >= 7 and sim > 0.45:
        if is_product:
            parts.append(
                f"Seasoned {title} with {yoe:.1f} years of experience and product-company depth"
            )
        else:
            parts.append(f"Senior {title} with {yoe:.1f} years of experience")
        if ml_years >= 2:
            parts[-1] += f", including {ml_years:.1f} years in ML-focused roles."
        else:
            parts[-1] += " and a strong technical foundation."
    elif yoe >= 3:
        parts.append(
            f"Mid-career {title} with {yoe:.1f} years of experience, showing a clear growth trajectory"
        )
        if ml_years > 0:
            parts[-1] += f" and {ml_years:.1f} years of hands-on ML work."
        else:
            parts[-1] += " in software/data engineering."
    else:
        parts.append(f"Early-career {title} with {yoe:.1f} years of experience")
        if ai_count > 0:
            parts[-1] += f", already building competence across {ai_count} AI-relevant skills."
        else:
            parts[-1] += ", with potential to grow into the role."

    # JD alignment
    jd_links = []
    if retrieval_skills:
        jd_links.append(f"search/retrieval expertise ({', '.join(retrieval_skills)})")
    if embedding_skills:
        jd_links.append(f"embedding/vector experience ({', '.join(embedding_skills)})")
    if not jd_links and ml_skills:
        jd_links.append(f"ML competencies ({', '.join(ml_skills[:2])})")
    if eval_skills:
        jd_links.append(f"evaluation practice ({', '.join(eval_skills)})")
    if infra_skills:
        jd_links.append(f"ML infrastructure ({', '.join(infra_skills[:2])})")

    if jd_links:
        parts.append(
            "Directly relevant to the JD through "
            + " and ".join(jd_links[:2])
            + "."
        )
    elif ai_count > 0 and ml_skills:
        parts.append(
            f"Brings {ai_count} AI-relevant skills including {', '.join(ml_skills[:2])}, "
            "though search/ranking signals are lighter."
        )
    elif ai_count > 0:
        parts.append(
            f"Lists {ai_count} AI-adjacent skills, but limited explicit overlap "
            "with the JD’s search/ranking/embedding focus."
        )
    else:
        domain = "IT services" if is_services else (industry.lower() or "a non-ML domain")
        parts.append(
            f"Profile leans toward {domain} with limited direct ML/retrieval signals, "
            "but adjacent technical experience."
        )

    # Behavioral signals
    behaviors = []
    if resp_rate >= 0.7:
        behaviors.append(f"highly responsive to recruiters ({int(resp_rate*100)}% reply rate)")
    if interview_rate >= 0.8:
        behaviors.append(
            f"strong interview follow-through ({int(interview_rate*100)}% completion)"
        )
    if open_to_work:
        behaviors.append("actively open to new roles")
    if last_active and last_active >= "2026-04":
        behaviors.append(f"recently active ({last_active[:10]})")

    if behaviors:
        parts.append("Platform signals: " + ", ".join(behaviors) + ".")
    elif resp_rate > 0 or last_active:
        parts.append(f"Moderate platform engagement (reply rate {int(resp_rate*100)}%).")
    else:
        parts.append("Limited platform activity data available.")

    # Concerns
    concerns = []
    if is_services and not is_product:
        concerns.append("background is primarily in IT services rather than product companies")
    if notice_days > 60:
        concerns.append(f"extended notice period ({notice_days} days)")
    if ml_years < 0.5 and ai_count > 3:
        concerns.append(
            f"lists {ai_count} AI skills but only {ml_years:.1f} years of verified ML tenure"
        )
    if honeypot > 0.2:
        concerns.append("some profile inconsistencies flagged")

    if concerns:
        parts.append("Considerations: " + "; ".join(concerns) + ".")

    return " ".join(parts)
