from .embeddings import ai_pattern
import re as _re

# ==============================================================================
# Scoring Weights
# ==============================================================================
# Total positive weights = 0.88 (deliberately < 0.9 to prevent saturation)
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

alpha = 0.15  # Notice period penalty scale
beta = 0.30   # Honeypot penalty scale


def _clamp01(v):
    """Clamp a value to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(v)))


def compute_final_score(row, max_ml_years=1.0):
    """
    Compute the final candidate score as a weighted sum of normalised features.

    All raw feature values are clamped to [0, 1] before weighting so the
    pre-penalty sum can never exceed the total positive weight (0.88).
    This prevents score saturation at 1.0 and produces a healthy spread.
    """
    if max_ml_years == 0:
        max_ml_years = 1.0

    # --- Normalise every input to [0, 1] ---
    sem_sim = _clamp01(row.get("semantic_similarity", 0))
    role    = _clamp01(row.get("role_title_score", 0))          # may be negative in old data
    product = _clamp01(row.get("product_company_score", 0))
    exp_b   = _clamp01(row.get("experience_band_match", 0))
    ml_yr   = _clamp01(row.get("ml_years_estimate", 0) / max_ml_years)
    ai_dep  = _clamp01(row.get("ai_skill_depth_score", 0))     # was unbounded
    avail   = _clamp01(row.get("availability_score", 0))        # was unbounded (up to 11)
    rel     = _clamp01(row.get("reliability_score", 0))
    github  = _clamp01(row.get("github_fit_score", 0) / 100.0) # 0-100 → 0-1
    geo     = _clamp01(row.get("geo_fit_score", 0))
    wm      = _clamp01(row.get("work_mode_fit", 0))

    score = (
        sem_sim * weights["semantic"] +
        role    * weights["role"] +
        product * weights["product"] +
        exp_b   * weights["exp_band"] +
        ml_yr   * weights["ml_years"] +
        ai_dep  * weights["ai_depth"] +
        avail   * weights["avail"] +
        rel     * weights["rel"] +
        github  * weights["github"] +
        geo     * weights["geo"] +
        wm      * weights["work_mode"]
    )

    # Penalties
    notice_pen  = _clamp01(row.get("notice_period_penalty", 0))
    honeypot    = _clamp01(row.get("honeypot_risk_score", 0))
    score -= notice_pen * alpha
    score -= honeypot   * beta

    return max(0.0, min(1.0, score))


# ==============================================================================
# JD Keyword Patterns for Reasoning
# ==============================================================================

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
    r"mlflow|llm|neural|transformer|fine.?tun",
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

# Skills that should NEVER count as AI-relevant
_FALSE_POSITIVE_SKILLS = _re.compile(
    r"^(tailwind|bootstrap|figma|photoshop|illustrator|canva|"
    r"wordpress|wix|squarespace|seo|google ads|salesforce|hubspot|"
    r"quickbooks|autocad|solidworks)$",
    _re.IGNORECASE,
)


def _clean_title(title):
    """Remove duplicate adjectives like 'Senior Senior' from a title string."""
    if not title:
        return "Professional"
    # Collapse repeated words (case-insensitive)
    words = title.split()
    cleaned = [words[0]]
    for w in words[1:]:
        if w.lower() != cleaned[-1].lower():
            cleaned.append(w)
    return " ".join(cleaned)


def generate_reasoning(row, cand_raw=None):
    """
    Generate a concise, JD-aware, evidence-backed reasoning string.

    Rules:
    - No claim without supporting data.
    - No duplicate adjectives (e.g. "Senior Senior").
    - Concise: aim for 2-4 sentences.
    - Surfaces concerns when they materially affect ranking.
    """
    yoe = float(row.get("total_years_experience", 0) or 0)
    ai_count = int(row.get("ai_core_skills_count", 0) or 0)
    sim = float(row.get("semantic_similarity", 0) or 0)
    ml_years = float(row.get("ml_years_estimate", 0) or 0)
    prod_score = float(row.get("product_company_score", 0) or 0)
    honeypot = float(row.get("honeypot_risk_score", 0) or 0)

    cand = cand_raw or {}
    profile = cand.get("profile", {})
    raw_title = profile.get("current_title", row.get("current_title", "Professional"))
    title = _clean_title(raw_title)
    industry = profile.get("current_industry", "")
    signals = cand.get("redrob_signals", {})
    skills = cand.get("skills", [])

    # Filter false-positive skills
    skill_names = []
    for s in skills:
        name = s.get("name", str(s)) if isinstance(s, dict) else str(s)
        if not _FALSE_POSITIVE_SKILLS.match(name.strip()):
            skill_names.append(name)

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

    # ------- 1) Opening sentence — experience-tier -------
    if yoe >= 7 and sim > 0.40:
        opener = f"{title} with {yoe:.0f} years of experience"
        if is_product:
            opener += " in a product-company environment"
        if ml_years >= 2:
            opener += f", including {ml_years:.1f} years in ML-focused roles."
        elif ml_years > 0:
            opener += f" and {ml_years:.1f} years of ML exposure."
        else:
            opener += "."
    elif yoe >= 3:
        opener = f"{title} with {yoe:.0f} years of experience"
        if ml_years > 0:
            opener += f", {ml_years:.1f} of which in ML-related roles."
        else:
            opener += ", with a software/data engineering background."
    else:
        opener = f"Early-career {title} ({yoe:.0f} yrs)"
        if ai_count > 0:
            opener += f", developing competence across {ai_count} AI-relevant skills."
        else:
            opener += "."
    parts.append(opener)

    # ------- 2) JD alignment -------
    jd_links = []
    if retrieval_skills:
        jd_links.append(f"search/retrieval ({', '.join(retrieval_skills)})")
    if embedding_skills:
        jd_links.append(f"embeddings/vectors ({', '.join(embedding_skills)})")
    if not jd_links and ml_skills:
        jd_links.append(f"ML tools ({', '.join(ml_skills[:2])})")
    if eval_skills:
        jd_links.append(f"evaluation ({', '.join(eval_skills)})")
    if infra_skills:
        jd_links.append(f"MLOps ({', '.join(infra_skills[:2])})")

    if jd_links:
        parts.append(
            "Relevant to the JD through " + " and ".join(jd_links[:2]) + "."
        )
    elif ai_count > 0 and ml_skills:
        parts.append(
            f"Has {ai_count} AI-relevant skills including {', '.join(ml_skills[:2])}, "
            "though direct search/ranking experience is limited."
        )
    elif ai_count > 0:
        parts.append(
            f"Lists {ai_count} AI-adjacent skills but limited overlap "
            "with the JD's search/ranking/embedding focus."
        )
    else:
        domain = "IT services" if is_services else (industry.lower() or "general engineering")
        parts.append(
            f"Background leans toward {domain}; limited direct ML/retrieval signals."
        )

    # ------- 3) Platform signals (brief) -------
    platform_notes = []
    if resp_rate >= 0.7:
        platform_notes.append(f"{int(resp_rate*100)}% recruiter response rate")
    if interview_rate >= 0.8:
        platform_notes.append(f"{int(interview_rate*100)}% interview completion")
    if open_to_work:
        platform_notes.append("open to work")

    if platform_notes:
        parts.append("Platform: " + ", ".join(platform_notes) + ".")
    # Omit if no strong platform data — don't pad with filler

    # ------- 4) Concerns / considerations -------
    concerns = []
    if is_services and not is_product:
        concerns.append("primarily IT-services background")
    if notice_days > 60:
        concerns.append(f"{notice_days}-day notice period")
    if ml_years < 0.5 and ai_count > 3:
        concerns.append(
            f"{ai_count} AI skills listed but only {ml_years:.1f} yrs verified ML tenure"
        )
    if honeypot > 0.2:
        concerns.append("profile inconsistencies noted")

    if concerns:
        parts.append("Note: " + "; ".join(concerns) + ".")

    return " ".join(parts)
