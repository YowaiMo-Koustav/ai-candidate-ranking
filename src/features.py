import re
from datetime import datetime
from dateutil import parser
from .embeddings import ai_pattern

REF_DATE = datetime(2025, 2, 1)

# False-positive skills that should never count as AI-relevant
_FALSE_POSITIVE = re.compile(
    r"^(tailwind|bootstrap|figma|photoshop|illustrator|canva|"
    r"wordpress|wix|squarespace|seo|google ads|salesforce|hubspot|"
    r"quickbooks|autocad|solidworks|css|html|jquery|react|angular|vue)$",
    re.IGNORECASE,
)


def calc_role_title_score(title):
    """Score how relevant a candidate's title is to the ML/AI JD."""
    if not title:
        return 0.0
    t = title.lower()
    # Strong ML/AI/Search titles
    if any(k in t for k in ["machine learning", "ml engineer", "ai engineer",
                             "data scientist", "applied scientist"]):
        return 1.0
    if any(k in t for k in ["search", "relevance", "recommendation",
                             "ranking engineer", "nlp", "research"]):
        return 0.9
    # Adjacent technical titles
    if any(k in t for k in ["data engineer", "backend engineer",
                             "software engineer", "platform engineer"]):
        return 0.5
    # Management / non-technical
    if any(k in t for k in ["manager", "director", "vp", "consultant"]):
        return 0.3
    return 0.2


def calc_product_company_score(company, industry):
    """Penalise pure IT-services backgrounds; reward product companies."""
    services_keywords = ["infosys", "tcs", "wipro", "cognizant", "accenture",
                         "consulting", "services", "capgemini", "hcl", "tech mahindra"]
    if company and any(k in company.lower() for k in services_keywords):
        return 0.1
    if industry and "it services" in industry.lower():
        return 0.2
    # Known product companies (signal, not exhaustive)
    product_kw = ["google", "microsoft", "amazon", "meta", "apple", "netflix",
                  "uber", "airbnb", "stripe", "spotify", "linkedin"]
    if company and any(k in company.lower() for k in product_kw):
        return 1.0
    return 0.6  # Default: assume moderate product-company fit


def calc_experience_band_match(yoe):
    """Peak at 6-8 YOE (JD sweet spot), taper outside."""
    try:
        y = float(yoe)
    except (TypeError, ValueError):
        return 0.0
    if 6 <= y <= 8:
        return 1.0
    if 4 <= y < 6 or 8 < y <= 10:
        return 0.8
    if 10 < y <= 12:
        return 0.6
    if 2 <= y < 4:
        return 0.4
    return 0.2


def calc_ml_years_estimate(history):
    """Estimate years spent in ML-titled or ML-described roles."""
    ml_years = 0.0
    for role in history:
        t = role.get("title", "").lower()
        d = role.get("description", "").lower()

        is_ml_role = False
        if any(k in t for k in ["machine learning", "ml", "ai", "data scientist",
                                 "search", "nlp", "recommendation"]):
            is_ml_role = True
        elif any(k in d for k in ["trained model", "pytorch", "tensorflow",
                                   "recommendation", "embedding", "ranking"]):
            is_ml_role = True

        if is_ml_role:
            try:
                start = parser.parse(role["start_date"])
                end = parser.parse(role["end_date"]) if role.get("end_date") else REF_DATE
                duration = (end.replace(tzinfo=None) - start.replace(tzinfo=None)).days / 365.25
                ml_years += max(0, duration)
            except (KeyError, ValueError, TypeError):
                pass
    return ml_years


def calc_ai_skill_depth(skills):
    """
    Score AI skill depth.  Returns a value clamped to [0, 1].
    Factors: count of AI-relevant skills and months of experience.
    """
    if not skills:
        return 0.0
    ai_count = 0
    total_months = 0
    for s in skills:
        name = s.get("name", str(s)) if isinstance(s, dict) else str(s)
        # Skip false positives
        if _FALSE_POSITIVE.match(name.strip()):
            continue
        if ai_pattern.search(name):
            ai_count += 1
            if isinstance(s, dict):
                total_months += s.get("months_experience", 0)

    # Normalise: up to 8 AI skills → 0.5, up to 10 yrs months → 0.5
    count_part = min(ai_count / 8.0, 1.0) * 0.5
    months_part = min(total_months / 120.0, 1.0) * 0.5
    return count_part + months_part


def calc_availability_score(signals):
    """Availability from recruiter response rate + recency.  Returns [0, 1]."""
    try:
        resp = float(signals.get("recruiter_response_rate", 0))
    except (TypeError, ValueError):
        resp = 0.0
    resp = min(resp, 1.0)  # Clamp to 1.0

    act_date = signals.get("last_active_date")
    if act_date:
        try:
            d = parser.parse(act_date)
            days_ago = (REF_DATE - d.replace(tzinfo=None)).days
            act_score = max(0.0, 1.0 - (days_ago / 180.0))  # 6 months decay
        except (ValueError, TypeError):
            act_score = 0.5
    else:
        act_score = 0.0

    open_flag = 1.0 if signals.get("open_to_work_flag") else 0.0

    return resp * 0.4 + act_score * 0.3 + open_flag * 0.3


def calc_reliability_score(signals):
    """Interview completion rate → [0, 1]."""
    try:
        ic = float(signals.get("interview_completion_rate", 0))
    except (TypeError, ValueError):
        ic = 0.0
    return min(ic, 1.0)


def calc_geo_fit(location):
    if not location:
        return 0.5
    loc = location.lower()
    if any(k in loc for k in ["san francisco", "bay area", "new york", "seattle", "remote"]):
        return 1.0
    if "usa" in loc or "united states" in loc:
        return 0.8
    if "india" in loc or "bangalore" in loc or "hyderabad" in loc:
        return 0.6
    return 0.3


def calc_notice_penalty(notice_days):
    try:
        days = float(notice_days)
    except (TypeError, ValueError):
        return 0.0
    if days <= 30:
        return 0.0
    if days <= 60:
        return 0.1
    return min(1.0, (days - 60) / 90.0)


def calc_honeypot_risk(yoe, hist_years, skills, ml_years):
    """Detect suspicious profiles with implausible claims."""
    risk = 0.0

    if yoe > 0 and hist_years > 0:
        if yoe > hist_years + 3:
            risk += 0.4
        if hist_years > yoe + 3:
            risk += 0.2

    skill_names = [s.get("name", str(s)) if isinstance(s, dict) else str(s) for s in skills]
    ai_count = sum(1 for s in skill_names if ai_pattern.search(s))

    if ai_count > 10 and ml_years < 1.0:
        risk += 0.5

    # Very short career but extreme AI skill count
    if yoe < 2 and ai_count > 6:
        risk += 0.3

    return min(1.0, risk)


def compute_all_features(cand):
    """
    Process a raw candidate dict and return an engineered features dict.
    All values are designed to be in [0, 1] range (or bounded).
    """
    cid = cand.get("candidate_id")
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})

    role_score = calc_role_title_score(profile.get("current_title"))
    company_score = calc_product_company_score(
        profile.get("current_company"), profile.get("current_industry")
    )

    yoe = profile.get("years_of_experience", 0)
    try:
        yoe_float = float(yoe) if yoe else 0.0
    except (TypeError, ValueError):
        yoe_float = 0.0

    exp_band_match = calc_experience_band_match(yoe)
    ml_years = calc_ml_years_estimate(history)

    skill_names = [s.get("name", str(s)) if isinstance(s, dict) else str(s) for s in skills]
    ai_skills_count = sum(
        1 for s in skill_names
        if ai_pattern.search(s) and not _FALSE_POSITIVE.match(s.strip())
    )
    ai_depth_score = calc_ai_skill_depth(skills)

    avail_score = calc_availability_score(signals)
    rel_score = calc_reliability_score(signals)

    # Normalise GitHub score: raw is 0-100
    gh_raw = max(0, float(signals.get("github_activity_score", 0) or 0))
    gh_score = min(gh_raw / 100.0, 1.0)

    geo_score = calc_geo_fit(profile.get("location"))
    pref_work = (profile.get("preferred_work_mode") or "").lower()
    wm_fit = 1.0 if any(k in pref_work for k in ["hybrid", "flexible", "remote"]) else 0.5
    notice_penalty = calc_notice_penalty(profile.get("notice_period_days"))

    hist_years = 0
    if history:
        try:
            earliest = min(
                parser.parse(r["start_date"])
                for r in history if "start_date" in r
            )
            hist_years = (REF_DATE - earliest.replace(tzinfo=None)).days / 365.25
        except (ValueError, TypeError):
            pass
    honeypot_risk = calc_honeypot_risk(yoe_float, hist_years, skills, ml_years)

    return {
        "candidate_id": cid,
        "role_title_score": role_score,
        "product_company_score": company_score,
        "total_years_experience": yoe_float,
        "experience_band_match": exp_band_match,
        "ml_years_estimate": ml_years,
        "ai_core_skills_count": ai_skills_count,
        "ai_skill_depth_score": ai_depth_score,
        "availability_score": avail_score,
        "reliability_score": rel_score,
        "github_fit_score": gh_score,    # Now 0-1 not 0-100
        "geo_fit_score": geo_score,
        "work_mode_fit": wm_fit,
        "notice_period_penalty": notice_penalty,
        "honeypot_risk_score": honeypot_risk,
    }
