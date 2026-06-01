from .embeddings import ai_pattern

weights = {
    "semantic": 0.35,
    "role": 0.10,
    "product": 0.10,
    "exp_band": 0.10,
    "ml_years": 0.10,
    "ai_depth": 0.05,
    "avail": 0.05,
    "rel": 0.05,
    "github": 0.05,
    "geo": 0.03,
    "work_mode": 0.02
}

alpha = 0.5
beta = 1.0

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

def generate_reasoning(row, cand_raw=None):
    title = row.get("current_title", "Professional")
    if cand_raw:
        title = cand_raw.get("profile", {}).get("current_title", title)
        
    yoe = row.get("total_years_experience", 0)
    ai_count = row.get("ai_core_skills_count", 0)
    sim = row.get("semantic_similarity", 0)
    
    named_skills = []
    signals = {}
    if cand_raw:
        skills = cand_raw.get("skills", [])
        signals = cand_raw.get("redrob_signals", {})
        skill_names = [s.get("name", str(s)) if isinstance(s, dict) else str(s) for s in skills]
        named_skills = [s for s in skill_names if ai_pattern.search(s)][:3]
        
    if sim > 0.6: alignment = "exceptional alignment with JD requirements for semantic ranking"
    elif sim > 0.4: alignment = "strong background in machine learning and embeddings"
    else: alignment = "moderate exposure to AI tooling"
        
    try: resp_rate = float(signals.get("recruiter_response_rate", 0))
    except: resp_rate = 0.0
    last_act = signals.get("last_active_date", "")
    
    if resp_rate > 0.8: behavior = f"Highly responsive to recruiters ({int(resp_rate*100)}% rate)"
    elif last_act: behavior = f"Recently active on the platform ({last_act[:10]})"
    else: behavior = "Moderate platform engagement"
        
    sentence1 = f"An experienced {title} with {yoe} years of experience, demonstrating {alignment}."
    skills_str = f" Includes core AI competencies like {', '.join(named_skills)} among {ai_count} ML skills." if named_skills else f" Displays {ai_count} relevant AI skills."
    sentence2 = f" {behavior}."
    
    concern = ""
    if row.get("product_company_score", 0) < 0.3: concern = " However, background leans heavily towards IT services."
    elif row.get("notice_period_penalty", 0) > 0: concern = " Note: Long notice period."
    elif row.get("ml_years_estimate", 0) < 1.0 and ai_count > 5: concern = " Note: Many AI skills but limited explicit ML tenure."
        
    return sentence1 + skills_str + sentence2 + concern
