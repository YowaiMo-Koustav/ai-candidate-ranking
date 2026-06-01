import re
import numpy as np

ai_keywords = [
    "machine learning", "deep learning", "nlp", "search", "retrieval", 
    "embedding", "vector", "pytorch", "tensorflow", "mlflow", "llm", "ai",
    "artificial intelligence", "recommender", "recommendation"
]
ai_pattern = re.compile("|".join(ai_keywords), re.IGNORECASE)

def build_candidate_text(candidate):
    """
    Returns a single string representation of the candidate for semantic search.
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    
    parts = []
    
    # Profile text
    headline = profile.get("headline")
    if headline: parts.append(f"Headline: {headline}")
    
    summary = profile.get("summary")
    if summary: parts.append(f"Summary: {summary}")
        
    # Last 2 roles
    for role in history[:2]:
        title = role.get("title")
        desc = role.get("description")
        if title or desc:
            role_str = f"Role: {title or ''} - {desc or ''}"
            parts.append(role_str)
            
    # Skills (up to 20, AI prioritized)
    skill_names = [s.get("name", str(s)) if isinstance(s, dict) else str(s) for s in skills]
    
    ai_skills = [s for s in skill_names if ai_pattern.search(s)]
    other_skills = [s for s in skill_names if not ai_pattern.search(s)]
    
    top_skills = (ai_skills + other_skills)[:20]
    if top_skills:
        parts.append("Skills: " + ", ".join(top_skills))
        
    return " | ".join(parts)

def load_encoder_model():
    """
    Loads and returns the sentence transformer model. 
    This should be cached in Streamlit to avoid reloading.
    """
    from sentence_transformers import SentenceTransformer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return SentenceTransformer(model_name)
