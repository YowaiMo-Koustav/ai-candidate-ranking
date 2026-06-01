import json
from docx import Document

def load_candidates(path, limit=None):
    """
    Generator to yield candidate dicts line by line from JSONL.
    """
    count = 0
    
    # Handle sample fallback array if needed
    is_json_array = False
    try:
        with open(path, 'r', encoding="utf-8") as f:
            first_char = f.read(1)
        is_json_array = (first_char == '[')
    except Exception:
        pass

    if is_json_array:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for cand in data:
                yield cand
                count += 1
                if limit and count >= limit:
                    break
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                count += 1
                if limit and count >= limit:
                    break

def build_jd_text(docx_path):
    """
    Loads the JD and extracts critical sections to form a concise query string.
    """
    try:
        doc = Document(docx_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full_text = "\n".join(paragraphs)
        return full_text
    except Exception as e:
        print(f"Error reading JD: {e}")
        return ""
