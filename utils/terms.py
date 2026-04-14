from config import MEDICAL_TERMS

def find_terms(text):
    """Returns a dictionary of found medical terms and their explanations."""
    if not text: return {}
    return {term: expl for term, expl in MEDICAL_TERMS.items() if term.lower() in text.lower()}
