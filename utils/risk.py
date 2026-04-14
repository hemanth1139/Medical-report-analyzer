from config import HIGH_RISK_KEYWORDS, MODERATE_RISK_KEYWORDS, NORMAL_KEYWORDS

def detect_risk(data):
    """Scans analysis text for risk keywords and returns level and matches."""
    blob = str(data).lower()
    for level, words in [("HIGH", HIGH_RISK_KEYWORDS), ("MODERATE", MODERATE_RISK_KEYWORDS), ("NORMAL", NORMAL_KEYWORDS)]:
        hits = [w for w in words if w.lower() in blob]
        if hits:
            return level, hits
    return "UNKNOWN", []
