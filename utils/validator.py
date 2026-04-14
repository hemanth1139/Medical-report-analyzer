def validate_schema(data, mode):
    """Checks if the required keys exist in the analysis result."""
    if not data or not isinstance(data, dict) or "error" in data:
        return False
    keys = ["report_type", "summary", "key_findings"] if mode == "lab" else \
           ["prescription_type", "summary", "medications"]
    return all(key in data for key in keys)
