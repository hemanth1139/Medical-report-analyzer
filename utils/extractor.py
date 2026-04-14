def extract_content(uploaded_file):
    """Simply returns the raw bytes of the uploaded file. 
    Gemini 1.5 handles PDF and Images natively as binary data."""
    return uploaded_file.read()
