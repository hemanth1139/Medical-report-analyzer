# config.py — Central Knowledge & Prompts for AI Medical Report Analyzer

# ── Model & Retry Settings ─────────────────────────────────────────────────
GEMINI_MODEL    = "gemini-2.5-flash"
MAX_RETRIES     = 2

# ── Risk Detection Keywords ────────────────────────────────────────────────
HIGH_RISK_KEYWORDS = [
    "high risk", "critical", "severely elevated", "dangerously low",
    "immediate attention", "urgent", "consult immediately", "emergency"
]
MODERATE_RISK_KEYWORDS = [
    "abnormal", "elevated", "borderline", "slightly high", "slightly low",
    "monitor closely", "follow up", "above normal", "below normal"
]
NORMAL_KEYWORDS = [
    "normal", "within range", "healthy", "optimal", "good"
]

# ── Risk Level Display Messages ────────────────────────────────────────────
RISK_MESSAGES = {
    "HIGH":     "🚨 High Risk — Please consult your doctor immediately.",
    "MODERATE": "⚠️ Some abnormal values detected — Follow up with your doctor.",
    "NORMAL":   "✅ Report looks normal — Values appear within healthy ranges.",
    "UNKNOWN":  "ℹ️ Risk level unclear — Please review findings with your doctor."
}

# ── Medical Terms Dictionary ───────────────────────────────────────────────
MEDICAL_TERMS = {
    "HbA1c":        "A 3-month average of blood sugar levels. High values suggest diabetes risk.",
    "LDL":          "Bad cholesterol. High LDL clogs arteries and raises heart disease risk.",
    "HDL":          "Good cholesterol. Higher HDL protects your heart.",
    "triglycerides":"Fats in your blood. High levels raise risk of heart disease.",
    "hemoglobin":   "Protein in red blood cells that carries oxygen. Low = anemia.",
    "creatinine":   "Waste product filtered by kidneys. High = possible kidney issue.",
    "TSH":          "Thyroid Stimulating Hormone. Controls thyroid function.",
    "platelets":    "Blood cells that help clotting. Low = bleeding risk.",
    "WBC":          "White Blood Cells. High = infection or inflammation.",
    "RBC":          "Red Blood Cells. Low = anemia.",
    "bilirubin":    "Yellow pigment from broken red blood cells. High = liver issue.",
    "ALT":          "Liver enzyme. High ALT = liver stress or damage.",
    "AST":          "Another liver enzyme. High AST = liver or muscle issue.",
    "eGFR":         "Estimated kidney filtration rate. Low = reduced kidney function.",
    "sodium":       "Electrolyte controlling fluid balance. Abnormal = dehydration.",
    "potassium":    "Electrolyte for heart and muscle function. Abnormal = dangerous.",
    "glucose":      "Blood sugar. High = diabetes risk. Low = hypoglycemia.",
    "urea":         "Kidney waste product. High = kidney dysfunction.",
    "albumin":      "Protein made by liver. Low = liver or nutrition problem.",
    "ESR":          "Inflammation marker. High = infection or autoimmune condition.",
    "INR":          "Measures blood clotting speed. High = bleeding risk.",
    "calcium":      "Mineral for bones and heart. Abnormal = parathyroid or kidney issue.",
    "vitamin D":    "Bone and immune health. Low = deficiency, common in India.",
    "vitamin B12":  "Nerve and blood health. Low = fatigue, numbness, anemia.",
    "uric acid":    "Waste from digestion. High = gout or kidney stone risk.",
}

# ── JSON Schema Validation Keys ────────────────────────────────────────────
SCHEMA_REQUIRED_KEYS = [
    "report_type", "summary", "key_findings", "risk_level",
    "risk_reasons", "recommendation", "lifestyle_tips", "medical_terms_used"
]
REQUIRED_FINDING_KEYS = [
    "test_name", "patient_value", "normal_range", "status", "plain_explanation"
]

# ── Chat Suggestion Buttons ────────────────────────────────────────────────
CHAT_SUGGESTIONS = [
    "Should I see a doctor urgently?"
]

# ── Lab Report Analysis Prompt ─────────────────────────────────────────────
LAB_PROMPT = """You are a professional Medical Report Analysis AI.
Analyze the provided lab report and return ONLY a valid JSON object.
NO markdown. NO code fences. NO text before or after the JSON.

Follow this EXACT schema:
{
  "report_type": "Specific type e.g. CBC, Lipid Panel, Thyroid Panel",
  "summary": "3-4 sentence plain English overview of overall health status",
  "key_findings": [
    {
      "test_name": "Name of the test",
      "patient_value": "Value with unit",
      "normal_range": "Expected range with unit",
      "status": "One of: Normal, Borderline, Abnormal, Critical",
      "plain_explanation": "One sentence anyone can understand"
    }
  ],
  "risk_level": "One of: Low, Moderate, High, Critical",
  "risk_reasons": [
    "Each string is one specific reason this risk level was assigned"
  ],
  "recommendation": "2-3 action steps the patient should take. No diagnosis.",
  "lifestyle_tips": [
    "Each string is one practical tip based on the findings"
  ],
  "medical_terms_used": [
    {
      "term": "Medical term found in this report",
      "simple_meaning": "Plain English explanation of that term"
    }
  ]
}

If you cannot read any medical lab values from this document, return exactly:
{"error": "Invalid document. Please upload a lab report."}

Report Content: """

# ── Prescription Analysis Prompt ──────────────────────────────────────────
PRESCRIPTION_PROMPT = """You are a Clinical Pharmacy Assistant AI.
Analyze this prescription and return ONLY a valid JSON object.
NO markdown. NO code fences. NO text before or after the JSON.

Follow this EXACT schema:
{
  "prescription_type": "e.g. Outpatient, Handwritten, Hospital Discharge",
  "summary": "2-4 sentences in plain English describing the full regimen",
  "urgency": "One of: Routine, Soon, Urgent",
  "medications": [
    {
      "name": "Medication name as written",
      "strength": "Dose with unit",
      "instructions": "Frequency, timing, and food instructions",
      "duration": "Length of course",
      "common_side_effects": "1-2 common side effects in plain English",
      "notes": "Warnings, refills, or specific text from the prescription"
    }
  ],
  "general_notes": "Clinic info, date, allergies, or other text on the prescription",
  "questions_for_doctor_or_pharmacist": [
    "Important question the patient should ask"
  ]
}

If the document is not a prescription, return exactly:
{"error": "Invalid document. Please upload a prescription."}

Prescription Content: """

# ── Chat System Prompt ─────────────────────────────────────────────────────
CHAT_SYSTEM_PROMPT = """You are a helpful Medical Assistant AI.
You have read the patient's full medical document and its structured analysis.

Your rules:
- Answer based ONLY on the uploaded report — never give generic advice
- When user says "this value" or "that test", refer to the last discussed item
- If user asks "is this dangerous", use the detected risk level in your answer
- Explain all medical terms in simple plain English
- Never diagnose. Never prescribe. Explain and inform only.
- End every response with: "Please consult your doctor before making any medical decisions."
"""