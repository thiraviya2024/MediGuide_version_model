# utils/validator.py
def is_medical_document(text: str):
    """
    Improved medical document validator.
    Returns (is_medical: bool, matched_keywords: list)
    """
    if not text or len(text.strip()) < 50:
        return False, []

    text_lower = text.lower()
    
    # Expanded comprehensive medical keywords
    medical_keywords = [
        "patient", "hospital", "laboratory", "lab report", "diagnosis", "prescription",
        "physician", "doctor", "dr.", "consultant", "discharge summary", "clinical notes",
        "hemoglobin", "haemoglobin", "rbc", "wbc", "platelet", "platelet count",
        "glucose", "fasting glucose", "hba1c", "cholesterol", "hdl", "ldl", "triglycerides",
        "creatinine", "urea", "bilirubin", "sodium", "potassium", "sgot", "sgpt", "alt", "ast",
        "thyroid", "tsh", "t3", "t4", "ecg", "eeg", "xray", "x-ray", "mri", "ct scan", "ultrasound",
        "blood pressure", "bp", "pulse", "temperature", "pathology", "radiology",
        "reference range", "normal range", "mg/dl", "g/dl", "mmol/l",
        "cbc", "complete blood count", "liver function test", "kidney function test",
        "lipid profile", "blood test", "urine test", "medical report", "hospital report"
    ]
    
    matched = [kw for kw in medical_keywords if kw in text_lower]
    
    # Medical validation logic
    has_medical_term = len(matched) >= 3
    has_patient_context = any(word in text_lower for word in ["patient", "name", "age", "sex", "gender", "uhid", "ip no", "op no"])
    
    is_medical = has_medical_term and has_patient_context
    
    return is_medical, matched[:10]  # Return top 10 matched keywords