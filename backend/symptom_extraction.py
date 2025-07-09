"""Module for extracting and clarifying symptoms from user input."""
import re

symptom_phrases = {
    # Digestive symptoms
    "can't eat": "loss of appetite",
    "don't feel like eating": "loss of appetite",
    "no appetite": "loss of appetite",
    "not hungry": "loss of appetite",
    "reduced appetite": "loss of appetite",
    "stomach hurts": "abdominal pain",
    "tummy hurts": "abdominal pain",
    "belly hurts": "abdominal pain",
    "stomach ache": "abdominal pain",
    "abdominal discomfort": "abdominal pain",
    "feel sick": "nausea",
    "feel like throwing up": "nausea",
    "want to vomit": "nausea",
    "nauseous": "nausea",
    "sick to stomach": "nausea",
    "bloated": "bloating",
    "stomach is swollen": "bloating",
    "abdominal swelling": "bloating",
    "burping a lot": "burping",
    "keep burping": "burping",
    "belching": "burping",
    "sour taste": "sour belching",
    "heartburn": "heartburn",
    "chest burning": "chest burning",
    "acid reflux": "heartburn",
    # Respiratory symptoms
    "shortness of breath": "shortness of breath",
    "can't breathe": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "breathing problems": "shortness of breath",
    "wheezing": "wheezing",
    "chest tightness": "chest tightness",
    "chest pain": "chest pain",
    "cough": "cough",
    "persistent cough": "persistent cough",
    "dry cough": "dry cough",
    "productive cough": "productive cough",
    "throat irritation": "throat irritation",
    "hoarseness": "hoarseness",
    "post-nasal drip": "post-nasal drip",
    # Fatigue and weakness
    "feel tired": "fatigue",
    "no energy": "fatigue",
    "always tired": "fatigue",
    "exhausted": "fatigue",
    "weak": "weakness",
    "feeling weak": "weakness",
    "lack of energy": "fatigue",
    "lethargy": "fatigue",
    # Liver and spleen symptoms
    "enlarged liver": "enlarged liver",
    "liver problems": "enlarged liver",
    "enlarged spleen": "enlarged spleen",
    "spleen problems": "enlarged spleen",
    "abdominal fullness": "abdominal fullness",
    "weight loss": "weight loss",
    "losing weight": "weight loss",
    "jaundice": "jaundice",
    "yellow skin": "jaundice",
    "yellow eyes": "jaundice",
    "ascites": "ascites",
    "fluid retention": "fluid retention",
    "abdominal swelling": "abdominal swelling",
    # Metabolic symptoms
    "high cholesterol": "high cholesterol",
    "weight gain": "weight gain",
    "gaining weight": "weight gain",
    "breathlessness": "breathlessness",
    "excessive sweating": "excessive sweating",
    "snoring": "snoring",
    # Anemia symptoms
    "pallor": "pallor",
    "pale skin": "pallor",
    "dizziness": "dizziness",
    "cold extremities": "cold extremities",
    "brittle nails": "brittle nails",
    "hair loss": "hair loss",
    # Skin symptoms
    "skin lesions": "skin lesions",
    "discoloration": "discoloration",
    "itching": "itching",
    "scaling": "scaling",
    "dry skin": "dry skin",
    "pigmentation": "pigmentation",
    "skin inflammation": "skin inflammation",
    "white patches": "white patches on skin",
    "loss of pigmentation": "loss of pigmentation",
    # Piles symptoms
    "anal pain": "anal pain",
    "bleeding": "bleeding",
    "rectal bleeding": "bleeding",
    "swelling": "swelling",
    "prolapse": "prolapse",
    "constipation": "constipation",
    # Diabetes symptoms
    "excessive urination": "excessive urination",
    "frequent urination": "excessive urination",
    "excessive thirst": "excessive thirst",
    "always thirsty": "excessive thirst",
    "excessive hunger": "excessive hunger",
    "blurred vision": "blurred vision",
    "slow healing wounds": "slow healing wounds",
    # Parasitic symptoms
    "anal itching": "anal itching",
    "intestinal worms": "parasitic infections",
    # Food poisoning symptoms
    "fever": "fever",
    "dehydration": "dehydration",
    # Gallbladder symptoms
    "RUQ pain": "RUQ pain",
    "right upper quadrant pain": "RUQ pain",
    "back pain": "back pain",
    "shoulder pain": "shoulder pain",
    "indigestion": "indigestion",
    # Tuberculosis symptoms
    "blood in sputum": "blood in sputum",
    "night sweats": "night sweats",
    # Arthritis symptoms
    "joint pain": "joint pain",
    "stiffness": "stiffness",
    "morning stiffness": "morning stiffness",
    "reduced mobility": "reduced mobility",
    "muscle weakness": "muscle weakness",
    # Tumor symptoms
    "lumps": "lumps",
    "unusual bleeding": "unusual bleeding",
    # General symptoms
    "anxiety": "anxiety",
    "discomfort": "general discomfort",
    "feeling bad": "general discomfort",
    "not well": "general discomfort",
    "sick": "general discomfort",
    "unwell": "general discomfort"
}

skin_rash_variants = {
    "skin rash": "skin lesions",
    "skin rashes": "skin lesions",
    "rash": "skin lesions",
    "rashes": "skin lesions",
    "itchy rash": "itching",
    "red spots": "skin lesions",
    "red rash": "skin lesions",
    "rash on legs": "skin lesions",
    "rash on arms": "skin lesions",
    "itchy skin": "itching",
    "itching on skin": "itching",
    "itching on legs": "itching",
    "itching on arms": "itching",
    "spots on skin": "skin lesions",
    "spots on legs": "skin lesions",
    "spots on arms": "skin lesions",
    "skin irritation": "skin inflammation",
    "skin redness": "discoloration",
    "redness on skin": "discoloration",
    "skin bumps": "skin lesions",
    "bumps on skin": "skin lesions",
    "hives": "skin lesions",
    "urticaria": "skin lesions"
}
symptom_phrases.update(skin_rash_variants)

def extract_canonical_symptoms(user_input, synonym_map):
    text = user_input.lower()
    tokens = re.split(r'[,.!?;]| and | or |/|\\|\n', text)
    found = set()
    for token in tokens:
        token = token.strip()
        for phrase, canonical in synonym_map.items():
            if phrase in token:
                found.add(canonical)
        for phrase, canonical in symptom_phrases.items():
            if phrase in token:
                found.add(canonical)
    return list(found)

def clarify_symptom(user_input, all_symptoms):
    return f"Could you clarify your symptom? For example: {', '.join(all_symptoms[:10])}."

def get_symptom_categories():
    return {
        "digestive": ["bloating", "gas", "loss of appetite", "burping", "abdominal pain", "nausea", "heartburn", "sour belching", "chest burning", "indigestion"],
        "respiratory": ["shortness of breath", "wheezing", "chest tightness", "cough", "difficulty breathing", "rapid breathing", "chest pain", "throat irritation", "hoarseness", "post-nasal drip"],
        "hepatic": ["enlarged liver", "enlarged spleen", "abdominal fullness", "jaundice", "ascites", "fluid retention"],
        "metabolic": ["weight gain", "high cholesterol", "breathlessness", "excessive sweating", "lethargy", "snoring"],
        "hematological": ["pallor", "fatigue", "weakness", "shortness of breath", "dizziness", "cold extremities", "brittle nails", "hair loss"],
        "dermatological": ["skin lesions", "discoloration", "itching", "scaling", "dry skin", "pigmentation", "skin inflammation", "white patches on skin", "loss of pigmentation"],
        "gastrointestinal": ["anal pain", "bleeding", "itching", "swelling", "prolapse", "discomfort", "constipation"],
        "endocrine": ["excessive urination", "excessive thirst", "excessive hunger", "fatigue", "weight loss", "blurred vision", "slow healing wounds"],
        "infectious": ["fever", "dehydration", "blood in sputum", "night sweats", "anal itching"],
        "musculoskeletal": ["joint pain", "stiffness", "swelling", "reduced mobility", "morning stiffness", "fatigue", "muscle weakness"],
        "oncological": ["lumps", "swelling", "pain", "weight loss", "fatigue", "loss of appetite", "unusual bleeding"]
    }

synonym_map = {}
for phrase, canonical in symptom_phrases.items():
    synonym_map[phrase] = canonical

SYMPTOM_SYNONYM_MAP = synonym_map
SYMPTOM_PHRASES = symptom_phrases