import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import random
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import re
import requests
import string
import itertools
import difflib
from fastapi.responses import JSONResponse
from fastapi import status

from backend.intent_detection import detect_intent
from backend.symptom_extraction import extract_canonical_symptoms, SYMPTOM_SYNONYM_MAP, SYMPTOM_PHRASES, get_symptom_categories
from backend.symptom_dialogue import ask_for_more_symptoms, handle_symptom_clarification, handle_symptom_analysis
from backend.disease_matching import match_disease
from backend.session_manager import get_session, update_session, reset_session, hard_reset_session
from backend.llm_fallback import get_llm_response, filter_llm_output

# --- Fix: Robust data file path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'chatbot_data.json')

# --- Fix: API key error handling ---
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    # Set the provided OpenRouter API key as default
    OPENROUTER_API_KEY = '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'
    os.environ['OPENROUTER_API_KEY'] = OPENROUTER_API_KEY
    print("[INFO] Using default OpenRouter API key.")
else:
    os.environ['OPENROUTER_API_KEY'] = OPENROUTER_API_KEY

# --- Fix: Load data robustly ---
try:
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
except Exception as e:
    raise RuntimeError(f"[ERROR] Failed to load chatbot_data.json: {e}")

# Print the loaded dataset at startup
print("[DEBUG] Loaded dataset (first 2 diseases):", data['diseases'][:2])
if not data.get('diseases'):
    raise RuntimeError("[ERROR] No diseases found in dataset! Check your data/chatbot_data.json file.")

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings for all entries
faq_questions = [item['question'] for item in data['faq']]
faq_embeddings = embedder.encode(faq_questions)

small_talk_examples = []
small_talk_intents = []
small_talk_responses = []
for entry in data['small_talk']:
    for ex in entry['examples']:
        small_talk_examples.append(ex)
        small_talk_intents.append(entry['intent'])
        small_talk_responses.append(entry['responses'])
small_talk_embeddings = embedder.encode(small_talk_examples)

disease_symptoms = []
disease_indices = []
for idx, entry in enumerate(data['diseases']):
    for symptom in entry['symptoms']:
        disease_symptoms.append(symptom)
        disease_indices.append(idx)
disease_symptom_embeddings = embedder.encode(disease_symptoms)

# Session memory
sessions = {}

# Helper: get all unique symptoms
all_symptoms = sorted(set(symptom for entry in data['diseases'] for symptom in entry['symptoms']))

# --- Build master symptom index from dataset at startup ---
master_symptom_index = set()
symptom_to_diseases = {}
for disease in data['diseases']:
    for symptom in disease['symptoms']:
        s = symptom.lower().strip()
        master_symptom_index.add(s)
        if s not in symptom_to_diseases:
            symptom_to_diseases[s] = []
        symptom_to_diseases[s].append(disease)

# --- New robust extraction: n-gram matching ---
def extract_ngrams(text, n):
    tokens = text.split()
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def extract_all_ngrams(text, max_n=4):
    tokens = text.split()
    ngrams = set()
    for n in range(1, max_n+1):
        ngrams.update([' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    return ngrams

def robust_extract_symptoms(user_input):
    text = user_input.lower().replace(',', ' ').replace('.', ' ').replace(';', ' ').replace('!', ' ').replace('?', ' ')
    text = text.replace(' and ', ' ').replace(' or ', ' ')
    ngrams = extract_all_ngrams(text, max_n=4)
    found = set()
    for ng in ngrams:
        if ng in master_symptom_index:
            found.add(ng)
    return list(found)

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

app = FastAPI()

# Force CORS to allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler to always return JSON with session_id (if available)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Try to extract session_id from request if present
    try:
        body = await request.json()
        session_id = body.get('session_id', None)
    except Exception:
        session_id = None
    if not session_id:
        session_id = str(uuid.uuid4())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "session_id": session_id}
    )

def preprocess_input(user_input):
    # Lowercase for easier matching, but preserve original for replacements
    text = user_input.strip()
    # Replace 'it' or 'It' at the start
    text = re.sub(r'^(it\s+)', 'Tamra Bhasma ', text, flags=re.IGNORECASE)
    # Replace 'its' or 'Its' at the start
    text = re.sub(r'^(its\s+)', "Tamra Bhasma's ", text, flags=re.IGNORECASE)
    # Replace 'is it' or 'Is it' at the start
    text = re.sub(r'^(is it\s+)', 'is Tamra Bhasma ', text, flags=re.IGNORECASE)
    return text

# For fuzzy matching, create a list of all synonyms
all_synonym_phrases = list(SYMPTOM_SYNONYM_MAP.keys())

# For fuzzy matching FAQ and small talk
faq_questions_lower = [q['question'].lower() for q in data['faq']]
small_talk_examples_lower = [ex.lower() for ex in small_talk_examples]

# --- Intent-based small talk mapping ---
small_talk_intent_map = {}
for entry in data['small_talk']:
    for ex in entry['examples']:
        small_talk_intent_map[ex.lower()] = entry['intent']

intent_to_responses = {entry['intent']: entry['responses'] for entry in data['small_talk']}

# Normalize user input for intent matching

def get_small_talk_intent(user_input):
    text = user_input.lower().strip()
    for ex, intent in small_talk_intent_map.items():
        if ex == text:
            return intent
    return None

def get_small_talk_response(user_input, last_intent=None):
    """Get a response for small talk based on detected intent.
    Returns a tuple of (response, intent).
    """
    intent = get_small_talk_intent(user_input)
    if intent and intent != last_intent:
        responses = intent_to_responses.get(intent, [])
        if responses:
            return random.choice(responses), intent
    return None, None

# --- Semantic FAQ matching using embeddings ---
def get_faq_response_semantic(user_input, threshold=0.85):
    user_emb = embedder.encode([user_input])[0]
    sims = cosine_similarity([user_emb], faq_embeddings)[0]
    best_idx = int(np.argmax(sims))
    best_score = sims[best_idx]
    if best_score >= threshold:
        return data['faq'][best_idx]['answer']
    return None

# --- Interactive symptom collection ---
def extract_symptoms_from_input(user_input):
    # Normalize input: lowercase, remove punctuation
    text = user_input.lower().translate(str.maketrans('', '', string.punctuation))
    
    # Preprocess input to handle common phrases and variations
    text = text.replace("im feeling", "")
    text = text.replace("i am feeling", "")
    text = text.replace("i have", "")
    text = text.replace("ive got", "")
    text = text.replace("suffering from", "")
    text = text.replace("experiencing", "")
    
    found = set()
    tokens = set(text.split())
    for phrase in all_synonym_phrases:
        phrase_tokens = set(phrase.split())
        if phrase in text or text in phrase or (len(tokens & phrase_tokens) >= max(1, len(phrase_tokens)//2)):
            found.add(SYMPTOM_SYNONYM_MAP[phrase])
    
    # Check for common symptom phrases that might not be direct matches
    symptom_phrases = {
        "cant eat": "loss of appetite",
        "dont feel like eating": "loss of appetite",
        "stomach hurts": "abdominal pain",
        "belly hurts": "abdominal pain",
        "feel sick": "nausea",
        "feel tired": "fatigue",
        "no energy": "fatigue",
        "throwing up": "vomiting",
        "skin is yellow": "jaundice",
        "eyes are yellow": "jaundice",
        "stomach pain": "abdominal pain",
        "head hurts": "headache",
        "head is pounding": "headache"
    }
    
    for phrase, symptom in symptom_phrases.items():
        if phrase in text and symptom not in found:
            found.add(symptom)
    
    # Debug: print extracted symptoms
    print(f"[DEBUG] Extracted symptoms from '{user_input}': {found}")
    return list(found)

def get_disease_match(user_input, threshold=0.6):
    user_emb = embedder.encode([user_input])
    sims = cosine_similarity(user_emb, disease_symptom_embeddings)[0]
    agg_scores = {}
    for i, sim in enumerate(sims):
        idx = disease_indices[i]
        agg_scores.setdefault(idx, []).append(sim)
    avg_scores = {idx: np.mean(sims) for idx, sims in agg_scores.items()}
    best_idx = max(avg_scores, key=avg_scores.get)
    if avg_scores[best_idx] > threshold:
        disease = data['diseases'][best_idx]
        remedy = disease['remedy']
        return f"Possible match: {disease['disease']}\nSymptoms: {', '.join(disease['symptoms'])}\nRemedy: {remedy['description']}\nDosage: {remedy['dosage']}\nCautions: {remedy['cautions']}"
    return None

def get_rag_context(user_input, top_k=2):
    # Get top_k most relevant FAQ answers
    user_emb = embedder.encode([user_input])
    faq_sims = cosine_similarity(user_emb, faq_embeddings)[0]
    context = []
    for idx in np.argsort(faq_sims)[-top_k:][::-1]:
        context.append(f"FAQ: Q: {data['faq'][idx]['question']}\nA: {data['faq'][idx]['answer']}")
    # Add all disease names and remedies
    for entry in data['diseases']:
        symptoms = ', '.join(entry['symptoms'])
        context.append(f"Disease: {entry['disease']}\nSymptoms: {symptoms}\nRemedy: {entry['remedy']['description']}\nDosage: {entry['remedy']['dosage']}\nCautions: {entry['remedy']['cautions']}")
    return '\n\n'.join(context)

def generate_rag_context(user_input, dataset, top_k=3):
    """Generate RAG context for LLM based on user input and dataset.
    
    This function creates a context for the LLM by finding relevant information
    from the dataset based on the user's query. It includes:
    1. Most semantically similar FAQ entries
    2. Relevant disease information if query contains symptom-related terms
    3. General information about Tamra Bhasma for broader queries
    
    Args:
        user_input: The user's query text
        dataset: The full dataset containing FAQ, diseases, and other information
        top_k: Number of most relevant items to include
        
    Returns:
        A string containing the assembled context
    """
    # Preprocess the input
    processed_input = preprocess_input(user_input)
    
    # Initialize context sections
    context_parts = []
    
    # Get semantically similar FAQ entries
    user_emb = embedder.encode([processed_input])
    faq_sims = cosine_similarity(user_emb, faq_embeddings)[0]
    top_faq_indices = np.argsort(faq_sims)[-top_k:][::-1]
    
    # Add FAQ entries to context
    for idx in top_faq_indices:
        if faq_sims[idx] > 0.4:  # Only include if similarity is reasonable
            context_parts.append(f"FAQ: Q: {dataset['faq'][idx]['question']}\nA: {dataset['faq'][idx]['answer']}")
    
    # Check if query might be symptom-related
    symptom_terms = ['symptom', 'pain', 'ache', 'discomfort', 'feeling', 'sick', 'illness', 'disease', 'condition', 'health']
    is_symptom_query = any(term in processed_input.lower() for term in symptom_terms)
    
    # If symptom-related, add relevant disease information
    if is_symptom_query:
        # Extract potential symptoms from query
        potential_symptoms = extract_canonical_symptoms(processed_input, dataset)
        
        # Find diseases that match these symptoms
        relevant_diseases = []
        for disease in dataset['diseases']:
            disease_symptoms = disease.get('symptoms', [])
            # Check for symptom overlap
            overlap = [s for s in potential_symptoms if s in disease_symptoms]
            if overlap:
                relevant_diseases.append((disease, len(overlap)))
        
        # Sort by number of matching symptoms and take top matches
        relevant_diseases.sort(key=lambda x: x[1], reverse=True)
        for disease, _ in relevant_diseases[:2]:  # Include top 2 most relevant diseases
            symptoms = ', '.join(disease.get('symptoms', []))
            context_parts.append(f"Disease: {disease.get('name', '')}\nSymptoms: {symptoms}\nDescription: {disease.get('description', '')}\nRemedies: {disease.get('remedies', '')}")
    
    # Always include general Tamra Bhasma information for context
    tamra_info = next((item for item in dataset.get('faq', []) if 'tamra bhasma' in item.get('question', '').lower()), None)
    if tamra_info:
        context_parts.append(f"General Information: {tamra_info.get('answer', '')}")
    
    # Join all context parts
    return '\n\n'.join(context_parts)

def suggest_next_symptom(current_symptoms, matched_diseases, dataset):
    """Suggest the next most useful symptom to ask about, based on current symptoms and potential matches.
    
    This function analyzes the current symptoms and potential disease matches to determine
    which additional symptom would be most informative to ask about next.
    
    Args:
        current_symptoms: List of symptoms already identified
        matched_diseases: List of potential disease matches (can be empty)
        dataset: The full dataset containing disease information
        
    Returns:
        A string containing the suggested next symptom to ask about, or None if no good suggestion
    """
    # If we have matched diseases, focus on symptoms from those diseases
    if matched_diseases:
        # Get disease objects from names
        disease_objects = []
        for disease_name in matched_diseases:
            disease_obj = next((d for d in dataset['diseases'] if d.get('name') == disease_name), None)
            if disease_obj:
                disease_objects.append(disease_obj)
        
        # Find symptoms that appear in these diseases but aren't in current symptoms
        candidate_symptoms = {}
        for disease in disease_objects:
            for symptom in disease.get('symptoms', []):
                if symptom not in current_symptoms:
                    candidate_symptoms[symptom] = candidate_symptoms.get(symptom, 0) + 1
        
        # Return the most common symptom among candidate diseases
        if candidate_symptoms:
            # Sort by frequency (most common first)
            sorted_symptoms = sorted(candidate_symptoms.items(), key=lambda x: x[1], reverse=True)
            return sorted_symptoms[0][0]
    
    # If no matched diseases or no good candidates from them, suggest from all diseases
    all_symptoms = set()
    symptom_frequency = {}
    
    # Collect all symptoms and their frequencies across diseases
    for disease in dataset['diseases']:
        for symptom in disease.get('symptoms', []):
            if symptom not in current_symptoms:
                all_symptoms.add(symptom)
                symptom_frequency[symptom] = symptom_frequency.get(symptom, 0) + 1
    
    # Return the most common symptom overall
    if symptom_frequency:
        sorted_symptoms = sorted(symptom_frequency.items(), key=lambda x: x[1], reverse=True)
        return sorted_symptoms[0][0]
    
    return None

def match_disease_from_symptoms(symptom_list):
    # Return the best matching disease if symptoms overlap enough
    best_match = None
    best_count = 0
    for entry in data['diseases']:
        overlap = set(symptom_list) & set(entry['symptoms'])
        if len(overlap) > best_count and len(overlap) > 0:
            best_count = len(overlap)
            best_match = entry
    return best_match if best_count > 0 else None

def openrouter_fallback(user_input, history):
    prompt = (
        "You are TamraBot, an expert, friendly, and interactive assistant. Answer the user's latest question as accurately, concisely, and precisely as possible. If the answer is a list, provide a short, direct list. If the answer is a fact, state it directly. Avoid unnecessary details. If the question is outside your specialty, do your best to provide a helpful, accurate answer."
        f"\nConversation history (for reference only):\n"
    )
    for h in history:
        prompt += f"User: {h[0]}\nTamraBot: {h[1]}\n"
    prompt += f"User: {user_input}\nTamraBot:"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://yourdomain.com",
        "X-Title": "TamraBot"
    }
    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.4,
        "top_p": 0.95
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )
    return response.json()["choices"][0]["message"]["content"].strip()

# Expanded acknowledgement replies
acknowledgement_replies = [
    "Let me know if you have any more questions or symptoms!",
    "I'm here if you need anything else about Tamra Bhasma or Ayurveda.",
    "Feel free to ask if you have more questions.",
    "Happy to help! Ask me anything else if you wish.",
    "If you have more symptoms or queries, just let me know.",
    "Glad I could assist. Reach out if you need more info.",
    "Anytime! If you think of something else, I'm here.",
    "You're welcome! Let me know if you want to know more."
]

def is_acknowledgement(text):
    t = text.lower().strip()
    return any(t == phrase or t in phrase or phrase in t for phrase in acknowledgement_replies)

def is_out_of_scope(text):
    # Very basic: if not about tamra, ayurveda, health, or small talk, consider out of scope
    t = text.lower()
    keywords = ['tamra', 'bhasma', 'ayurveda', 'health', 'medicine', 'disease', 'symptom', 'remedy', 'treatment', 'doctor', 'dose', 'dosage', 'side effect', 'cure', 'safe', 'use', 'benefit', 'effect', 'indication', 'contraindication', 'how', 'what', 'why', 'when', 'where', 'can', 'should']
    return not any(k in t for k in keywords)

# Add a set of vague/general terms
VAGUE_TERMS = [
    'pain', 'discomfort', 'not feeling well', 'feeling bad', 'unwell', 'sick', 'ill', 'malaise', 'bad', 'weak', 'tired', 'ache', 'hurting', 'problem', 'issue', 'trouble', 'suffering', 'symptom', 'feeling weird', 'weird', 'off', 'strange', 'uncomfortable', 'uncomfortable feeling', 'general discomfort'
]

# Clarification function for vague/general symptoms
VAGUE_LOCATIONS = ['head', 'stomach', 'chest', 'back', 'joints', 'skin', 'abdomen', 'throat', 'legs', 'arms']
VAGUE_TYPES = ['sharp', 'dull', 'constant', 'intermittent', 'burning', 'throbbing', 'aching', 'pressure', 'tightness', 'swelling', 'itching', 'numbness']

def is_vague_symptom(user_input):
    text = user_input.lower()
    return any(term in text for term in VAGUE_TERMS)

def clarify_vague_symptom(user_input, all_symptoms):
    # Try to guess what the user means and ask for more details
    clarification = (
        "I noticed your symptom is a bit general. Could you clarify it?\n"
        "- Where do you feel it? (e.g., " + ', '.join(VAGUE_LOCATIONS) + ")\n"
        "- What type of feeling is it? (e.g., " + ', '.join(VAGUE_TYPES) + ")\n"
        "Or, do you have any of these specific symptoms: " + ', '.join(all_symptoms[:10]) + "?\n"
        "You can also describe your symptom in another way."
    )
    return clarification

# Mapping of (location, type) to canonical symptoms
LOCATION_TYPE_TO_SYMPTOM = {
    ('stomach', 'sharp'): 'abdominal pain',
    ('stomach', 'dull'): 'abdominal pain',
    ('stomach', 'ache'): 'abdominal pain',
    ('stomach', 'burning'): 'heartburn',
    ('chest', 'tightness'): 'chest tightness',
    ('chest', 'pain'): 'chest pain',
    ('chest', 'burning'): 'chest burning',
    ('joints', 'aching'): 'joint pain',
    ('joints', 'pain'): 'joint pain',
    ('back', 'pain'): 'back pain',
    ('head', 'pain'): 'headache',
    ('head', 'pounding'): 'headache',
    ('abdomen', 'pain'): 'abdominal pain',
    ('abdomen', 'sharp'): 'abdominal pain',
    ('skin', 'itching'): 'itching',
    ('skin', 'rash'): 'skin lesions',
    ('skin', 'scaling'): 'scaling',
    ('throat', 'pain'): 'throat irritation',
    ('throat', 'sore'): 'throat irritation',
    # Add more as needed
}

# --- Expanded and simplified synonym map for robust symptom matching ---
USER_SYNONYM_MAP = {
    # Stomach/abdominal pain
    'stomach pain': 'abdominal pain',
    'pain in stomach': 'abdominal pain',
    'my stomach hurts': 'abdominal pain',
    'stomach hurts': 'abdominal pain',
    'my stomach is hurting': 'abdominal pain',
    'hurting stomach': 'abdominal pain',
    'tummy ache': 'abdominal pain',
    'belly ache': 'abdominal pain',
    'belly pain': 'abdominal pain',
    'pain in belly': 'abdominal pain',
    'pain in abdomen': 'abdominal pain',
    'abdominal pain': 'abdominal pain',
    'cramp in stomach': 'abdominal pain',
    'cramping stomach': 'abdominal pain',
    'cramping in abdomen': 'abdominal pain',
    'crampy pain in stomach': 'abdominal pain',
    'crampy pain abdomen': 'abdominal pain',
    'sharp pain in stomach': 'abdominal pain',
    'sharp pain stomach': 'abdominal pain',
    'dull pain in stomach': 'abdominal pain',
    'ache in stomach': 'abdominal pain',
    'pain abdomen': 'abdominal pain',
    'sharp abdominal pain': 'abdominal pain',
    'dull abdominal pain': 'abdominal pain',
    'pain in tummy': 'abdominal pain',
    'tummy pain': 'abdominal pain',
    'sharp tummy pain': 'abdominal pain',
    'sharp pain in belly': 'abdominal pain',
    'sharp pain in abdomen': 'abdominal pain',
    # Add more mappings for other common symptoms as needed
}

# --- Simplified symptom extraction ---
def extract_user_symptoms(user_input):
    text = user_input.lower().strip()
    text = text.translate(str.maketrans('', '', '.,!?'))
    # Token-based match for phrases like 'my stomach hurts'
    found = set()
    tokens = [t.strip() for t in re.split(r',| and | or |/|;|\n|\.|\?|!', text) if t.strip()]
    for t in tokens:
        # Fuzzy match to canonical symptoms
        close = difflib.get_close_matches(t, CANONICAL_SYMPTOM_LIST, n=1, cutoff=0.7)
        if close:
            found.add(close[0])
        elif t in USER_SYNONYM_MAP:
            found.add(USER_SYNONYM_MAP[t])
    return list(found)

# --- Improved clarification logic ---
def needs_clarification(user_input, mapped_symptoms):
    # Only ask for clarification if nothing was mapped and input is very short or generic
    if mapped_symptoms:
        return False
    generic_words = {'pain', 'hurt', 'discomfort', 'feeling bad', 'not well', 'sick', 'unwell', 'problem', 'issue'}
    text = user_input.lower().strip()
    return any(word in text for word in generic_words) or len(text.split()) <= 2

# Build canonical symptom list and synonym map from dataset
CANONICAL_SYMPTOMS = set()
CANONICAL_SYMPTOM_LIST = []
for disease in data['diseases']:
    for symptom in disease['symptoms']:
        if symptom not in CANONICAL_SYMPTOMS:
            CANONICAL_SYMPTOMS.add(symptom)
            CANONICAL_SYMPTOM_LIST.append(symptom)

def map_user_symptom_to_canonical(user_symptom):
    s = user_symptom.lower().strip()
    # Direct match
    if s in CANONICAL_SYMPTOMS:
        return s
    # Synonym map
    if s in USER_SYNONYM_MAP:
        return USER_SYNONYM_MAP[s]
    # Fuzzy match
    close = difflib.get_close_matches(s, CANONICAL_SYMPTOM_LIST, n=1, cutoff=0.75)
    if close:
        return close[0]
    return None

def map_user_disease_to_canonical(user_disease):
    disease_names = [d['disease'].lower() for d in data['diseases']]
    close = difflib.get_close_matches(user_disease.lower(), disease_names, n=1, cutoff=0.7)
    if close:
        for d in data['diseases']:
            if d['disease'].lower() == close[0]:
                return d
    return None

def get_next_symptom_suggestions(current_symptoms, possible_diseases):
    """
    Suggests next possible symptoms to help narrow down the diagnosis.
    Returns a list of symptoms that are present in possible_diseases but not in current_symptoms.
    """
    suggestions = set()
    for disease in possible_diseases:
        for symptom in disease.get('symptoms', []):
            if symptom not in current_symptoms:
                suggestions.add(symptom)
    # Return a few suggestions (e.g., up to 5)
    return list(suggestions)[:5]

# Add a set of acknowledgment terms
ACKNOWLEDGMENT_TERMS = {"ok", "thanks", "thank you", "i see", "got it", "understood", "alright", "cool", "great", "good", "fine", "sure", "noted", "sounds good", "awesome", "nice"}

# Add a set of negative responses (denials)
NEGATIVE_RESPONSES = {"no", "none", "none of these", "nope", "not really", "nah"}

# Add a set of closure terms
CLOSURE_TERMS = {"that's all", "im fine now", "i'm fine now", "no more issues", "thank you that's it", "that's it", "i'm good", "im good", "i'm okay", "im okay", "i'm healthy", "im healthy", "all good", "done", "finished", "no more symptoms", "no more problems", "no more concerns"}

def get_active_thread(symptom_threads):
    for thread in symptom_threads:
        if thread["status"] == "active":
            return thread
    return None

def llm_generate_followup(user_input, current_symptoms, history):
    prompt = (
        f"The user said: '{user_input}'. Their current symptoms are: {', '.join(current_symptoms) if current_symptoms else 'None'}. "
        "What follow-up questions would help clarify their symptoms? Respond as a helpful, empathetic medical assistant. "
        "If the user described a headache, ask about the nature, location, and associated symptoms. "
        "If the user described a vague or complex symptom, ask clarifying questions to help narrow down the possible causes. "
        "If the user asks for help, respond with a supportive, guiding message."
    )
    context = "You are TamraBot, an intelligent Ayurveda assistant."
    return get_llm_response(prompt, context, data, OPENROUTER_API_KEY, history)

def llm_summarize_and_next_steps(history, current_symptoms):
    prompt = (
        f"So far, the user has described these symptoms: {', '.join(current_symptoms) if current_symptoms else 'None'}. "
        "Summarize the conversation and suggest the next best step. "
        "If symptoms span multiple systems, ask if they are related or separate. "
        "Always include a medical disclaimer at the end."
    )
    context = "You are TamraBot, an intelligent Ayurveda assistant."
    return get_llm_response(prompt, context, data, OPENROUTER_API_KEY, history)

def llm_acknowledgment_response(user_input, history):
    prompt = (
        f"The user said: '{user_input}'. Respond as a friendly, empathetic assistant, acknowledging their message and inviting further questions if appropriate."
    )
    context = "You are TamraBot, an intelligent Ayurveda assistant."
    return get_llm_response(prompt, context, data, OPENROUTER_API_KEY, history)

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    user_input = req.message.strip()
    session = get_session(session_id)
    history = session.get("history", [])
    # Multi-threaded symptom management
    symptom_threads = session.get("symptom_threads", [])
    if not symptom_threads:
        symptom_threads = [{"symptoms": [], "denied_symptoms": [], "status": "active"}]
    active_thread = get_active_thread(symptom_threads)
    if not active_thread:
        # All threads closed, start a new one
        active_thread = {"symptoms": [], "denied_symptoms": [], "status": "active"}
        symptom_threads.append(active_thread)
    discussed_symptoms = set(active_thread["symptoms"])
    denied_symptoms = set(active_thread["denied_symptoms"])
    clarification_attempts = session.get("clarification_attempts", 0)
    low_confidence_turns = session.get("low_confidence_turns", 0)
    last_asked_symptoms = session.get("last_asked_symptoms", [])
    max_clarification_attempts = 5
    max_low_confidence_turns = 5
    human_handoff_threshold = 5
    disclaimer_needed = False

    # --- Intent detection first ---
    intent = detect_intent(user_input, data)
    # 1. Handle acknowledgments
    if user_input.lower().strip() in ACKNOWLEDGMENT_TERMS:
        response = llm_acknowledgment_response(user_input, history)
        update_session(session_id, "history", history + [(user_input, response)])
        active_thread_index = symptom_threads.index(active_thread)
        active_thread_status = active_thread["status"]
        return {"response": response, "history": history + [(user_input, response)], "session_id": session_id, "active_thread_index": active_thread_index, "active_thread_status": active_thread_status}
    # 1b. Handle explicit closure
    if user_input.lower().strip() in CLOSURE_TERMS:
        active_thread["status"] = "closed"
        session["symptom_threads"] = symptom_threads
        response = llm_acknowledgment_response(user_input, history)
        update_session(session_id, "history", history + [(user_input, response)])
        active_thread_index = symptom_threads.index(active_thread)
        active_thread_status = active_thread["status"]
        return {"response": response, "history": history + [(user_input, response)], "session_id": session_id, "active_thread_index": active_thread_index, "active_thread_status": active_thread_status}

    # 2. Handle negative responses (denials)
    if user_input.lower().strip() in NEGATIVE_RESPONSES:
        denied_symptoms.update(last_asked_symptoms)
        active_thread["denied_symptoms"] = list(denied_symptoms)
        response = "Thank you for letting me know. I'll avoid asking about those symptoms again. If you have any other symptoms or concerns, please share them."
        update_session(session_id, "history", history + [(user_input, response)])
        # Find the index of the active thread
        active_thread_index = symptom_threads.index(active_thread)
        active_thread_status = active_thread["status"]
        return {"response": response, "history": history + [(user_input, response)], "session_id": session_id, "active_thread_index": active_thread_index, "active_thread_status": active_thread_status}

    if intent in ["small_talk", "greeting", "out_of_scope"]:
        context = "You are TamraBot, a friendly Ayurveda assistant. Respond naturally to small talk, greetings, or unrelated questions. Do not ask about symptoms unless the user brings up a health concern."
        llm_response = get_llm_response(user_input, context, data, OPENROUTER_API_KEY, history)
        response = llm_response
        update_session(session_id, "history", history + [(user_input, response)])
        # Find the index of the active thread
        active_thread_index = symptom_threads.index(active_thread)
        active_thread_status = active_thread["status"]
        return {"response": response, "history": history + [(user_input, response)], "session_id": session_id, "active_thread_index": active_thread_index, "active_thread_status": active_thread_status}
    elif intent == "faq":
        faq_answer = get_faq_response_semantic(user_input)
        if faq_answer:
            response = faq_answer
            update_session(session_id, "history", history + [(user_input, response)])
            # Find the index of the active thread
            active_thread_index = symptom_threads.index(active_thread)
            active_thread_status = active_thread["status"]
            return {"response": response, "history": history + [(user_input, response)], "session_id": session_id, "active_thread_index": active_thread_index, "active_thread_status": active_thread_status}
        context = generate_rag_context(user_input, data)
        llm_response = get_llm_response(user_input, context, data, OPENROUTER_API_KEY, history)
        response = filter_llm_output(llm_response, data)
        update_session(session_id, "history", history + [(user_input, response)])
        # Find the index of the active thread
        active_thread_index = symptom_threads.index(active_thread)
        active_thread_status = active_thread["status"]
        return {"response": response, "history": history + [(user_input, response)], "session_id": session_id, "active_thread_index": active_thread_index, "active_thread_status": active_thread_status}
    if intent == "symptom":
        user_symptoms, qualifiers = robust_extract_all_symptoms(user_input)
        user_symptoms = [map_user_symptom_to_canonical(s) for s in user_symptoms if map_user_symptom_to_canonical(s)]
        new_symptoms = set(user_symptoms) - discussed_symptoms
        discussed_symptoms.update(user_symptoms)
        active_thread["symptoms"] = list(discussed_symptoms)
        # Detect if new/unrelated symptom domain is introduced
        if user_symptoms:
            # Check if symptoms span multiple systems
            categories = get_symptom_categories()
            system_map = {}
            for cat, symlist in categories.items():
                for sym in symlist:
                    system_map[sym] = cat
            systems_mentioned = set(system_map.get(sym, None) for sym in discussed_symptoms if system_map.get(sym, None))
            if len(systems_mentioned) > 1:
                # Ask if symptoms are related or separate
                response = llm_summarize_and_next_steps(history, list(discussed_symptoms))
                update_session(session_id, "history", history + [(user_input, response)])
                active_thread_index = symptom_threads.index(active_thread)
                active_thread_status = active_thread["status"]
                return {"response": response, "history": history + [(user_input, response)], "session_id": session_id, "active_thread_index": active_thread_index, "active_thread_status": active_thread_status}
        # If the input is vague, complex, or not recognized, use LLM for follow-up
        if (not user_symptoms or is_vague_symptom(user_input) or needs_clarification(user_input, user_symptoms)):
            llm_followup = llm_generate_followup(user_input, list(discussed_symptoms), history)
            response = llm_followup
            update_session(session_id, "history", history + [(user_input, response)])
            active_thread_index = symptom_threads.index(active_thread)
            active_thread_status = active_thread["status"]
            return {"response": response, "history": history + [(user_input, response)], "session_id": session_id, "active_thread_index": active_thread_index, "active_thread_status": active_thread_status}
        # If qualifiers are present and a main symptom is found, acknowledge and proceed to analysis
        if user_symptoms and (qualifiers["location"] or qualifiers["type"]):
            qualifier_msg = []
            if qualifiers["location"]:
                qualifier_msg.append(f"location: {', '.join(qualifiers['location'])}")
            if qualifiers["type"]:
                qualifier_msg.append(f"type: {', '.join(qualifiers['type'])}")
            ack_response = f"Thank you for clarifying. I've noted your symptom(s) ({', '.join(user_symptoms)}) with {', '.join(qualifier_msg)}."
            # Instead of returning, proceed to analysis below
            # Optionally, store qualifiers in the thread for future use
            active_thread["qualifiers"] = {k: list(v) for k, v in qualifiers.items()}
            # Continue to analysis below, with ack_response prepended to the main response
        else:
            ack_response = None
        # 6. If match found, build response with confidence and dataset reasoning
        if user_symptoms:
            best_match, best_count, potential_matches, match_message = match_disease(list(discussed_symptoms), data["diseases"])
            if best_match is None:
                response = (
                    (ack_response + " ") if ack_response else ""
                ) + (
                    f"Thank you for sharing more details: {', '.join(discussed_symptoms)}. "
                    f"{match_message or 'Based on what you\'ve shared, I couldn\'t confidently match your symptoms to a specific condition in my dataset.'} "
                    "Could you describe your symptoms in more detail, or mention any other symptoms you have? This will help me provide a more accurate suggestion."
                )
                update_session(session_id, "history", history + [(user_input, response)])
                active_thread_index = symptom_threads.index(active_thread)
                active_thread_status = active_thread["status"]
                return {"response": response, "history": history + [(user_input, response)], "session_id": session_id, "active_thread_index": active_thread_index, "active_thread_status": active_thread_status}
            # Get confidence score from best_match (score is in potential_matches[0]["score"] if present)
            confidence = potential_matches[0]["score"] if potential_matches else 0.0
            # Track low confidence turns
            if confidence < 0.4:
                low_confidence_turns += 1
            else:
                low_confidence_turns = 0
            session["low_confidence_turns"] = low_confidence_turns
            # Escalate if persistent low confidence
            if low_confidence_turns >= max_low_confidence_turns:
                active_thread["status"] = "closed"
                session["symptom_threads"] = symptom_threads
                response = (ack_response + " ") if ack_response else ""
                response += "Thank you for letting me know. This topic is now closed. If you have another health concern or a new set of symptoms, please describe them and I'll start a new analysis."
                update_session(session_id, "history", history + [(user_input, response)])
                active_thread_index = symptom_threads.index(active_thread)
                active_thread_status = active_thread["status"]
                return {"response": response, "history": history + [(user_input, response)], "session_id": session_id, "active_thread_index": active_thread_index, "active_thread_status": active_thread_status}
            # Personalized advice
            personalized_advice = ""
            if "bloating" in discussed_symptoms or "gas" in discussed_symptoms:
                personalized_advice = "It's a good idea to avoid carbonated drinks, heavy meals, and foods that cause gas. "
            if "fatigue" in discussed_symptoms:
                personalized_advice += "Make sure to get enough rest and stay hydrated. "
            # Build follow-up if confidence is not high
            follow_up = ""
            if confidence < 0.7:
                from backend.disease_matching import suggest_follow_up_questions
                follow_ups = suggest_follow_up_questions(list(discussed_symptoms), data["diseases"])
                follow_ups = [q for q in follow_ups if all(sym not in discussed_symptoms and sym not in denied_symptoms for sym in re.findall(r'\b([a-zA-Z ]+?)\b', q))][:2]
                session["last_asked_symptoms"] = follow_ups
                if follow_ups:
                    follow_up = " To help me be more certain, do you also have any of these symptoms: " + " or ".join(follow_ups)
            else:
                session["last_asked_symptoms"] = []
            remedy = best_match.get('remedy', {})
            remedy_str = f"Remedy: {remedy.get('description', 'N/A')}\nDosage: {remedy.get('dosage', 'N/A')}\nCautions: {remedy.get('cautions', 'N/A')}"
            if remedy.get('description') or remedy.get('dosage') or remedy.get('cautions'):
                disclaimer_needed = True
            response = (
                (ack_response + " ") if ack_response else ""
            ) + (
                f"Thank you for sharing more details: {', '.join(discussed_symptoms)}. Based on what you've shared ({', '.join(discussed_symptoms)}), I am highly confident this could be {best_match.get('disease', 'N/A')}. "
                f"{remedy_str} "
                f"{personalized_advice}"
                f"{follow_up} "
            )
            # Always append a full disclaimer if giving advice
            response += "Note: This information is based on traditional Ayurvedic texts and is for informational purposes only. Please consult a qualified healthcare professional before making any health decisions or using any remedies."
            update_session(session_id, "history", history + [(user_input, response)])
            # Find the index of the active thread
            active_thread_index = symptom_threads.index(active_thread)
            active_thread_status = active_thread["status"]
            return {"response": response, "history": history + [(user_input, response)], "session_id": session_id, "active_thread_index": active_thread_index, "active_thread_status": active_thread_status}

    # If the active thread is closed, block further analysis and only invite new topics
    if active_thread["status"] == "closed":
        response = "Thank you for letting me know. This topic is now closed. If you have another health concern or a new set of symptoms, please describe them and I'll start a new analysis."
        update_session(session_id, "history", history + [(user_input, response)])
        active_thread_index = symptom_threads.index(active_thread)
        active_thread_status = active_thread["status"]
        return {"response": response, "history": history + [(user_input, response)], "session_id": session_id, "active_thread_index": active_thread_index, "active_thread_status": active_thread_status}

    session["symptom_threads"] = symptom_threads

    # FINAL CATCH-ALL: Always return a valid JSON response
    return {
        "response": "Sorry, I didn't understand that. Could you please rephrase or provide more details about your symptoms? If you are describing a health issue, please mention as many symptoms as you can. If you want to ask about Ayurveda or Tamra Bhasma, feel free!",
        "session_id": session_id,
        "history": history,
        "active_thread_index": 0,
        "active_thread_status": "active"
    }

@app.get("/")
def read_root():
    return {"message": "TamraBot API is running. Use /chat endpoint."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

def robust_extract_all_symptoms(user_input):
    """Extract canonical symptoms using all available methods: synonym, canonical, fuzzy, n-gram. Also extract location/type qualifiers."""
    text = user_input.lower().replace(',', ' ').replace('.', ' ').replace(';', ' ').replace('!', ' ').replace('?', ' ')
    text = text.replace(' and ', ' ').replace(' or ', ' ')
    tokens = [t.strip() for t in re.split(r',| and | or |/|;|\n|\.|\?|!', text) if t.strip()]
    ngrams = extract_all_ngrams(text, max_n=4)
    found = set()
    qualifiers = {"location": set(), "type": set()}
    # Canonical and synonym map
    for token in tokens:
        for phrase, canonical in SYMPTOM_SYNONYM_MAP.items():
            if phrase in token:
                found.add(canonical)
        for phrase, canonical in SYMPTOM_PHRASES.items():
            if phrase in token:
                found.add(canonical)
        # Fuzzy match
        close = difflib.get_close_matches(token, CANONICAL_SYMPTOM_LIST, n=1, cutoff=0.7)
        if close:
            found.add(close[0])
        if token in USER_SYNONYM_MAP:
            found.add(USER_SYNONYM_MAP[token])
    # N-gram match
    for ng in ngrams:
        if ng in master_symptom_index:
            found.add(ng)
    # Extract location/type qualifiers
    locations = ["head", "stomach", "chest", "back", "joints", "skin", "abdomen", "throat", "legs", "arms"]
    types = ["sharp", "dull", "constant", "intermittent", "burning", "throbbing", "aching", "pressure", "tightness", "swelling", "itching", "numbness"]
    for loc in locations:
        if loc in text:
            qualifiers["location"].add(loc)
    for typ in types:
        if typ in text:
            qualifiers["type"].add(typ)
    return list(found), qualifiers