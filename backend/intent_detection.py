"""
intent_detection.py
Module for detecting user intent: small talk, FAQ, symptom/diagnosis, out-of-scope.
"""

# Enhanced keyword lists for better intent detection
SMALL_TALK_KEYWORDS = [
    "hi", "hello", "hey", "how are you", "good morning", "good evening", "good afternoon",
    "bye", "goodbye", "see you", "talk to you later", "thanks", "thank you", "thank", 
    "yo", "what's up", "wassup", "sup", "howdy", "greetings", "nice to meet you",
    "how do you do", "pleased to meet", "good to see you", "welcome", "cheers"
]

FAQ_KEYWORDS = [
    "what", "how", "can", "is", "are", "does", "do", "should", "when", "where", "why",
    "which", "who", "whom", "whose", "will", "would", "could", "tell me about", "explain",
    "describe", "define", "meaning of", "purpose of", "use of", "benefit of", "advantage of",
    "disadvantage of", "side effect of", "risk of", "danger of", "safety of", "price of",
    "cost of", "preparation of", "ingredient in", "composition of", "difference between"
]

SYMPTOM_KEYWORDS = [
    # Physical symptoms
    "pain", "ache", "hurt", "sore", "tender", "swollen", "inflamed", "burning", "stinging",
    "itching", "rash", "fever", "chill", "cold", "hot", "sweating", "dizzy", "faint",
    "weak", "tired", "exhausted", "fatigue", "nausea", "vomit", "sick", "queasy",
    "bloating", "bloated", "gas", "flatulence", "burping", "belching", "hiccup",
    "cough", "sneeze", "sneezing", "congestion", "runny nose", "stuffy", "phlegm",
    "mucus", "spit", "saliva", "dry mouth", "thirsty", "dehydrated", "headache",
    "migraine", "pressure", "pounding", "throbbing", "sharp", "dull", "constant",
    "intermittent", "chronic", "acute", "sudden", "gradual", "worsen", "improve",
    
    # Digestive symptoms
    "diarrhea", "constipation", "indigestion", "heartburn", "acid reflux", "stomach",
    "intestine", "bowel", "stool", "poop", "feces", "urine", "pee", "urinate",
    "frequent", "urgent", "burning", "painful", "difficulty", "unable", "jaundice",
    "yellow", "pale", "dark", "black", "red", "bloody", "blood", "mucus", "pus",
    "discharge", "leak", "seep", "ooze", "loss of appetite", "hungry", "thirst",
    
    # General terms
    "symptom", "problem", "issue", "condition", "disease", "disorder", "syndrome",
    "illness", "sickness", "ailment", "malady", "affliction", "suffering", "distress",
    "discomfort", "unwell", "unease", "malaise", "feel", "feeling", "felt", "experiencing",
    "having", "got", "gotten", "developed", "noticed", "observed", "detected", "diagnosed",
    "doctor said", "medical", "health", "healthy", "unhealthy", "sick", "ill", "poor health"
]

# Add Ayurveda keywords for stricter FAQ detection
AYURVEDA_KEYWORDS = ['tamra', 'bhasma', 'ayurved', 'copper', 'medicine', 'herb', 'mineral']

# Add special patterns for 'which diseases' and 'tell me more' queries
WHICH_DISEASES_PATTERNS = [
    'which diseases', 'diseases that can be healed', 'what diseases', 'what can it heal', 'what does it treat', 'what is it used for'
]
TELL_ME_MORE_PATTERNS = [
    'tell me more', 'more details', 'more information', 'explain more', 'give me more', 'elaborate'
]

BODY_PARTS = ['stomach', 'chest', 'head', 'skin', 'abdomen', 'back', 'joints', 'throat', 'leg', 'arm', 'belly', 'tummy', 'shoulder', 'eye', 'ear', 'mouth', 'face']
VAGUE_HEALTH_PHRASES = ['not well', 'not feeling', 'unwell', 'not good', 'not okay', 'not ok', 'not right', 'feeling off', 'feeling weird', 'feeling bad', 'not normal', 'not healthy', 'feeling sick', 'feeling ill']

# Add more intent categories
JOKE_KEYWORDS = [
    'joke', 'funny', 'laugh', 'make me laugh', 'tell me a joke', 'pun', 'humor', 'knock knock', 'why did', 'chicken cross', 'dad joke', 'lol', 'lmao', 'rofl', 'haha', 'hehe', 'hilarious'
]
COMPLIMENT_KEYWORDS = [
    'good bot', 'nice bot', 'smart', 'intelligent', 'awesome', 'amazing', 'great job', 'well done', 'thank you', 'thanks', 'love you', 'cool', 'impressive', 'helpful', 'useful', 'best bot'
]
INSULT_KEYWORDS = [
    'bad bot', 'stupid', 'dumb', 'useless', 'idiot', 'hate you', 'worst', 'annoying', 'boring', 'shut up', 'silly', 'lame', 'terrible', 'not helpful', 'not useful', 'waste', 'sucks'
]
HELP_KEYWORDS = [
    'help', 'assist', 'support', 'how to use', 'instructions', 'guide', 'what can you do', 'features', 'capabilities', 'usage', 'manual', 'explain yourself', 'who are you', 'about you', 'your purpose', 'your job', 'your function', 'your role'
]
META_QUESTION_KEYWORDS = [
    'are you real', 'are you ai', 'are you human', 'are you a robot', 'who made you', 'who created you', 'your creator', 'your developer', 'your name', 'your age', 'your birthday', 'your origin', 'your story', 'your background'
]
CHITCHAT_KEYWORDS = [
    'how is the weather', 'what is the time', 'what time is it', 'how is your day', 'how are you doing', 'what are you doing', 'what is up', "what's up", 'how is life', 'how is everything', 'how is it going', "what's going on", 'what are you up to', 'how do you feel', 'do you sleep', 'do you eat', 'do you dream', 'do you have feelings', 'do you have emotions', 'do you get tired', 'do you get bored', 'do you get angry', 'do you get sad', 'do you get happy', 'do you have friends', 'do you have family', 'do you have a pet', 'do you like music', 'do you like movies', 'do you like books', 'do you like games', 'do you like sports', 'do you like food', 'do you like travel', 'do you like jokes', 'do you like humans', 'do you like ai', 'do you like robots', 'do you like animals', 'do you like nature', 'do you like art', 'do you like science', 'do you like technology', 'do you like learning', 'do you like helping', 'do you like chatting', 'do you like talking', 'do you like questions', 'do you like answers', 'do you like advice', 'do you like suggestions', 'do you like compliments', 'do you like criticism', 'do you like feedback', 'do you like challenges', 'do you like puzzles', 'do you like riddles', 'do you like trivia', 'do you like fun', 'do you like to laugh', 'do you like to smile', 'do you like to think', 'do you like to work', 'do you like to rest', 'do you like to play', 'do you like to win', 'do you like to lose', 'do you like to try', 'do you like to learn', 'do you like to teach', 'do you like to share', 'do you like to listen', 'do you like to talk', 'do you like to read', 'do you like to write', 'do you like to draw', 'do you like to code', 'do you like to build', 'do you like to create', 'do you like to explore', 'do you like to discover', 'do you like to invent', 'do you like to imagine', 'do you like to wonder', 'do you like to ask', 'do you like to answer', 'do you like to help', 'do you like to support', 'do you like to care', 'do you like to inspire', 'do you like to motivate', 'do you like to encourage', 'do you like to empower', 'do you like to connect', 'do you like to communicate', 'do you like to interact', 'do you like to collaborate', 'do you like to cooperate', 'do you like to participate', 'do you like to contribute', 'do you like to learn new things', 'do you like to try new things', 'do you like to meet new people', 'do you like to make friends', 'do you like to have fun', 'do you like to enjoy life', 'do you like to be happy', 'do you like to be kind', 'do you like to be helpful', 'do you like to be positive', 'do you like to be creative', 'do you like to be curious', 'do you like to be open-minded', 'do you like to be honest', 'do you like to be respectful', 'do you like to be responsible', 'do you like to be reliable', 'do you like to be trustworthy', 'do you like to be friendly', 'do you like to be polite', 'do you like to be patient', 'do you like to be understanding', 'do you like to be generous', 'do you like to be grateful', 'do you like to be humble', 'do you like to be confident', 'do you like to be brave', 'do you like to be strong', 'do you like to be wise', 'do you like to be smart', 'do you like to be funny', 'do you like to be silly', 'do you like to be serious', 'do you like to be quiet', 'do you like to be loud', 'do you like to be fast', 'do you like to be slow', 'do you like to be early', 'do you like to be late', 'do you like to be on time', 'do you like to be organized', 'do you like to be messy', 'do you like to be neat', 'do you like to be tidy', 'do you like to be clean', 'do you like to be dirty', 'do you like to be busy', 'do you like to be free', 'do you like to be alone', 'do you like to be with others', 'do you like to be in a group', 'do you like to be in a team', 'do you like to be in a crowd', 'do you like to be in a quiet place', 'do you like to be in a noisy place', 'do you like to be in a big place', 'do you like to be in a small place', 'do you like to be in a new place', 'do you like to be in a familiar place', 'do you like to be in a safe place', 'do you like to be in an exciting place', 'do you like to be in a comfortable place', 'do you like to be in a challenging place', 'do you like to be in a fun place', 'do you like to be in a happy place', 'do you like to be in a peaceful place', 'do you like to be in a beautiful place', 'do you like to be in a special place', 'do you like to be in a favorite place', 'do you like to be in a secret place', 'do you like to be in a magical place', 'do you like to be in a mysterious place', 'do you like to be in a wonderful place', 'do you like to be in an interesting place', 'do you like to be in a boring place', 'do you like to be in a strange place', 'do you like to be in a weird place', 'do you like to be in a different place', 'do you like to be in a unique place', 'do you like to be in a creative place', 'do you like to be in a learning place', 'do you like to be in a working place', 'do you like to be in a playing place', 'do you like to be in a relaxing place', 'do you like to be in a thinking place', 'do you like to be in a dreaming place', 'do you like to be in a remembering place', 'do you like to be in a forgetting place', 'do you like to be in a loving place', 'do you like to be in a caring place', 'do you like to be in a sharing place', 'do you like to be in a giving place', 'do you like to be in a receiving place', 'do you like to be in a helping place', 'do you like to be in a supporting place', 'do you like to be in a connecting place', 'do you like to be in a communicating place', 'do you like to be in an inspiring place', 'do you like to be in a motivating place', 'do you like to be in an encouraging place', 'do you like to be in an empowering place', 'do you like to be in a collaborating place', 'do you like to be in a cooperating place', 'do you like to be in a participating place', 'do you like to be in a contributing place', 'do you like to be in a learning environment', 'do you like to be in a working environment', 'do you like to be in a playing environment', 'do you like to be in a relaxing environment', 'do you like to be in a thinking environment', 'do you like to be in a dreaming environment', 'do you like to be in a remembering environment', 'do you like to be in a forgetting environment', 'do you like to be in a loving environment', 'do you like to be in a caring environment', 'do you like to be in a sharing environment', 'do you like to be in a giving environment', 'do you like to be in a receiving environment', 'do you like to be in a helping environment', 'do you like to be in a supporting environment', 'do you like to be in a connecting environment', 'do you like to be in a communicating environment', 'do you like to be in an inspiring environment', 'do you like to be in a motivating environment', 'do you like to be in an encouraging environment', 'do you like to be in an empowering environment', 'do you like to be in a collaborating environment', 'do you like to be in a cooperating environment', 'do you like to be in a participating environment', 'do you like to be in a contributing environment'
]

def detect_intent(user_input, dataset):
    """Detect the intent of the user input (expanded)."""
    text = user_input.lower().strip()
    import string
    text_clean = text.translate(str.maketrans('', '', string.punctuation)).strip()
    short_greetings = {"hi", "hello", "hey", "yo", "sup", "howdy", "hola"}
    if text_clean in short_greetings:
        return "greeting"
    if any(kw in text for kw in JOKE_KEYWORDS):
        return "joke"
    if any(kw in text for kw in COMPLIMENT_KEYWORDS):
        return "compliment"
    if any(kw in text for kw in INSULT_KEYWORDS):
        return "insult"
    if any(kw in text for kw in HELP_KEYWORDS):
        return "help_request"
    if any(kw in text for kw in META_QUESTION_KEYWORDS):
        return "meta_question"
    if any(kw in text for kw in CHITCHAT_KEYWORDS):
        return "chitchat"
    if any(bp in text for bp in BODY_PARTS) or any(vague in text for vague in VAGUE_HEALTH_PHRASES):
        return 'symptom'
    if not text or len(text) < 2:
        return "small_talk"
    if any(pat in text for pat in WHICH_DISEASES_PATTERNS):
        return "which_diseases"
    if any(pat in text for pat in TELL_ME_MORE_PATTERNS):
        return "tell_me_more"
    small_talk_score = calculate_small_talk_score(text)
    faq_score = calculate_faq_score(text, dataset)
    symptom_score = calculate_symptom_score(text)
    scores = {
        "small_talk": small_talk_score,
        "faq": faq_score,
        "symptom": symptom_score
    }
    highest_intent = max(scores, key=scores.get)
    highest_score = scores[highest_intent]
    if highest_intent == "faq" and not any(word in text for word in AYURVEDA_KEYWORDS):
        return "out_of_scope"
    if highest_score < 0.3:
        if text_clean in short_greetings:
            return "greeting"
        return "out_of_scope"
    for greet in short_greetings:
        if greet in text_clean:
            return "greeting"
    return highest_intent


def calculate_small_talk_score(text):
    """Calculate a score for how likely the input is small talk."""
    # Direct match with small talk keywords
    direct_matches = sum(1 for kw in SMALL_TALK_KEYWORDS if kw in text)
    
    # Check for exact greetings
    exact_greetings = ['hi', 'hello', 'hey', 'thanks', 'thank you', 'bye']
    exact_match = any(text == greeting for greeting in exact_greetings)
    
    # Very short text is likely small talk
    length_factor = 1.0 if len(text) < 15 else 0.5
    
    # Calculate final score
    score = (direct_matches * 0.3) + (exact_match * 0.5) + (length_factor * 0.2)
    return min(score, 1.0)  # Cap at 1.0


def calculate_faq_score(text, dataset):
    """Calculate a score for how likely the input is an FAQ."""
    # Check for question marks
    has_question_mark = 0.4 if '?' in text else 0
    
    # Check for FAQ keywords at the start
    starts_with_faq = any(text.startswith(kw) for kw in FAQ_KEYWORDS)
    faq_keyword_score = 0.4 if starts_with_faq else 0
    
    # Check for mentions of Tamra Bhasma or related terms
    tamra_terms = ['tamra', 'bhasma', 'ayurved', 'copper', 'medicine', 'herb', 'mineral']
    tamra_mentions = sum(1 for term in tamra_terms if term in text) * 0.1
    
    # Check similarity with existing FAQs in dataset
    faq_similarity = 0
    if dataset and 'faq' in dataset:
        for faq in dataset['faq']:
            question = faq.get('question', '').lower()
            if any(word in question and word in text for word in question.split() if len(word) > 3):
                faq_similarity += 0.1
    
    # Calculate final score
    score = has_question_mark + faq_keyword_score + min(tamra_mentions, 0.3) + min(faq_similarity, 0.3)
    return min(score, 1.0)  # Cap at 1.0


def calculate_symptom_score(text):
    """Calculate a score for how likely the input is symptom-related."""
    # Count symptom keywords
    symptom_matches = sum(1 for kw in SYMPTOM_KEYWORDS if kw in text)
    symptom_score = min(symptom_matches * 0.15, 0.6)  # Cap at 0.6
    
    # Check for first-person health statements
    health_phrases = ['i feel', 'i am feeling', 'i have', 'i am experiencing', 'i got', 'i am having']
    health_score = 0.3 if any(phrase in text for phrase in health_phrases) else 0
    
    # Check for body parts
    body_parts = ['head', 'stomach', 'chest', 'back', 'throat', 'nose', 'eye', 'ear', 'leg', 'arm', 'skin']
    body_part_score = min(sum(0.05 for part in body_parts if part in text), 0.2)
    
    # Calculate final score
    score = symptom_score + health_score + body_part_score
    return min(score, 1.0)  # Cap at 1.0


# Keep these for backward compatibility
def is_small_talk(user_input):
    text = user_input.lower()
    return calculate_small_talk_score(text) > 0.3


def is_faq(user_input):
    text = user_input.lower()
    # Simple check without dataset
    return text.strip().endswith('?') or any(text.strip().startswith(kw) for kw in FAQ_KEYWORDS)


def is_symptom_query(user_input):
    text = user_input.lower()
    return calculate_symptom_score(text) > 0.3


def is_out_of_scope(user_input):
    # If not any of the above
    return not (is_small_talk(user_input) or is_faq(user_input) or is_symptom_query(user_input))