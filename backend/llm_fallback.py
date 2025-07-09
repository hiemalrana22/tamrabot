"""
llm_fallback.py
Module for LLM fallback and output filtering.
"""
import requests

def get_llm_response(user_input, context, dataset, api_key, conversation_history=None):
    """Get a response from the LLM, using the dataset as context and conversation history."""
    history_context = ""
    if conversation_history and len(conversation_history) > 0:
        history_context = "\nPrevious conversation:\n"
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        for i, (user_msg, bot_msg) in enumerate(recent_history):
            history_context += f"User: {user_msg}\nTamraBot: {bot_msg}\n"
    
    system_prompt = (
        "You are TamraBot, an expert, friendly, and interactive AI assistant specializing in Tamra Bhasma and Ayurveda. "
        "You are provided with a dataset of diseases, symptoms, remedies, dosages, and cautions. "
        "When a user describes symptoms or asks about their health, always use the dataset as your primary source: frame your follow-up questions about symptoms, possible diseases, and remedies using only the symptoms and diseases found in the dataset. "
        "After suggesting a possible disease, always ask targeted follow-up questions using symptoms from the dataset to confirm or rule out possibilities. Present a short list of related symptoms for the user to confirm or deny. "
        "State your confidence in any disease match (e.g., 'I am moderately confident this could be...'). If confidence is low, ask for more details or escalate to fallback. "
        "Use warm, empathetic, and conversational language in all responses. "
        "Always explain why a disease or remedy is suggested, referencing the matching symptoms from the dataset. "
        "If a symptom is not in the dataset, clearly state this and ask the user to describe their symptoms using the provided options. "
        "Always provide remedy, dosage, and cautions from the dataset if available, and clearly state if Tamra Bhasma is not recommended for the described symptoms. "
        "Track which symptoms have already been discussed and avoid repeating questions. "
        "If you cannot determine a specific disease, or if the user input remains vague or confidence is low after several attempts, say so politely and suggest the user consult a healthcare professional. "
        "If you give medical advice, add the disclaimer only once at the end: 'Note: This information is based on traditional Ayurvedic texts. Please consult a qualified healthcare professional before use.' "
        "Never repeat disclaimers or information. "
        "Persona: You are knowledgeable, warm, and approachable."
    )

    prompt = (
        system_prompt +
        f"\n\n{context}\n" +
        f"{history_context}\n" 
        f"User: {user_input}\nTamraBot:"
    )
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://tamrabot.app",
        "X-Title": "TamraBot - General Assistant"
    }
    
    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.4,
        "top_p": 0.95,
        "timeout": 30
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenRouter API: {e}")
        return "I'm having trouble connecting to my knowledge base right now. Please try again in a moment."
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error processing OpenRouter API response: {e}")
        return "I encountered an issue processing your request. Please try asking in a different way."

def filter_llm_output(llm_output, dataset):
    """Filter LLM output to ensure it is safe, accurate, and dataset-aligned. Only add the disclaimer once if needed."""
    if not llm_output:
        return "I'm sorry, I couldn't generate a proper response. Please try asking in a different way."
    
    unsafe_keywords = [
        "poison", "toxic", "lethal", "deadly", "kill", "suicide", "harmful", 
        "overdose", "dangerous", "illegal", "death", "fatal", "die", "died",
        "mortality", "lethal dose", "ld50", "self-harm", "self harm"
    ]
    
    medical_disclaimers = [
        "not a doctor", "not medical advice", "consult a healthcare professional", 
        "seek medical advice", "consult your physician", "medical emergency",
        "consult a doctor", "consult an ayurvedic practitioner", "consult a qualified",
        "speak with your doctor", "medical supervision", "professional guidance",
        "Note: This information is based on traditional Ayurvedic texts. Please consult a qualified healthcare professional before use."
    ]
    
    output_lower = llm_output.lower()
    disclaimer_text = "Note: This information is based on traditional Ayurvedic texts. Please consult a qualified healthcare professional before use."
    
    for keyword in unsafe_keywords:
        if keyword in output_lower:
            safe_contexts = [f"not {keyword}", f"isn't {keyword}", f"is not {keyword}", 
                            f"no {keyword}", f"avoid {keyword}", f"prevent {keyword}",
                            f"without {keyword}", f"non-{keyword}", f"non {keyword}"]
            
            if not any(context in output_lower for context in safe_contexts):
                if disclaimer_text not in llm_output:
                    llm_output += f"\n\n{disclaimer_text}"
                break
    
    medical_advice_indicators = [
        "take", "use", "consume", "apply", "dosage", "dose", "treatment", "remedy",
        "prescription", "recommended", "should take", "can take", "may take", "helps with",
        "beneficial for", "effective for", "treats", "cures", "heals", "relieves"
    ]
    
    if any(indicator in output_lower for indicator in medical_advice_indicators) and \
       not any(disclaimer in output_lower for disclaimer in medical_disclaimers):
        if disclaimer_text not in llm_output:
            llm_output += f"\n\n{disclaimer_text}"
    
    if llm_output.count(disclaimer_text) > 1:
        parts = llm_output.split(disclaimer_text)
        llm_output = parts[0] + disclaimer_text + (parts[-1] if len(parts) > 2 else "")
    
    dataset_terms = set()
    
    for item in dataset.get('faq', []):
        question = item.get('question', '').lower()
        for word in question.split():
            if len(word) > 3 and word not in ['what', 'when', 'where', 'which', 'how', 'does', 'is', 'are', 'can', 'should']:
                dataset_terms.add(word)
        
        answer = item.get('answer', '').lower()
        for word in answer.split():
            if len(word) > 3 and word not in ['this', 'that', 'with', 'from', 'have', 'has', 'had', 'been', 'was', 'were']:
                dataset_terms.add(word)
    
    for item in dataset.get('diseases', []):
        for word in item.get('disease', '').lower().split():
            if len(word) > 3:
                dataset_terms.add(word)
        
        for symptom in item.get('symptoms', []):
            for word in symptom.lower().split():
                if len(word) > 3:
                    dataset_terms.add(word)
        
        if 'remedy' in item:
            remedy = item['remedy']
            for word in remedy.get('description', '').lower().split():
                if len(word) > 3:
                    dataset_terms.add(word)
    
    common_words = {'the', 'and', 'for', 'that', 'this', 'with', 'you', 'your', 'have', 'has', 'been', 'from', 'will'}
    output_words = set(word for word in output_lower.split() if len(word) > 3 and word not in common_words)
    matching_words = output_words.intersection(dataset_terms)
    
    match_percentage = len(matching_words) / max(1, len(output_words)) if output_words else 0
    
    if match_percentage < 0.25 and len(output_words) > 10:
        if disclaimer_text not in llm_output:
            llm_output += f"\n\nNote: Some of this information may go beyond what is in my core knowledge base. Please verify with authoritative sources."
    
    hallucination_phrases = [
        "according to research", "studies show", "clinical trials", "scientific evidence",
        "recent studies", "research indicates", "evidence suggests", "proven to", 
        "demonstrated that", "scientists have found"
    ]
    
    if any(phrase in output_lower for phrase in hallucination_phrases) and match_percentage < 0.4:
        if disclaimer_text not in llm_output:
            llm_output += "\n\nNote: I can only provide information based on traditional Ayurvedic knowledge in my dataset, not recent scientific research. Please consult authoritative sources for the latest scientific evidence."
    
    return llm_output