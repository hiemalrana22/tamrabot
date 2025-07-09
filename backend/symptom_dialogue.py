"""
symptom_dialogue.py
Module for multi-turn symptom dialogue and clarification.
"""

from .disease_matching import get_matching_analysis, suggest_follow_up_questions

def ask_for_more_symptoms(current_symptoms, all_symptoms):
    """Ask the user for more symptoms, prioritizing those that help narrow down the diagnosis."""
    if not current_symptoms:
        return "Could you please describe any symptoms you're experiencing? If you're not sure, you can mention any changes you've noticed or just say how you're feeling in your own words."
    else:
        remaining_symptoms = [s for s in all_symptoms if s not in current_symptoms]
        sample_symptoms = remaining_symptoms[:5] if len(remaining_symptoms) > 5 else remaining_symptoms
        return f"Thank you for sharing. Do you also have any of these symptoms: {', '.join(sample_symptoms)}?"

def suggest_next_symptom(current_symptoms, disease_matrix):
    """Suggest the next most useful symptom to ask about, based on current symptoms and disease matrix."""
    # Find diseases that match current symptoms
    possible_diseases = []
    for disease in disease_matrix:
        if all(symptom in disease['symptoms'] for symptom in current_symptoms):
            possible_diseases.append(disease)
    
    if not possible_diseases:
        return None
    
    # Find symptoms that appear in these diseases but aren't in current symptoms
    candidate_symptoms = set()
    for disease in possible_diseases:
        for symptom in disease['symptoms']:
            if symptom not in current_symptoms:
                candidate_symptoms.add(symptom)
    
    # Return the most common symptom among candidate diseases
    if not candidate_symptoms:
        return None
    
    symptom_counts = {}
    for symptom in candidate_symptoms:
        symptom_counts[symptom] = sum(1 for disease in possible_diseases if symptom in disease['symptoms'])
    
    return max(symptom_counts, key=symptom_counts.get)

def handle_symptom_clarification(user_input, clarification_context, disease_data):
    """Handle clarification of vague or ambiguous symptoms.
    
    This function processes user responses to symptom clarification requests,
    extracting more specific symptom information when possible.
    
    Args:
        user_input: The user's response text
        clarification_context: Dictionary containing context about what's being clarified
        disease_data: List of disease data for matching
        
    Returns:
        Dictionary with extracted symptom information or clarification response
    """
    user_input_lower = user_input.lower()
    
    # Define common vague terms and their specific variants
    vague_to_specific = {
        "pain": {
            "locations": ["head", "stomach", "chest", "back", "joint", "throat", "ear"],
            "types": ["sharp", "dull", "throbbing", "burning", "stabbing", "aching"]
        },
        "discomfort": {
            "locations": ["abdominal", "chest", "throat", "digestive"],
            "types": ["bloating", "pressure", "tightness", "heaviness"]
        },
        "feeling": {
            "types": ["nauseous", "dizzy", "weak", "tired", "fatigued", "exhausted"]
        }
    }
    
    # Check if we're in the middle of clarifying a specific vague term
    vague_term = clarification_context.get("vague_term") if clarification_context else None
    all_symptoms = clarification_context.get("all_symptoms", []) if clarification_context else []
    
    if vague_term:
        # We're following up on a previous clarification request
        specific_variants = []
        
        # Check for locations
        if "locations" in vague_to_specific.get(vague_term, {}):
            for location in vague_to_specific[vague_term]["locations"]:
                if location in user_input_lower:
                    specific_variants.append(f"{location} {vague_term}")
        
        # Check for types
        if "types" in vague_to_specific.get(vague_term, {}):
            for type_ in vague_to_specific[vague_term]["types"]:
                if type_ in user_input_lower:
                    specific_variants.append(f"{type_} {vague_term}")
        
        # If we found specific variants, return the first one
        if specific_variants:
            return {"symptom": specific_variants[0], "clarified": True}
        
        # If no specific variants found, provide another prompt
        return {"response": f"I'm still not clear about your {vague_term}. If it's hard to describe, you can just tell me how it feels or if anything else has changed. I'm here to help!", "clarified": False}
    
    # Check if the input contains any vague terms that need clarification
    vague_terms = list(vague_to_specific.keys()) + ["feeling bad", "not well", "sick", "unwell"]
    for term in vague_terms:
        if term in user_input_lower:
            # Create context for follow-up
            new_context = {"vague_term": term}
            
            # For compound terms like "feeling bad", map to the base term
            base_term = term
            if term not in vague_to_specific:
                if "feeling" in term:
                    base_term = "feeling"
                else:
                    base_term = None
            
            # Generate a clarification request
            if base_term and base_term in vague_to_specific:
                locations = vague_to_specific[base_term].get("locations", [])
                types = vague_to_specific[base_term].get("types", [])
                
                location_examples = ", ".join(locations[:3]) if locations else ""
                type_examples = ", ".join(types[:3]) if types else ""
                
                if location_examples and type_examples:
                    response = f"Could you be more specific about your {term}? Is it in your {location_examples}, or is it {type_examples}? If it's hard to describe, just let me know how it feels."
                elif location_examples:
                    response = f"Could you tell me where your {term} is located? For example, is it in your {location_examples}? Or just describe it in your own words."
                elif type_examples:
                    response = f"Could you describe what kind of {term} it is? For example, is it {type_examples}? Or just let me know how it feels."
                else:
                    response = f"Could you be more specific about your {term}? Or just describe it in your own words."
            else:
                # For terms without specific mappings, suggest some symptoms
                if all_symptoms:
                    response = f"Could you be more specific about your symptoms? For example, do you have any of these: {', '.join(all_symptoms[:5])}? Or just describe how you feel."
                else:
                    response = f"Could you be more specific about your symptoms? For example, do you have fever, headache, or stomach pain? Or just describe how you feel."
            
            return {"response": response, "clarified": False, "context": new_context}
    
    # If no vague terms found, return a generic clarification request
    if all_symptoms:
        return {"response": f"I'm not sure I understand your symptoms. Could you mention if you have any specific symptoms like {', '.join(all_symptoms[:5])}? Or just describe how you feel.", "clarified": False}
    else:
        return {"response": "I'm not sure I understand your symptoms. Could you describe them more clearly? For example, do you have pain, fever, cough, or other specific symptoms? Or just describe how you feel.", "clarified": False}

def get_intelligent_follow_up(symptoms, disease_data):
    """Get intelligent follow-up questions based on current symptoms and disease matching."""
    analysis = get_matching_analysis(symptoms, disease_data)
    
    if analysis['needs_clarification']:
        if analysis['potential_matches']:
            # Generate specific questions based on potential matches
            questions = []
            top_matches = analysis['potential_matches'][:3]
            
            for match in top_matches:
                disease = match['disease']
                missing_symptoms = [s for s in disease['symptoms'] if s.lower() not in [s.lower() for s in symptoms]]
                if missing_symptoms:
                    questions.append(f"For {disease['disease']}, do you also have: {', '.join(missing_symptoms[:3])}?")
            
            if questions:
                return questions[:2]  # Limit to 2 questions
        
        # Generic follow-up questions
        return [
            "Could you describe your symptoms in more detail?",
            "When did these symptoms start?",
            "Are there any specific triggers or patterns?",
            "Do you have any other health conditions?"
        ]
    
    return []

def handle_symptom_analysis(symptoms, disease_data):
    """Handle comprehensive symptom analysis and provide intelligent responses."""
    # Normalize symptoms
    norm_symptoms = [s.lower().strip() for s in symptoms]
    best_match = None
    best_overlap = 0
    best_overlap_set = set()
    for disease in disease_data:
        disease_symptoms = [s.lower().strip() for s in disease['symptoms']]
        overlap = set(norm_symptoms) & set(disease_symptoms)
        if len(overlap) > best_overlap:
            best_overlap = len(overlap)
            best_match = disease
            best_overlap_set = overlap
    response = {
        'disease_match': None,
        'needs_clarification': False,
        'message': None,
        'follow_up_questions': [],
        'detailed_info': None
    }
    if best_match and best_overlap > 0:
        # Always return the best match if any overlap
        remedy = best_match['remedy']
        response['disease_match'] = best_match
        response['needs_clarification'] = False
        response['message'] = None
        response['detailed_info'] = {
            'disease_name': best_match['disease'],
            'applicable': best_match['tamra_applicable'],
            'remedy': remedy,
            'matching_symptoms': list(best_overlap_set)
        }
        return response
    else:
        # No match at all, ask for more symptoms
        response['disease_match'] = None
        response['needs_clarification'] = True
        response['message'] = "I couldn't find a specific match for your symptoms. Could you provide more detailed symptoms or describe what you're experiencing? Could you describe your symptoms in more detail? When did these symptoms start? Are there any specific triggers or patterns? Do you have any other health conditions?"
        response['follow_up_questions'] = [
            "Could you describe your symptoms in more detail?",
            "When did these symptoms start?",
            "Are there any specific triggers or patterns?",
            "Do you have any other health conditions?"
        ]
        return response