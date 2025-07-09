"""
disease_matching.py
Module for matching user symptoms to diseases in the dataset.
"""

def match_disease(symptoms, disease_data):
    """Return the best matching disease and overlap count, or None if no match."""
    best_match = None
    best_count = 0
    best_score = 0
    potential_matches = []
    
    if not symptoms:
        return (None, 0, [], "Please provide specific symptoms to help me identify the condition.")
    
    symptoms = [symptom.lower().strip() for symptom in symptoms]
    
    for entry in disease_data:
        disease_symptoms = set([s.lower().strip() for s in entry['symptoms']])
        overlap = set(symptoms) & disease_symptoms
        
        overlap_count = len(overlap)
        
        coverage = overlap_count / len(disease_symptoms) if disease_symptoms else 0
        
        relevance = overlap_count / len(symptoms) if symptoms else 0
        
        score = (0.7 * coverage) + (0.3 * relevance)
        
        if overlap_count > 0:
            potential_matches.append({
                'disease': entry,
                'overlap_count': overlap_count,
                'coverage': coverage,
                'relevance': relevance,
                'score': score
            })
    
    potential_matches.sort(key=lambda x: x['score'], reverse=True)
    
    if potential_matches:
        best_match_data = potential_matches[0]
        
        if (best_match_data['overlap_count'] >= 3 or best_match_data['coverage'] >= 0.6):
            best_match = best_match_data['disease']
            best_count = best_match_data['overlap_count']
            best_score = best_match_data['score']
            message = None
        elif (best_match_data['overlap_count'] >= 2 or best_match_data['coverage'] >= 0.4):
            best_match = best_match_data['disease']
            best_count = best_match_data['overlap_count']
            best_score = best_match_data['score']
            message = f"Based on your symptoms, this might be {best_match['disease']}. However, to be more certain, could you describe any additional symptoms you're experiencing?"
        else:
            similar_matches = [m for m in potential_matches if m['score'] >= best_match_data['score'] * 0.8]
            
            if len(similar_matches) > 1:
                disease_names = [m['disease']['disease'] for m in similar_matches[:3]]
                message = f"Your symptoms could relate to several conditions: {', '.join(disease_names)}. Could you provide more specific symptoms to help me identify the exact condition?"
            else:
                message = "Your symptoms are quite general. Could you describe any additional symptoms, or specify which symptoms are most severe?"
            
            best_match = None
            best_count = 0
    else:
        message = "I couldn't find a specific match for your symptoms. Could you provide more detailed symptoms or describe what you're experiencing?"
    
    return (best_match, best_count, potential_matches, message)

def get_best_disease_match(symptoms, disease_data):
    """Return the disease with the highest symptom overlap, or None if no match."""
    match, count, potential_matches, message = match_disease(symptoms, disease_data)
    return match

def get_matching_analysis(symptoms, disease_data):
    """Return detailed matching analysis including potential matches and recommendations."""
    match, count, potential_matches, message = match_disease(symptoms, disease_data)
    return {
        'best_match': match,
        'overlap_count': count,
        'potential_matches': potential_matches,
        'message': message,
        'needs_clarification': match is None or message is not None
    }

def suggest_follow_up_questions(symptoms, disease_data):
    """Suggest follow-up questions based on the user's symptoms."""
    analysis = get_matching_analysis(symptoms, disease_data)
    
    if analysis['needs_clarification']:
        if analysis['potential_matches']:
            top_matches = analysis['potential_matches'][:3]
            questions = []
            
            for match in top_matches:
                disease = match['disease']
                missing_symptoms = [s for s in disease['symptoms'] if s not in symptoms]
                if missing_symptoms:
                    questions.append(f"Do you experience any of these symptoms: {', '.join(missing_symptoms[:3])}?")
            
            return questions
        else:
            return [
                "Could you describe your symptoms in more detail?",
                "When did these symptoms start?",
                "Are there any specific triggers or patterns?",
                "Do you have any other health conditions?"
            ]
    
    return []