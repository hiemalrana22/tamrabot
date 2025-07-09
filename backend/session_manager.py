"""
session_manager.py
Module for managing user sessions and state.
"""

sessions = {}

def get_session(session_id):
    """Retrieve the session object for a given session_id."""
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "symptoms": [],
            "denied_symptoms": [],
            "collecting_symptoms": False,
            "last_intent": None,
            "last_answer_type": None,
            "welcomed": False,
            "symptom_attempts": 0,
            # Enhanced fields for intelligent dialogue
            "clarification_attempts": 0,
            "clarification_loop_count": 0,
            "escalation_status": False,
            "last_asked_question": None,
            "last_confidence": None,
            "last_user_input": None,
            "last_user_symptoms": set(),
            # Multi-threaded symptom management
            "symptom_threads": [
                {
                    "symptoms": [],
                    "denied_symptoms": [],
                    "status": "active"  # can be 'active', 'closed', or 'pending_clarification'
                }
            ]
        }
    return sessions[session_id]

def update_session(session_id, key, value):
    """Update a session key with a new value."""
    session = get_session(session_id)
    session[key] = value
    sessions[session_id] = session

def reset_session(session_id):
    """Reset the session state for a given session_id."""
    sessions[session_id] = {
        "history": [],
        "symptoms": [],
        "denied_symptoms": [],
        "collecting_symptoms": False,
        "last_intent": None,
        "last_answer_type": None,
        "welcomed": False,
        "symptom_attempts": 0,
        # Enhanced fields for intelligent dialogue
        "clarification_attempts": 0,
        "clarification_loop_count": 0,
        "escalation_status": False,
        "last_asked_question": None,
        "last_confidence": None,
        "last_user_input": None,
        "last_user_symptoms": set(),
        # Multi-threaded symptom management
        "symptom_threads": [
            {
                "symptoms": [],
                "denied_symptoms": [],
                "status": "active"
            }
        ]
    }

def hard_reset_session(session_id):
    sessions[session_id] = {
        "history": [],
        "symptoms": [],
        "denied_symptoms": [],
        "collecting_symptoms": False,
        "last_intent": None,
        "last_answer_type": None,
        "welcomed": False,
        "symptom_attempts": 0,
        "clarification_attempts": 0,
        "clarification_loop_count": 0,
        "escalation_status": False,
        "last_asked_question": None,
        "last_confidence": None,
        "last_user_input": None,
        "last_user_symptoms": set(),
        # Multi-threaded symptom management
        "symptom_threads": [
            {
                "symptoms": [],
                "denied_symptoms": [],
                "status": "active"
            }
        ]
    }