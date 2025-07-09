#!/usr/bin/env python3
"""
Simple test script for TamraBot functionality.
This script tests the basic functionality of the TamraBot API.
"""

import requests
import json
import sys
import os

# Check if the backend is running
def check_backend():
    try:
        response = requests.get("http://localhost:9000")
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
        else:
            print("❌ Backend returned unexpected status code:", response.status_code)
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running. Please start the backend server first.")
        print("   Run: cd backend && uvicorn app:app --reload")
        return False

# Test small talk functionality
def test_small_talk():
    print("\n🔍 Testing small talk...")
    payload = {"message": "Hello", "session_id": "test_session"}
    response = requests.post("http://localhost:9000/chat", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Small talk test passed. Response: {data['response'][:50]}...")
        return data["session_id"]
    else:
        print("❌ Small talk test failed")
        return None

# Test FAQ functionality
def test_faq(session_id):
    print("\n🔍 Testing FAQ...")
    payload = {"message": "What is Tamra Bhasma?", "session_id": session_id}
    response = requests.post("http://localhost:9000/chat", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ FAQ test passed. Response: {data['response'][:50]}...")
    else:
        print("❌ FAQ test failed")

# Test symptom functionality
def test_symptoms(session_id):
    print("\n🔍 Testing symptom collection...")
    payload = {"message": "I have abdominal pain and nausea", "session_id": session_id}
    response = requests.post("http://localhost:9000/chat", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Symptom test passed. Response: {data['response'][:50]}...")
    else:
        print("❌ Symptom test failed")

# Test LLM fallback
def test_llm_fallback(session_id):
    print("\n🔍 Testing LLM fallback...")
    # Check if OPENROUTER_API_KEY is set
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("⚠️ OPENROUTER_API_KEY not set. LLM fallback test skipped.")
        return
        
    payload = {"message": "Tell me about the history of Tamra Bhasma", "session_id": session_id}
    response = requests.post("http://localhost:9000/chat", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ LLM fallback test passed. Response: {data['response'][:50]}...")
    else:
        print("❌ LLM fallback test failed")

def main():
    print("🤖 TamraBot Test Script")
    print("=========================")
    
    # Check if backend is running
    if not check_backend():
        return
    
    # Run tests
    session_id = test_small_talk()
    if session_id:
        test_faq(session_id)
        test_symptoms(session_id)
        test_llm_fallback(session_id)
    
    print("\n✨ Tests completed")

if __name__ == "__main__":
    main()