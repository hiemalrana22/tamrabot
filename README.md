# TamraBot Project

## Overview
TamraBot is an AI-powered chatbot for answering questions about Tamra Bhasma and Ayurveda. It uses a FastAPI backend and a simple HTML/JS frontend.

---

## Directory Structure

- `backend/` — FastAPI backend (API server)
- `frontend/` — Static frontend (HTML/JS/CSS)
- `backend/data/chatbot_data.json` — Data file required by backend

---

## Setup & Run Instructions

### Option 1: Using the start_tamrabot.sh script (Recommended)

```sh
# Install dependencies first
cd backend
pip install -r requirements.txt
cd ..

# Set your API keys
export OPENAI_API_KEY=your-openai-api-key-here
export OPENROUTER_API_KEY=your-openrouter-api-key-here  # Optional but recommended

# Run the application
./start_tamrabot.sh
```

This script will:
- Check if the required API key is set
- Start the backend server
- Start a simple HTTP server for the frontend
- Open the application in your default browser

### Option 2: Using the run.sh script

```sh
# Install dependencies first
cd backend
pip install -r requirements.txt
cd ..

# Set your API keys
export OPENAI_API_KEY=your-openai-api-key-here
export OPENROUTER_API_KEY=your-openrouter-api-key-here  # Optional but recommended

# Run the application
./run.sh
```

### Option 3: Manual Setup

#### 1. Backend (API Server)

##### a. Install Python dependencies
```sh
cd backend
pip install -r requirements.txt
```

##### b. Set your API keys
```sh
# Required for basic functionality
export OPENAI_API_KEY=your-openai-api-key-here

# Required for LLM fallback functionality
export OPENROUTER_API_KEY=your-openrouter-api-key-here
```

##### c. Run the backend server
```sh
uvicorn app:app --reload
```

The backend will be available at `http://localhost:9000`.

#### 2. Frontend (Chat UI)

```sh
cd frontend
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

---

## Health Check
- The backend now provides a `/health` endpoint at `http://localhost:9000/health`.
- The frontend will warn you if the backend is not reachable.

---

## Troubleshooting
- If you see `{ "detail": "Not Found" }` at `/`, the backend is running! Use the chat UI to interact.
- If you get an error about the OpenAI API key, make sure you set the `OPENAI_API_KEY` environment variable.
- If the LLM fallback isn't working, check that you've set the `OPENROUTER_API_KEY` environment variable.
- If you see a warning about the backend not being reachable, make sure you:
  - Started the backend from the correct directory (the backend now finds its data file robustly, but always run from the project root or backend directory for best results).
  - Set your API keys.
  - Have no port conflicts on 9000 (backend) or 8080 (frontend).
- If you see a server error in the chat window, the actual error message will be shown for easier debugging.

---

## Security Note
**Never commit your API keys (OpenAI or OpenRouter) to source control.**

---

## License
MIT