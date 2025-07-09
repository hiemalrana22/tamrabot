#!/bin/bash

# Load environment variables from .env if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded environment variables from .env file."
fi

# Set your OpenAI API key here or ensure it's set in your environment
# Check if OPENAI_API_KEY is already set in the environment
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY is not set. Please set it before running this script."
    echo "You can set it by running: export OPENAI_API_KEY=your-key-here"
    exit 1
fi

# Check if OPENROUTER_API_KEY is set in the environment
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Warning: OPENROUTER_API_KEY is not set. LLM fallback will not work properly."
    echo "You can set it by running: export OPENROUTER_API_KEY=your-key-here"
    # Not exiting as this is optional
fi

# Check if required ports are free
check_port() {
    local port=$1
    if lsof -i :$port >/dev/null 2>&1; then
        echo "Error: Port $port is already in use. Please free it before running TamraBot."
        exit 1
    fi
}

check_port 9000
check_port 8080

# Install backend dependencies if needed
cd backend
if ! python3 -c "import fastapi, uvicorn, sentence_transformers, langchain, openai, numpy, sklearn, multipart" 2>/dev/null; then
    echo "Installing backend dependencies..."
    pip install -r requirements.txt || { echo "Dependency installation failed."; exit 1; }
else
    echo "All backend dependencies are already installed."
fi
cd ..

# Start backend (in background)
echo "Starting backend server on http://localhost:9000 ..."
cd backend
uvicorn app:app --reload --host 127.0.0.1 --port 9000 &
BACKEND_PID=$!
cd ..

# Start frontend server (in background)
echo "Starting frontend server on http://localhost:8080 ..."
cd frontend
python3 -m http.server 8080 &
FRONTEND_PID=$!
cd ..

# Wait a moment for servers to start
sleep 2

# Print URLs
echo "----------------------------------------"
echo "TamraBot is starting!"
echo "Backend:  http://localhost:9000"
echo "Frontend: http://localhost:8080"
echo "----------------------------------------"

# Open the frontend in the default browser (cross-platform)
if command -v open >/dev/null; then
    open http://localhost:8080
elif command -v xdg-open >/dev/null; then
    xdg-open http://localhost:8080
elif command -v start >/dev/null; then
    start http://localhost:8080
else
    echo "Please open http://localhost:8080 in your browser."
fi

# Wait for user to press Ctrl+C, then clean up
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait