#!/bin/bash

# TamraBot Startup Script
echo "🌟 Starting TamraBot..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Check if psutil is installed (required for the startup script)
python3 -c "import psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing psutil..."
    pip3 install psutil
fi

# Run the Python startup script
python3 start_tamrabot.py