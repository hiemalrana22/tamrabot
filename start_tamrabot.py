#!/usr/bin/env python3
"""
TamraBot Startup Script
This script ensures TamraBot starts correctly every time.
"""

import os
import sys
import subprocess
import time
import signal
import psutil
from pathlib import Path

def kill_existing_processes():
    """Kill any existing TamraBot processes."""
    print("🔄 Stopping any existing TamraBot processes...")
    
    # Kill processes by name
    processes_to_kill = [
        "python app.py",
        "uvicorn app:app",
        "python -m http.server",
        "http.server"
    ]
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            for kill_pattern in processes_to_kill:
                if kill_pattern in cmdline:
                    print(f"🛑 Killing process {proc.info['pid']}: {cmdline}")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    time.sleep(2)  # Wait for processes to die

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'requests',
        'numpy',
        'scikit-learn',
        'sentence-transformers',
        'langchain',
        'langchain-community'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages, check=True)
        print("✅ Dependencies installed!")
    else:
        print("✅ All dependencies are installed!")

def start_backend():
    """Start the backend server."""
    print("🚀 Starting backend server...")
    
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    # Start backend with uvicorn
    backend_process = subprocess.Popen([
        sys.executable, '-m', 'uvicorn', 'backend.app:app', 
        '--host', '127.0.0.1', '--port', '9000', '--reload'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for backend to start
    time.sleep(5)
    
    if backend_process.poll() is None:
        print("✅ Backend server started successfully!")
        return backend_process
    else:
        stdout, stderr = backend_process.communicate()
        print(f"❌ Backend failed to start!")
        print(f"STDOUT: {stdout.decode()}")
        print(f"STDERR: {stderr.decode()}")
        return None

def start_frontend():
    """Start the frontend server."""
    print("🚀 Starting frontend server...")
    
    frontend_dir = Path(__file__).parent / "frontend"
    os.chdir(frontend_dir)
    
    # Start frontend with Python's built-in server
    frontend_process = subprocess.Popen([
        sys.executable, '-m', 'http.server', '8080'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for frontend to start
    time.sleep(3)
    
    if frontend_process.poll() is None:
        print("✅ Frontend server started successfully!")
        return frontend_process
    else:
        stdout, stderr = frontend_process.communicate()
        print(f"❌ Frontend failed to start!")
        print(f"STDOUT: {stdout.decode()}")
        print(f"STDERR: {stderr.decode()}")
        return None

def check_servers():
    """Check if both servers are running."""
    print("🔍 Checking server status...")
    
    import requests
    
    # Check backend
    try:
        response = requests.get("http://127.0.0.1:9000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is responding!")
        else:
            print(f"⚠️ Backend responded with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend server not responding: {e}")
        return False
    
    # Check frontend
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend server is responding!")
        else:
            print(f"⚠️ Frontend responded with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend server not responding: {e}")
        return False
    
    return True

def main():
    """Main startup function."""
    print("🌟 Starting TamraBot...")
    print("=" * 50)
    
    # Kill existing processes
    kill_existing_processes()
    
    # Check dependencies
    check_dependencies()
    
    # Start servers
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend. Exiting.")
        return
    
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend. Exiting.")
        backend_process.kill()
        return
    
    # Check if servers are running
    time.sleep(2)
    if check_servers():
        print("\n" + "=" * 50)
        print("🎉 TamraBot is running successfully!")
        print("🌐 Frontend: http://localhost:8080")
        print("🔧 Backend API: http://127.0.0.1:9000")
        print("=" * 50)
        print("Press Ctrl+C to stop the servers")
        print("=" * 50)
        
        try:
            # Keep the script running
            while True:
                time.sleep(1)
                # Check if processes are still running
                if backend_process.poll() is not None:
                    print("❌ Backend server stopped unexpectedly!")
                    break
                if frontend_process.poll() is not None:
                    print("❌ Frontend server stopped unexpectedly!")
                    break
        except KeyboardInterrupt:
            print("\n🛑 Stopping TamraBot...")
        finally:
            # Cleanup
            if backend_process.poll() is None:
                backend_process.kill()
            if frontend_process.poll() is None:
                frontend_process.kill()
            print("✅ TamraBot stopped successfully!")
    else:
        print("❌ Servers are not responding properly.")
        backend_process.kill()
        frontend_process.kill()

if __name__ == "__main__":
    main()