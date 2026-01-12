#!/bin/bash

# Exit on error
set -e

echo "Starting deployment setup..."

# Check if node is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js (v18+) first."
    exit 1
fi

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 first."
    exit 1
fi

# 1. Build Frontend
echo "Building Frontend..."
cd ../frontend
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi
npm run build
echo "Frontend build complete."

# 2. Setup Backend
echo "Setting up Backend..."
cd ../backend

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install requirements
echo "Installing backend dependencies..."
pip install -r requirements.txt

# 3. Start Server
echo "Starting FastAPI server..."
# Go back to project root to ensure imports like 'from backend.routes' work
cd ..
# Run uvicorn (assuming venv is still active or using path)
# We need to make sure we use the venv's python/uvicorn
# Since we activated it in the previous step, it should be fine.
exec uvicorn backend.app:app --host 0.0.0.0 --port 8000
