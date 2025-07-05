#!/bin/bash

echo "🚀 Starting Ayuda Backend Server (Fast Mode)..."
echo "📍 Server will be available at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo "⚡ Fast mode: Skipping migrations, lazy loading enabled"
echo ""

# Set environment variables for fast startup
export RUN_MIGRATIONS_ON_STARTUP=false
export ENVIRONMENT=development

# Add current directory to Python path so app module can be found
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start server with optimized settings
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --log-level info 