#!/bin/bash

echo "🚀 Starting Ayuda Backend Server..."
echo "📍 Server will be available at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 