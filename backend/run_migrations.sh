#!/bin/bash

echo "🔄 Running Database Migrations..."
echo ""

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run migrations using Alembic
alembic upgrade head

echo ""
echo "✅ Migrations completed!" 