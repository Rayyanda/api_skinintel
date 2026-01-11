#!/usr/bin/env bash
# Build script untuk Render.com

set -o errexit

echo "=== Starting build process ==="

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Initialize database
echo "Initializing database..."
python -c "import database; database.init_db()"

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads/pending uploads/reviewed model templates

echo "=== Build complete ==="