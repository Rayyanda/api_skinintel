#!/usr/bin/env bash
# Build script untuk Render.com

set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Initialize database
python setup.py

# Create necessary directories
mkdir -p uploads/pending uploads/reviewed