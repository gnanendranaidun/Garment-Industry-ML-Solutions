#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating necessary directories..."
mkdir -p data
mkdir -p models
mkdir -p static/css
mkdir -p static/js
mkdir -p templates

echo "Creating .env file..."
echo "SECRET_KEY=your-secret-key" > .env
echo "DATABASE_URL=sqlite:///garment_ml.db" >> .env

echo "Initializing database..."
flask db init
flask db migrate
flask db upgrade

echo "Setup complete! You can now run the application with 'flask run'" 