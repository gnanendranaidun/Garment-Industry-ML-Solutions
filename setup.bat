@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo Creating necessary directories...
mkdir data 2>nul
mkdir models 2>nul
mkdir static\css 2>nul
mkdir static\js 2>nul
mkdir templates 2>nul

echo Creating .env file...
echo SECRET_KEY=your-secret-key> .env
echo DATABASE_URL=sqlite:///garment_ml.db>> .env

echo Initializing database...
flask db init
flask db migrate
flask db upgrade

echo Setup complete! You can now run the application with 'flask run' 