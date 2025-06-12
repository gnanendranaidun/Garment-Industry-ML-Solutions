# Garment ML Dashboard

A comprehensive web-based dashboard for garment manufacturing analytics, quality control, and machine learning predictions.

## Features

- **Production Monitoring**
  - Real-time production metrics
  - Production trends visualization
  - Efficiency analysis
  - Line balancing insights

- **Quality Control**
  - Quality metrics tracking
  - Defect analysis
  - Quality trends visualization
  - Defect type distribution

- **Machine Learning Predictions**
  - Production optimization
  - Quality prediction
  - Parameter optimization
  - Confidence scoring

- **Simulation Tools**
  - Production simulation
  - Quality impact analysis
  - Parameter optimization
  - What-if scenarios

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd garment-ml-dashboard
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following content:
```
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///garment_ml.db
```

## Running the Application

1. Start the Flask development server:
```bash
flask run
```

2. Access the dashboard at `http://localhost:5000`

## Project Structure

```
garment-ml-dashboard/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── static/              # Static files
│   ├── css/            # CSS styles
│   └── js/             # JavaScript files
├── templates/           # HTML templates
├── models/             # ML models
├── data/               # Data files
└── venv/               # Virtual environment
```

## Data Structure

The application uses the following data structure:

- Production Data:
  - Date
  - Product ID
  - Total Units
  - Good Units
  - Defect Units
  - Temperature
  - Pressure
  - Speed
  - Humidity

- Quality Metrics:
  - Quality Score
  - Defect Rate
  - Defect Types
  - Parameter Impact

## API Endpoints

- `/api/production-data` - Get production data
- `/api/quality-metrics` - Get quality metrics
- `/api/predictions` - Get ML predictions
- `/api/optimization` - Get optimal parameters
- `/api/quality-trends` - Get quality trends
- `/api/defect-analysis` - Get defect analysis

## Machine Learning Models

The application uses the following ML models:

1. Production Model:
   - Predicts production output based on parameters
   - Uses historical data for training
   - Provides confidence scores

2. Quality Model:
   - Predicts quality metrics
   - Identifies parameter impacts
   - Suggests optimizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the development team.

## Acknowledgments

- Flask framework
- Plotly for visualizations
- scikit-learn for ML models
- Bootstrap for UI components 