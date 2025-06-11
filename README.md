# Garment Industry ML Solutions

This repository contains machine learning solutions for various aspects of the garment industry, including production optimization, sales forecasting, and quality control.

## Dataset Overview

1. **Production Line Data** (`Capacity study, Line balancing sheet.xlsx`)
   - Production capacity metrics
   - Line balancing information
   - Worker efficiency data
   - Production timing data

2. **Loss Time Analysis** (`CCL loss time_.xlsx`)
   - Production downtime records
   - Machine breakdown data
   - Maintenance schedules
   - Efficiency loss metrics

3. **Store Performance** (`Stores - Data sets for AI training program.xlsx`)
   - Sales data
   - Inventory levels
   - Customer transactions
   - Store performance metrics

4. **Quadrant Analysis** (`Quadrant data - AI.xlsx`)
   - Product categorization
   - Performance metrics
   - Market positioning data

## ML Applications

### 1. Production Optimization
- **File**: `production_optimization.py`
- **Features**:
  - Line balancing prediction
  - Production capacity forecasting
  - Worker efficiency analysis
  - Bottleneck detection

### 2. Sales Forecasting
- **File**: `sales_forecasting.py`
- **Features**:
  - Store-level sales prediction
  - Product demand forecasting
  - Seasonal trend analysis
  - Inventory optimization

### 3. Quality Control
- **File**: `quality_control.py`
- **Features**:
  - Defect prediction
  - Quality metrics analysis
  - Production quality optimization
  - Anomaly detection

### 4. Loss Time Analysis
- **File**: `loss_time_analysis.py`
- **Features**:
  - Downtime prediction
  - Maintenance scheduling
  - Root cause analysis
  - Efficiency optimization

### 5. Product Analysis
- **File**: `product_analysis.py`
- **Features**:
  - Product categorization
  - Performance prediction
  - Market positioning
  - Trend analysis

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Each ML module can be run independently:

```bash
python production_optimization.py
python sales_forecasting.py
python quality_control.py
python loss_time_analysis.py
python product_analysis.py
```

## Model Performance

Each module includes:
- Model evaluation metrics
- Performance visualizations
- Prediction accuracy reports
- Optimization recommendations

## Data Preprocessing

- Data cleaning scripts
- Feature engineering
- Data normalization
- Missing value handling

## Output

Each module generates:
- Prediction results
- Performance metrics
- Visualization plots
- Optimization recommendations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 