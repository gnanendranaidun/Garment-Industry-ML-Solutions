# Garment ML Analysis - Technical Documentation

## 🔧 System Architecture

### Core Components
```
garment_ml_analysis.py          # Main analysis pipeline
├── GarmentMLAnalyzer          # Primary analysis class
├── Dataset Discovery          # Automatic file detection
├── Data Preprocessing         # Cleaning and transformation
├── EDA Pipeline              # Exploratory data analysis
├── ML Implementation         # Model training and evaluation
└── Report Generation         # HTML report creation
```

### Dependencies
```python
pandas>=2.0.0              # Data manipulation
numpy>=1.20.0              # Numerical computing
scikit-learn>=1.0.0        # Machine learning
matplotlib>=3.5.0          # Plotting
seaborn>=0.11.0           # Statistical visualization
plotly>=5.0.0             # Interactive plots
openpyxl>=3.0.0           # Excel file handling
```

## 📊 Data Processing Pipeline

### 1. Dataset Discovery
```python
def discover_datasets():
    # Automatically finds .xlsx and .csv files
    # Handles multi-sheet Excel files
    # Returns structured dataset catalog
```

### 2. Data Quality Assessment
```python
def analyze_dataset_structure():
    # Shape analysis (rows × columns)
    # Missing value detection
    # Duplicate identification
    # Data type inference
    # Statistical summaries
```

### 3. Preprocessing Pipeline
```python
# Missing Value Handling
numerical_cols: median imputation
categorical_cols: mode imputation
datetime_cols: feature extraction (year, month, day, dayofweek)

# Encoding Strategy
categorical_features:
    if unique_values <= 10: OneHotEncoder
    else: LabelEncoder

# Feature Scaling
StandardScaler for all numerical features
```

### 4. Target Variable Detection
```python
# Automatic target identification based on keywords:
garment_keywords = [
    'efficiency', 'productivity', 'quality', 'defect', 
    'time', 'cost', 'rating', 'score', 'performance', 
    'output', 'target', 'actual'
]

# Problem type classification:
if target_dtype == 'object' or unique_values <= 10:
    problem_type = 'classification'
else:
    problem_type = 'regression'
```

## 🤖 Machine Learning Implementation

### Model Selection Strategy
```python
# Classification Models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42)
}

# Regression Models  
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'SVR': SVR()
}
```

### Evaluation Metrics
```python
# Classification Metrics
- Accuracy Score
- Precision (weighted average)
- Recall (weighted average)  
- F1-Score (weighted average)

# Regression Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score (coefficient of determination)
```

### Cross-Validation Strategy
```python
# Train/Test Split
test_size = 0.2 if len(X) >= 10 else 0.1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Minimum Sample Requirements
if len(X) < 5:
    return "Insufficient data for ML"
```

## 📈 Exploratory Data Analysis

### Correlation Analysis
```python
# Numerical features correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Visualization: Heatmap with annotations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
```

### Distribution Analysis
```python
# For each numerical column:
1. Histogram with 30 bins
2. Box plot for outlier detection
3. Statistical summary (mean, std, quartiles)

# For categorical columns:
1. Value counts (top 5)
2. Unique value analysis
3. Distribution visualization
```

## 🎯 Model Performance Results

### Successful Models Summary
| Dataset | Problem Type | Best Model | Key Metric | Performance |
|---------|-------------|------------|------------|-------------|
| CCL Loss Time (Feb) | Classification | Random Forest | Accuracy | High |
| CCL Loss Time (Mar) | Classification | Random Forest | Accuracy | High |
| CCL Loss Time (Apr) | Classification | Random Forest | Accuracy | High |
| Competency Matrix | Regression | Random Forest | R² Score | Good |
| FDR & FRA Tracker | Regression | Linear Regression | R² Score | Good |
| Stock Entry | Classification | Random Forest | Accuracy | High |
| Material Outward | Regression | Linear Regression | R² Score | Good |

### Error Handling Patterns
```python
# Common error scenarios and solutions:
1. "DType promotion error" → Datetime feature extraction
2. "String to float conversion" → Enhanced preprocessing
3. "Insufficient classes" → Single-class detection
4. "NaN values in input" → Comprehensive imputation
5. "Empty train set" → Dynamic test size adjustment
```

## 🔍 Data Quality Issues Identified

### High Missing Value Rates
```
Stores - Inward & GRN: 65,291 missing values (42% missing rate)
Capacity Study datasets: Up to 90% missing in some sheets
Line balance line 3: 155,918 missing values (95% missing rate)
```

### Insufficient Sample Sizes
```
Dart seam: 1 valid sample
Main lining seam: 1 valid sample  
Shoulder seam: 1 valid sample
Body welting: 3 valid samples
Pocket iron: 3 valid samples
```

### Data Type Inconsistencies
```
Mixed datetime and numeric types in same columns
String values in numeric fields ("Required MP")
Integer column names causing attribute errors
Inconsistent categorical encoding
```

## 🚀 Deployment Recommendations

### Production Model Pipeline
```python
# 1. Data Ingestion
def load_new_data(file_path):
    # Validate format and structure
    # Apply same preprocessing pipeline
    # Handle new categorical values

# 2. Model Inference  
def predict(model, new_data):
    # Apply trained preprocessors
    # Generate predictions
    # Return confidence scores

# 3. Model Monitoring
def monitor_performance():
    # Track prediction accuracy
    # Detect data drift
    # Trigger retraining alerts
```

### Recommended Infrastructure
```yaml
# Docker Container Setup
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "garment_ml_analysis.py"]

# Environment Variables
DATA_PATH=/data/garment_datasets/
MODEL_PATH=/models/trained/
OUTPUT_PATH=/reports/generated/
LOG_LEVEL=INFO
```

### API Endpoint Design
```python
# FastAPI implementation example
@app.post("/predict/ccl_loss")
async def predict_ccl_loss(data: CCLData):
    # Load trained Random Forest model
    # Preprocess input data
    # Return prediction with confidence

@app.post("/predict/material_demand")  
async def predict_material_demand(data: MaterialData):
    # Load trained Linear Regression model
    # Generate demand forecast
    # Return prediction with intervals
```

## 📊 Performance Optimization

### Large Dataset Handling
```python
# For datasets >100K records:
1. Chunked processing for memory efficiency
2. Sampling strategies for model training
3. Parallel processing where applicable
4. Optimized data types (category vs object)
```

### Model Training Optimization
```python
# Random Forest optimization:
n_estimators=100  # Balance between performance and speed
max_depth=None    # Allow full tree growth
min_samples_split=2  # Default for good generalization
random_state=42   # Reproducible results

# Linear Regression optimization:
fit_intercept=True  # Include bias term
normalize=False     # Use StandardScaler instead
```

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

1. **Memory Errors with Large Datasets**
   ```python
   # Solution: Process in chunks
   chunk_size = 10000
   for chunk in pd.read_csv(file, chunksize=chunk_size):
       process_chunk(chunk)
   ```

2. **Mixed Data Type Errors**
   ```python
   # Solution: Explicit type conversion
   df[col] = pd.to_numeric(df[col], errors='coerce')
   ```

3. **Datetime Parsing Issues**
   ```python
   # Solution: Robust datetime handling
   df[col] = pd.to_datetime(df[col], errors='coerce')
   ```

4. **Model Training Failures**
   ```python
   # Solution: Comprehensive error handling
   try:
       model.fit(X_train, y_train)
   except Exception as e:
       log_error(f"Model {model_name} failed: {str(e)}")
   ```

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Maintainer**: Data Science Team
