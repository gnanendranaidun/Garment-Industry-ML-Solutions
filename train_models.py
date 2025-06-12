import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_and_preprocess_data():
    """Load and preprocess the production data"""
    try:
        data_path = 'data/production_data.csv'
        if not os.path.exists(data_path):
            print(f"Error: {data_path} not found. Please run convert_excel_to_csv.py first.")
            return None, None, None, None

        data = pd.read_csv(data_path)
        
        # Ensure 'date' column is datetime type for potential time-series features
        data['date'] = pd.to_datetime(data['date'])

        # Select relevant features. Ensure these columns exist in production_data.csv
        features = ['temperature', 'pressure', 'speed', 'humidity']
        target_production = 'total_units'
        target_quality = 'quality_score'
        
        # Handle missing values by filling with mean (or more sophisticated methods)
        data = data.fillna(data.mean(numeric_only=True))

        # Check if all required columns exist after loading
        required_columns = features + [target_production, target_quality]
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            print(f"Error: Missing required columns in {data_path}: {missing}")
            return None, None, None, None
        
        return data, features, target_production, target_quality
    except Exception as e:
        print(f"Error loading and preprocessing data: {str(e)}")
        return None, None, None, None

def train_production_model(data, features, target):
    """Train the production prediction model"""
    try:
        # Prepare data
        X = data[features]
        y = data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Production Model Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"R2 Score: {r2:.2f}")
        
        return model, scaler
    except Exception as e:
        print(f"Error training production model: {str(e)}")
        return None, None

def train_quality_model(data, features, target):
    """Train the quality prediction model"""
    try:
        # Prepare data
        X = data[features]
        y = data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nQuality Model Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"R2 Score: {r2:.2f}")
        
        return model
    except Exception as e:
        print(f"Error training quality model: {str(e)}")
        return None

def save_models(production_model, quality_model, scaler):
    """Save the trained models and scaler"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save models
        joblib.dump(production_model, 'models/production_model.joblib')
        joblib.dump(quality_model, 'models/quality_model.joblib')
        joblib.dump(scaler, 'models/scaler.joblib')
        
        print("\nModels saved successfully!")
    except Exception as e:
        print(f"Error saving models: {str(e)}")

def main():
    print("Loading and preprocessing data...")
    data, features, target_production, target_quality = load_and_preprocess_data()
    
    if data is not None:
        print("\nTraining production model...")
        production_model, scaler = train_production_model(data, features, target_production)
        
        print("\nTraining quality model...")
        quality_model = train_quality_model(data, features, target_quality)
        
        if all([production_model, quality_model, scaler]):
            print("\nSaving models...")
            save_models(production_model, quality_model, scaler)
        else:
            print("\nError: One or more models failed to train properly.")
    else:
        print("\nError: Failed to load and preprocess data.")

if __name__ == "__main__":
    main() 