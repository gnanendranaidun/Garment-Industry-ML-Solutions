import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime

class SalesForecaster:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.scalers = {}
        
    def load_data(self):
        """Load and preprocess sales data"""
        self.data = pd.read_excel(self.data_path)
        print("\nAvailable columns in the dataset:")
        print(self.data.columns.tolist())
        
        # Convert date column to datetime if it exists
        date_columns = [col for col in self.data.columns if 'date' in col.lower()]
        if date_columns:
            print("\nFound potential date columns:", date_columns)
            date_col = date_columns[0]  # Use the first date column found
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            print(f"Using '{date_col}' as the date column")
        else:
            print("\nNo date column found. Please ensure your data has a date column.")
        
        return self.data
    
    def prepare_prophet_data(self, target_column, date_column):
        """Prepare data for Prophet model"""
        if target_column not in self.data.columns:
            print(f"\nError: Target column '{target_column}' not found in the dataset.")
            return None
        
        if date_column not in self.data.columns:
            print(f"\nError: Date column '{date_column}' not found in the dataset.")
            return None
        
        prophet_data = self.data[[date_column, target_column]].copy()
        prophet_data.columns = ['ds', 'y']
        return prophet_data
    
    def train_prophet(self, data, target_column, date_column):
        """Train Prophet model"""
        prophet_data = self.prepare_prophet_data(target_column, date_column)
        if prophet_data is None:
            return None
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        model.fit(prophet_data)
        self.models[f'prophet_{target_column}'] = model
        return model
    
    def prepare_ml_features(self, target_column, date_column):
        """Prepare features for ML models"""
        if target_column not in self.data.columns:
            print(f"\nError: Target column '{target_column}' not found in the dataset.")
            return None, None, None, None
        
        if date_column not in self.data.columns:
            print(f"\nError: Date column '{date_column}' not found in the dataset.")
            return None, None, None, None
        
        # Create time-based features
        df = self.data.copy()
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['dayofweek'] = df[date_column].dt.dayofweek
        
        X = df.drop(columns=[target_column, date_column])
        y = df[target_column]
        
        # Split the data
        train_size = int(len(df) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target_column] = scaler
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_ml_models(self, X_train, y_train, target_column):
        """Train ML models"""
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        self.models[f'rf_{target_column}'] = rf_model
    
    def evaluate_models(self, X_test, y_test, target_column, date_column):
        """Evaluate model performance"""
        results = {}
        
        # Evaluate Prophet
        if f'prophet_{target_column}' in self.models:
            prophet_model = self.models[f'prophet_{target_column}']
            future = prophet_model.make_future_dataframe(periods=len(X_test))
            forecast = prophet_model.predict(future)
            prophet_predictions = forecast['yhat'][-len(X_test):]
            
            mse = mean_squared_error(y_test, prophet_predictions)
            r2 = r2_score(y_test, prophet_predictions)
            results['prophet'] = {'MSE': mse, 'R2': r2}
        
        # Evaluate Random Forest
        if f'rf_{target_column}' in self.models:
            rf_model = self.models[f'rf_{target_column}']
            rf_predictions = rf_model.predict(X_test)
            
            mse = mean_squared_error(y_test, rf_predictions)
            r2 = r2_score(y_test, rf_predictions)
            results['random_forest'] = {'MSE': mse, 'R2': r2}
        
        return results
    
    def plot_forecast(self, target_column, date_column, forecast_periods=30):
        """Plot forecast results"""
        if f'prophet_{target_column}' in self.models:
            prophet_model = self.models[f'prophet_{target_column}']
            future = prophet_model.make_future_dataframe(periods=forecast_periods)
            forecast = prophet_model.predict(future)
            
            plt.figure(figsize=(12, 6))
            plt.plot(self.data[date_column], self.data[target_column], label='Actual')
            plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
            plt.fill_between(forecast['ds'], 
                           forecast['yhat_lower'], 
                           forecast['yhat_upper'], 
                           color='gray', alpha=0.2)
            plt.title(f'Sales Forecast - {target_column}')
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'sales_forecast_{target_column}.png')
            plt.close()
    
    def save_models(self):
        """Save trained models"""
        for name, model in self.models.items():
            if 'prophet' in name:
                model.save(f'{name}_model.json')
            else:
                joblib.dump(model, f'{name}_model.joblib')
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{name}_scaler.joblib')
    
    def predict_sales(self, new_data, target_column, date_column):
        """Make predictions on new data"""
        predictions = {}
        
        # Prophet predictions
        if f'prophet_{target_column}' in self.models:
            prophet_model = self.models[f'prophet_{target_column}']
            future = prophet_model.make_future_dataframe(periods=len(new_data))
            forecast = prophet_model.predict(future)
            predictions['prophet'] = forecast['yhat'][-len(new_data):]
        
        # Random Forest predictions
        if f'rf_{target_column}' in self.models:
            rf_model = self.models[f'rf_{target_column}']
            scaled_data = self.scalers[target_column].transform(new_data)
            predictions['random_forest'] = rf_model.predict(scaled_data)
        
        return predictions

def main():
    # Initialize the forecaster
    forecaster = SalesForecaster('Stores - Data sets for AI training program.xlsx')
    
    # Load and prepare data
    data = forecaster.load_data()
    
    # Ask user to select target column
    print("\nPlease select a target column for forecasting:")
    target_column = input("Enter target column name: ")
    
    # Ask user to select date column
    print("\nPlease select a date column:")
    date_column = input("Enter date column name: ")
    
    # Train Prophet model
    prophet_model = forecaster.train_prophet(data, target_column, date_column)
    if prophet_model is None:
        return
    
    # Prepare features and train ML models
    result = forecaster.prepare_ml_features(target_column, date_column)
    if result is None:
        return
    
    X_train, X_test, y_train, y_test = result
    forecaster.train_ml_models(X_train, y_train, target_column)
    
    # Evaluate models
    results = forecaster.evaluate_models(X_test, y_test, target_column, date_column)
    print(f"\nModel evaluation results for {target_column}:")
    for model_name, metrics in results.items():
        print(f"{model_name}: MSE={metrics['MSE']:.4f}, R2={metrics['R2']:.4f}")
    
    # Plot forecast
    forecaster.plot_forecast(target_column, date_column)
    
    # Save models
    forecaster.save_models()

if __name__ == "__main__":
    main() 