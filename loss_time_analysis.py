import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import os

class LossTimeAnalyzer:
    def __init__(self, file_path):
        """Initialize the LossTimeAnalyzer with the Excel file path."""
        self.file_path = file_path
        self.data = None
        self.sheets = None
        self.loss_data = {}
        self.models = {}
        self.scalers = {}
        
    def load_data(self):
        """Load data from all sheets in the Excel file."""
        try:
            # Get all sheet names
            xl = pd.ExcelFile(self.file_path)
            self.sheets = xl.sheet_names
            print(f"Available sheets: {self.sheets}")
            
            # Load data from each sheet
            for sheet_name in self.sheets:
                if sheet_name not in ['Sheet3']:  # Skip summary sheet
                    df = pd.read_excel(self.file_path, sheet_name=sheet_name)
                    self.loss_data[sheet_name] = df
                    print(f"\nLoaded {sheet_name} with {len(df)} rows and {len(df.columns)} columns")
            
            return self.loss_data
        except Exception as e:
            print(f"Error loading data from {self.file_path}: {e}")
            return {}

    def prepare_features(self, target_column):
        """Prepare features for ML models"""
        df = self.data.copy()
        
        # Create time-based features
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek
            df['hour'] = df['date'].dt.hour
        
        # Create lag features for time series analysis
        if target_column in df.columns:
            for lag in [1, 2, 3, 7]:
                df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        # Drop rows with NaN values from lag features
        df = df.dropna()
        
        X = df.drop(columns=[target_column, 'date'] if 'date' in df.columns else [target_column])
        y = df[target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target_column] = scaler
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def train_model(self, X_train, y_train, target_column):
        """Train random forest model for loss time prediction"""
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models[target_column] = model
        return model
    
    def evaluate_model(self, X_test, y_test, target_column):
        """Evaluate model performance"""
        model = self.models[target_column]
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel evaluation for {target_column}:")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        return {'MSE': mse, 'R2': r2}
    
    def plot_feature_importance(self, target_column, feature_names):
        """Plot feature importance"""
        model = self.models[target_column]
        importances = model.feature_importances_
        
        # Create DataFrame for better visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Feature Importance for {target_column} Prediction')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{target_column}.png')
        plt.close()
    
    def analyze_loss_patterns(self):
        """Analyze loss time patterns across all sheets."""
        try:
            results = {}
            for sheet_name, df in self.loss_data.items():
                print(f"\nAnalyzing {sheet_name}...")
                
                # Clean and prepare data
                df = df.dropna(subset=['Line ', 'Date of Gap', 'Act'])
                
                # Convert Act column to numeric
                df['Act'] = pd.to_numeric(df['Act'], errors='coerce')
                
                # Group by line and calculate statistics
                line_stats = df.groupby('Line ')['Act'].agg(['count', 'sum', 'mean', 'std']).round(2)
                
                # Group by reason category and calculate statistics
                reason_stats = df.groupby('Unnamed: 8')['Act'].agg(['count', 'sum', 'mean', 'std']).round(2)
                
                results[sheet_name] = {
                    'line_statistics': line_stats,
                    'reason_statistics': reason_stats
                }
                
                print(f"\nLine statistics for {sheet_name}:")
                print(line_stats)
                print(f"\nReason statistics for {sheet_name}:")
                print(reason_stats)
            
            return results
        except Exception as e:
            print(f"Error analyzing loss patterns: {str(e)}")
            return None

    def identify_critical_issues(self):
        """Identify critical issues based on loss time data."""
        try:
            results = {}
            for sheet_name, df in self.loss_data.items():
                print(f"\nIdentifying critical issues in {sheet_name}...")
                
                # Clean and prepare data
                df = df.dropna(subset=['Line ', 'Date of Gap', 'Act'])
                df['Act'] = pd.to_numeric(df['Act'], errors='coerce')
                
                # Calculate total loss time by reason
                reason_totals = df.groupby('Unnamed: 8')['Act'].sum().sort_values(ascending=False)
                
                # Identify critical issues (top 3 reasons)
                critical_issues = reason_totals.head(3)
                
                # Calculate impact percentage
                total_loss = reason_totals.sum()
                impact_percentage = (critical_issues / total_loss * 100).round(2)
                
                results[sheet_name] = {
                    'critical_issues': critical_issues,
                    'impact_percentage': impact_percentage
                }
                
                print(f"\nCritical issues in {sheet_name}:")
                for reason, time in critical_issues.items():
                    impact = impact_percentage[reason]
                    print(f"- {reason}: {time:.2f} minutes ({impact}% of total loss time)")
            
            return results
        except Exception as e:
            print(f"Error identifying critical issues: {str(e)}")
            return None

    def generate_recommendations(self):
        """Generate recommendations based on loss time analysis."""
        try:
            recommendations = {}
            for sheet_name, df in self.loss_data.items():
                print(f"\nGenerating recommendations for {sheet_name}...")
                
                # Clean and prepare data
                df = df.dropna(subset=['Line ', 'Date of Gap', 'Act'])
                df['Act'] = pd.to_numeric(df['Act'], errors='coerce')
                
                # Group by reason and calculate statistics
                reason_stats = df.groupby('Unnamed: 8').agg({
                    'Act': ['count', 'sum', 'mean'],
                    'Line ': 'nunique'
                }).round(2)
                
                # Generate recommendations
                sheet_recommendations = []
                
                for reason, stats in reason_stats.iterrows():
                    total_time = stats[('Act', 'sum')]
                    avg_time = stats[('Act', 'mean')]
                    frequency = stats[('Act', 'count')]
                    affected_lines = stats[('Line ', 'nunique')]
                    
                    if total_time > 100:  # Significant total loss time
                        sheet_recommendations.append(
                            f"High total loss time for '{reason}': {total_time:.2f} minutes across {affected_lines} lines. "
                            f"Average duration: {avg_time:.2f} minutes, Frequency: {frequency} times"
                        )
                    elif frequency > 5:  # High frequency
                        sheet_recommendations.append(
                            f"Frequent issue '{reason}': {frequency} occurrences with average duration of {avg_time:.2f} minutes"
                        )
                
                recommendations[sheet_name] = sheet_recommendations
                
                print(f"\nRecommendations for {sheet_name}:")
                for rec in sheet_recommendations:
                    print(f"- {rec}")
            
            return recommendations
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return None

    def save_report(self, output_dir='reports'):
        """Save analysis results to a report."""
        try:
            # Create reports directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(output_dir, f'loss_time_analysis_report_{timestamp}.txt')
            
            with open(report_file, 'w') as f:
                f.write("Loss Time Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                
                # Write analysis for each sheet
                for sheet_name in self.loss_data.keys():
                    f.write(f"\nAnalysis for {sheet_name}\n")
                    f.write("-" * 30 + "\n")
                    
                    df = self.loss_data[sheet_name]
                    df = df.dropna(subset=['Line ', 'Date of Gap', 'Act'])
                    df['Act'] = pd.to_numeric(df['Act'], errors='coerce')
                    
                    # Write line statistics
                    f.write("\nLine Statistics:\n")
                    line_stats = df.groupby('Line ')['Act'].agg(['count', 'sum', 'mean', 'std']).round(2)
                    f.write(str(line_stats))
                    
                    # Write reason statistics
                    f.write("\n\nReason Statistics:\n")
                    reason_stats = df.groupby('Unnamed: 8')['Act'].agg(['count', 'sum', 'mean', 'std']).round(2)
                    f.write(str(reason_stats))
                    
                    # Write critical issues
                    f.write("\n\nCritical Issues:\n")
                    reason_totals = df.groupby('Unnamed: 8')['Act'].sum().sort_values(ascending=False)
                    critical_issues = reason_totals.head(3)
                    total_loss = reason_totals.sum()
                    
                    for reason, time in critical_issues.items():
                        impact = (time / total_loss * 100).round(2)
                        f.write(f"- {reason}: {time:.2f} minutes ({impact}% of total loss time)\n")
                    
                    f.write("\n" + "=" * 50 + "\n")
            
            print(f"\nReport saved to: {report_file}")
            return report_file
        except Exception as e:
            print(f"Error saving report: {str(e)}")
            return None

def main():
    # Initialize the analyzer
    analyzer = LossTimeAnalyzer('CCL loss time_.xlsx')
    
    # Load data
    if not analyzer.load_data():
        return
    
    # Analyze loss patterns
    print("\nAnalyzing loss patterns...")
    analysis_results = analyzer.analyze_loss_patterns()
    
    # Identify critical issues
    print("\nIdentifying critical issues...")
    critical_issues = analyzer.identify_critical_issues()
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    recommendations = analyzer.generate_recommendations()
    
    # Save report
    print("\nSaving report...")
    report_file = analyzer.save_report()
    
    if report_file:
        print(f"\nAnalysis complete. Report saved to: {report_file}")

if __name__ == "__main__":
    main() 