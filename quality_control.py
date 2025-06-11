import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
from datetime import datetime
import os

class QualityController:
    def __init__(self, file_path):
        """Initialize the QualityController with the Excel file path."""
        self.file_path = file_path
        self.data = None
        self.sheets = None
        self.quality_data = {}
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
                df = pd.read_excel(self.file_path, sheet_name=sheet_name)
                self.quality_data[sheet_name] = df
                print(f"\nLoaded {sheet_name} with {len(df)} rows and {len(df.columns)} columns")
            
            return self.quality_data
        except Exception as e:
            print(f"Error loading data from {self.file_path}: {e}")
            return {}

    def analyze_inspection_data(self):
        """Analyze inspection data from Inward & GRN sheet."""
        try:
            print("\nAnalyzing inspection data...")
            
            # Get Inward & GRN data
            df = self.quality_data['Inward & GRN']
            
            # Clean and prepare data
            df = df.dropna(subset=['Inspection Result', 'Fabric Quantity (Inspected)'])
            
            # Convert quantity to numeric
            df['Fabric Quantity (Inspected)'] = pd.to_numeric(df['Fabric Quantity (Inspected)'], errors='coerce')
            
            # Calculate inspection statistics
            inspection_stats = df.groupby('Inspection Result').agg({
                'Fabric Quantity (Inspected)': ['count', 'sum', 'mean'],
                'Fabric code': 'nunique'
            }).round(2)
            
            # Calculate pass rate
            total_inspected = df['Fabric Quantity (Inspected)'].sum()
            passed_quantity = df[df['Inspection Result'] == 'Pass']['Fabric Quantity (Inspected)'].sum()
            pass_rate = (passed_quantity / total_inspected * 100).round(2)
            
            print("\nInspection Statistics:")
            print(inspection_stats)
            print(f"\nOverall Pass Rate: {pass_rate}%")
            
            return {
                'inspection_stats': inspection_stats,
                'pass_rate': pass_rate
            }
        except Exception as e:
            print(f"Error analyzing inspection data: {str(e)}")
            return None

    def analyze_fra_data(self):
        """Analyze FRA (Fabric Rejection Analysis) data."""
        try:
            print("\nAnalyzing FRA data...")
            
            # Get FDR & FRA tracker data
            df = self.quality_data['FDR & FRA tracker']
            
            # Clean and prepare data
            df = df.dropna(subset=['FRA Qty', 'Status \n(Clear / EC / RTS / Pending)'])
            
            # Convert quantity to numeric
            df['FRA Qty'] = pd.to_numeric(df['FRA Qty'], errors='coerce')
            
            # Calculate FRA statistics by status
            fra_stats = df.groupby('Status \n(Clear / EC / RTS / Pending)').agg({
                'FRA Qty': ['count', 'sum', 'mean'],
                'Fabric Code': 'nunique',
                'Total Days \n(Taken for Closure)': 'mean'
            }).round(2)
            
            # Calculate average closure time
            avg_closure_time = df['Total Days \n(Taken for Closure)'].mean()
            
            print("\nFRA Statistics:")
            print(fra_stats)
            print(f"\nAverage Closure Time: {avg_closure_time:.2f} days")
            
            return {
                'fra_stats': fra_stats,
                'avg_closure_time': avg_closure_time
            }
        except Exception as e:
            print(f"Error analyzing FRA data: {str(e)}")
            return None

    def identify_quality_issues(self):
        """Identify major quality issues from the data."""
        try:
            print("\nIdentifying quality issues...")
            
            # Get FDR & FRA tracker data
            df = self.quality_data['FDR & FRA tracker']
            
            # Clean and prepare data
            df = df.dropna(subset=['FRA Reason', 'FRA Qty'])
            df['FRA Qty'] = pd.to_numeric(df['FRA Qty'], errors='coerce')
            
            # Group by reason and calculate statistics
            issue_stats = df.groupby('FRA Reason').agg({
                'FRA Qty': ['count', 'sum', 'mean'],
                'Fabric Code': 'nunique',
                'Total Days \n(Taken for Closure)': 'mean'
            }).round(2)
            
            # Identify top issues
            top_issues = issue_stats[('FRA Qty', 'sum')].sort_values(ascending=False).head(5)
            
            print("\nTop Quality Issues:")
            for reason, qty in top_issues.items():
                print(f"- {reason}: {qty:.2f} units")
            
            return {
                'issue_stats': issue_stats,
                'top_issues': top_issues
            }
        except Exception as e:
            print(f"Error identifying quality issues: {str(e)}")
            return None

    def generate_recommendations(self):
        """Generate quality improvement recommendations."""
        try:
            print("\nGenerating recommendations...")
            
            # Get FDR & FRA tracker data
            df = self.quality_data['FDR & FRA tracker']
            
            # Clean and prepare data
            df = df.dropna(subset=['FRA Reason', 'FRA Qty', 'Supplier Name'])
            df['FRA Qty'] = pd.to_numeric(df['FRA Qty'], errors='coerce')
            
            # Group by supplier and reason
            supplier_issues = df.groupby(['Supplier Name', 'FRA Reason']).agg({
                'FRA Qty': ['count', 'sum'],
                'Total Days \n(Taken for Closure)': 'mean'
            }).round(2)
            
            # Generate recommendations
            recommendations = []
            
            # Analyze supplier performance
            supplier_stats = df.groupby('Supplier Name').agg({
                'FRA Qty': 'sum',
                'Fabric Code': 'nunique'
            })
            
            for supplier, stats in supplier_stats.iterrows():
                if stats['FRA Qty'] > 100:  # Significant quality issues
                    recommendations.append(
                        f"Supplier '{supplier}' has significant quality issues: {stats['FRA Qty']:.2f} units rejected across {stats['Fabric Code']} fabric codes"
                    )
            
            # Analyze common issues
            common_issues = df.groupby('FRA Reason')['FRA Qty'].sum().sort_values(ascending=False)
            for reason, qty in common_issues.head(3).items():
                recommendations.append(
                    f"Common issue '{reason}': {qty:.2f} units affected. Consider implementing preventive measures."
                )
            
            print("\nRecommendations:")
            for rec in recommendations:
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
            report_file = os.path.join(output_dir, f'quality_control_report_{timestamp}.txt')
            
            with open(report_file, 'w') as f:
                f.write("Quality Control Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                
                # Write inspection analysis
                f.write("Inspection Analysis\n")
                f.write("-" * 30 + "\n")
                inspection_results = self.analyze_inspection_data()
                if inspection_results:
                    f.write("\nInspection Statistics:\n")
                    f.write(str(inspection_results['inspection_stats']))
                    f.write(f"\n\nOverall Pass Rate: {inspection_results['pass_rate']}%\n")
                
                # Write FRA analysis
                f.write("\n\nFRA Analysis\n")
                f.write("-" * 30 + "\n")
                fra_results = self.analyze_fra_data()
                if fra_results:
                    f.write("\nFRA Statistics:\n")
                    f.write(str(fra_results['fra_stats']))
                    f.write(f"\n\nAverage Closure Time: {fra_results['avg_closure_time']:.2f} days\n")
                
                # Write quality issues
                f.write("\n\nQuality Issues Analysis\n")
                f.write("-" * 30 + "\n")
                issues_results = self.identify_quality_issues()
                if issues_results:
                    f.write("\nTop Quality Issues:\n")
                    for reason, qty in issues_results['top_issues'].items():
                        f.write(f"- {reason}: {qty:.2f} units\n")
                
                # Write recommendations
                f.write("\n\nRecommendations\n")
                f.write("-" * 30 + "\n")
                recommendations = self.generate_recommendations()
                if recommendations:
                    for rec in recommendations:
                        f.write(f"- {rec}\n")
                
                f.write("\n" + "=" * 50 + "\n")
            
            print(f"\nReport saved to: {report_file}")
            return report_file
        except Exception as e:
            print(f"Error saving report: {str(e)}")
            return None

    def prepare_features(self, target_column=None):
        """Prepare features for ML models"""
        if target_column:
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[target_column] = scaler
            return X_train_scaled, X_test_scaled, y_train, y_test
        else:
            # For anomaly detection, use all features
            X = self.data.copy()
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            self.scalers['anomaly'] = scaler
            return X_scaled
    
    def train_anomaly_detector(self, X):
        """Train isolation forest for anomaly detection"""
        model = IsolationForest(
            contamination=0.1,  # Adjust based on expected anomaly rate
            random_state=42
        )
        model.fit(X)
        self.models['anomaly_detector'] = model
        return model
    
    def train_defect_classifier(self, X_train, y_train):
        """Train random forest classifier for defect prediction"""
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['defect_classifier'] = model
        return model
    
    def detect_anomalies(self, X):
        """Detect anomalies in the data"""
        model = self.models['anomaly_detector']
        predictions = model.predict(X)
        # Convert predictions: -1 for anomalies, 1 for normal
        return predictions == -1
    
    def predict_defects(self, X):
        """Predict defects using the classifier"""
        model = self.models['defect_classifier']
        return model.predict(X)
    
    def evaluate_defect_classifier(self, X_test, y_test):
        """Evaluate defect classifier performance"""
        model = self.models['defect_classifier']
        y_pred = model.predict(X_test)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    def plot_feature_importance(self):
        """Plot feature importance for the defect classifier"""
        model = self.models['defect_classifier']
        importances = model.feature_importances_
        feature_names = self.data.drop(columns=[self.target_column]).columns
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=feature_names)
        plt.title('Feature Importance for Defect Prediction')
        plt.tight_layout()
        plt.savefig('feature_importance_defects.png')
        plt.close()
    
    def save_models(self):
        """Save trained models"""
        for name, model in self.models.items():
            joblib.dump(model, f'{name}_model.joblib')
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{name}_scaler.joblib')
    
    def analyze_quality_metrics(self):
        """Analyze and visualize quality metrics"""
        # Example metrics (modify based on your actual data)
        metrics = {
            'defect_rate': self.data['defects'].mean(),
            'quality_score': self.data['quality_score'].mean(),
            'inspection_rate': self.data['inspections'].mean()
        }
        
        # Plot quality metrics
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Quality Metrics Overview')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('quality_metrics.png')
        plt.close()
        
        return metrics

def main():
    # Initialize the controller
    controller = QualityController('Stores - Data sets for AI training program.xlsx')
    
    # Load data
    if not controller.load_data():
        return
    
    # Analyze inspection data
    print("\nAnalyzing inspection data...")
    inspection_results = controller.analyze_inspection_data()
    
    # Analyze FRA data
    print("\nAnalyzing FRA data...")
    fra_results = controller.analyze_fra_data()
    
    # Identify quality issues
    print("\nIdentifying quality issues...")
    issues_results = controller.identify_quality_issues()
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    recommendations = controller.generate_recommendations()
    
    # Save report
    print("\nSaving report...")
    report_file = controller.save_report()
    
    if report_file:
        print(f"\nAnalysis complete. Report saved to: {report_file}")

if __name__ == "__main__":
    main() 