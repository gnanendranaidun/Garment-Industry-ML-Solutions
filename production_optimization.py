import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class ProductionOptimizer:
    def __init__(self, file_path):
        """Initialize the ProductionOptimizer with the Excel file path."""
        self.file_path = file_path
        self.data = None
        self.sheets = None
        self.operation_data = {}
        
    def load_data(self):
        """Load data from all sheets in the Excel file."""
        try:
            self.operation_data = {}
            xls = pd.ExcelFile(self.file_path)
            for sheet_name in xls.sheet_names:
                df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None) # Read without header initially
                
                # Try to find the header row dynamically
                header_row = None
                for i in range(min(10, len(df_raw))): # Check first 10 rows
                    row_values = df_raw.iloc[i].astype(str).str.lower()
                    if any(col in row_values.values for col in ['operation', 'time (min)', 'token no']):
                        header_row = i
                        break
                
                if header_row is not None:
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row)
                else:
                    # Fallback if header is not found within first 10 rows, assume first row is header
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
                
                # Drop columns that are entirely NaN after reading with the correct header
                df = df.dropna(axis=1, how='all')
                
                # Convert all column names to lowercase and strip whitespace for robust matching
                df.columns = df.columns.astype(str).str.strip().str.lower()
                
                # Standardize column names using a comprehensive mapping
                column_mapping = {
                    'operation': 'Operation',
                    'time (min)': 'Operation tack time',
                    'operation tack time': 'Operation tack time', # Handle if already correctly read
                    'req.tack time': 'Req.Tack time',
                    'required tack time': 'Req.Tack time', # Handle variations
                    'token no': 'Token No.',
                    'token no.': 'Token No.'
                }
                df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

                # Ensure critical columns exist, even if with NaN
                for col in ['Operation', 'Operation tack time', 'Req.Tack time', 'Token No.']:
                    if col not in df.columns:
                        df[col] = pd.NA

                # Ensure relevant columns are numeric, coercing errors
                df['Operation tack time'] = pd.to_numeric(df['Operation tack time'], errors='coerce')
                df['Req.Tack time'] = pd.to_numeric(df['Req.Tack time'], errors='coerce')
                # Also convert other potentially numeric columns
                if 'Token No.' in df.columns:
                    df['Token No.'] = pd.to_numeric(df['Token No.'], errors='coerce')

                self.operation_data[sheet_name] = df
                print(f"\nLoaded {sheet_name} with {len(df)} rows and {len(df.columns)} columns")
                print(f"Final Columns in {sheet_name}: {df.columns.tolist()}") # Diagnostic print
            return self.operation_data
        except Exception as e:
            print(f"Error loading data from {self.file_path}: {e}")
            return {}

    def analyze_operations(self):
        """Analyze operations across all sheets."""
        try:
            results = {}
            for sheet_name, df in self.operation_data.items():
                print(f"\nAnalyzing {sheet_name}...")
                
                # Find timing columns (they contain numeric values)
                timing_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if not timing_cols:
                    print(f"No timing data found in {sheet_name}")
                    continue
                
                # Calculate basic statistics
                stats = df[timing_cols].describe()
                
                # Identify bottlenecks (operations with highest times)
                bottlenecks = df[timing_cols].max().sort_values(ascending=False)
                
                results[sheet_name] = {
                    'statistics': stats,
                    'bottlenecks': bottlenecks
                }
                
                print(f"\nStatistics for {sheet_name}:")
                print(stats)
                print(f"\nTop bottlenecks in {sheet_name}:")
                print(bottlenecks.head())
            
            return results
        except Exception as e:
            print(f"Error analyzing operations: {str(e)}")
            return None

    def optimize_line_balancing(self, target_cycle_time=None):
        """Optimize line balancing across all operations."""
        try:
            results = {}
            for sheet_name, df in self.operation_data.items():
                print(f"\nOptimizing {sheet_name}...")
                
                # Find timing columns
                timing_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if not timing_cols:
                    print(f"No timing data found in {sheet_name}")
                    continue
                
                # Calculate current cycle time
                current_cycle_time = df[timing_cols].max().max()
                
                if target_cycle_time is None:
                    target_cycle_time = current_cycle_time * 0.9  # 10% improvement target
                
                # Calculate efficiency
                efficiency = (df[timing_cols].sum().sum() / (len(timing_cols) * target_cycle_time)) * 100
                
                # Identify operations that need optimization
                optimization_needed = df[timing_cols].max() > target_cycle_time
                
                results[sheet_name] = {
                    'current_cycle_time': current_cycle_time,
                    'target_cycle_time': target_cycle_time,
                    'efficiency': efficiency,
                    'optimization_needed': optimization_needed
                }
                
                print(f"\nLine balancing analysis for {sheet_name}:")
                print(f"Current cycle time: {current_cycle_time:.2f}")
                print(f"Target cycle time: {target_cycle_time:.2f}")
                print(f"Line efficiency: {efficiency:.2f}%")
                print("\nOperations needing optimization:")
                print(optimization_needed[optimization_needed].index.tolist())
            
            return results
        except Exception as e:
            print(f"Error optimizing line balancing: {str(e)}")
            return None

    def generate_recommendations(self):
        """Generate optimization recommendations based on the analysis."""
        try:
            recommendations = {}
            for sheet_name, df in self.operation_data.items():
                print(f"\nGenerating recommendations for {sheet_name}...")
                
                # Find timing columns
                timing_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if not timing_cols:
                    print(f"No timing data found in {sheet_name}")
                    continue
                
                # Calculate average times
                avg_times = df[timing_cols].mean()
                
                # Identify slow operations
                slow_ops = avg_times[avg_times > avg_times.mean() + avg_times.std()]
                
                # Generate recommendations
                sheet_recommendations = []
                
                for op, time in slow_ops.items():
                    if time > avg_times.mean() * 1.5:
                        sheet_recommendations.append(f"Operation {op} is significantly slower than average. Consider process improvement or additional resources.")
                    elif time > avg_times.mean() * 1.2:
                        sheet_recommendations.append(f"Operation {op} is moderately slower than average. Review for potential optimization.")
                
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
            report_file = os.path.join(output_dir, f'production_optimization_report_{timestamp}.txt')
            
            with open(report_file, 'w') as f:
                f.write("Production Optimization Report\n")
                f.write("=" * 50 + "\n\n")
                
                # Write analysis for each sheet
                for sheet_name in self.sheets:
                    f.write(f"\nAnalysis for {sheet_name}\n")
                    f.write("-" * 30 + "\n")
                    
                    # Get timing columns
                    timing_cols = self.operation_data[sheet_name].select_dtypes(include=[np.number]).columns.tolist()
                    
                    if timing_cols:
                        # Write statistics
                        f.write("\nBasic Statistics:\n")
                        f.write(str(self.operation_data[sheet_name][timing_cols].describe()))
                        
                        # Write bottlenecks
                        f.write("\n\nTop Bottlenecks:\n")
                        bottlenecks = self.operation_data[sheet_name][timing_cols].max().sort_values(ascending=False)
                        f.write(str(bottlenecks.head()))
                        
                        # Write recommendations
                        f.write("\n\nRecommendations:\n")
                        avg_times = self.operation_data[sheet_name][timing_cols].mean()
                        slow_ops = avg_times[avg_times > avg_times.mean() + avg_times.std()]
                        
                        for op, time in slow_ops.items():
                            if time > avg_times.mean() * 1.5:
                                f.write(f"- Operation {op} is significantly slower than average. Consider process improvement or additional resources.\n")
                            elif time > avg_times.mean() * 1.2:
                                f.write(f"- Operation {op} is moderately slower than average. Review for potential optimization.\n")
                    
                    f.write("\n" + "=" * 50 + "\n")
            
            print(f"\nReport saved to: {report_file}")
            return report_file
        except Exception as e:
            print(f"Error saving report: {str(e)}")
            return None

def main():
    # Initialize the optimizer
    optimizer = ProductionOptimizer('Capacity study , Line balancing sheet.xlsx')
    
    # Load data
    print("\nLoading data...")
    data = optimizer.load_data()
    
    # Analyze operations
    print("\nAnalyzing operations...")
    analysis_results = optimizer.analyze_operations()
    
    # Optimize line balancing
    print("\nOptimizing line balancing...")
    optimization_results = optimizer.optimize_line_balancing()
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    recommendations = optimizer.generate_recommendations()
    
    # Save report
    print("\nSaving report...")
    report_file = optimizer.save_report()
    
    if report_file:
        print(f"\nAnalysis complete. Report saved to: {report_file}")

if __name__ == "__main__":
    main() 