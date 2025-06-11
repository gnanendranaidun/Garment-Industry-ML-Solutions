import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import os

class ProductAnalyzer:
    def __init__(self, file_path):
        """Initialize the ProductAnalyzer with the Excel file path."""
        self.file_path = file_path
        self.product_data = {}
        
    def load_data(self):
        """Load data from all sheets in the Excel file."""
        try:
            xls = pd.ExcelFile(self.file_path)
            for sheet_name in xls.sheet_names:
                # Assuming header is in the first row (index 0) for Quadrant data. Adjust if 'Unnamed' columns persist.
                df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
                # Drop columns that are entirely NaN after reading
                df = df.dropna(axis=1, how='all')
                self.product_data[sheet_name] = df
                print(f"\nLoaded {sheet_name} with {len(df)} rows and {len(df.columns)} columns")
            return self.product_data
        except Exception as e:
            print(f"Error loading data from {self.file_path}: {e}")
            return {} # Ensure an empty dictionary is returned on error

    def analyze_performance(self):
        """Analyze operator performance from Competency Matrix sheet."""
        try:
            print("\nAnalyzing operator performance...")
            
            # Get Competency Matrix data
            df = self.product_data['Competency Matrix']
            
            # Clean and prepare data
            df = df.dropna(subset=['SMV', 'Target', 'production', 'Performance %'])
            
            # Convert numeric columns
            numeric_cols = ['SMV', 'Target', 'production', 'Performance %']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate section-wise statistics
            section_stats = df.groupby('Sections').agg({
                'SMV': ['mean', 'std'],
                'Target': ['mean', 'std'],
                'production': ['mean', 'std'],
                'Performance %': ['mean', 'std', 'count']
            }).round(2)
            
            # Calculate overall statistics
            overall_stats = df[numeric_cols].describe().round(2)
            
            print("\nSection-wise Statistics:")
            print(section_stats)
            print("\nOverall Statistics:")
            print(overall_stats)
            
            return {
                'section_stats': section_stats,
                'overall_stats': overall_stats
            }
        except Exception as e:
            print(f"Error analyzing performance: {str(e)}")
            return None

    def analyze_quadrants(self):
        """Analyze performance quadrants."""
        try:
            print("\nAnalyzing performance quadrants...")
            
            # Get Competency Matrix data
            df = self.product_data['Competency Matrix']
            
            # Clean and prepare data
            df = df.dropna(subset=['Quadrant', 'Performance %'])
            df['Performance %'] = pd.to_numeric(df['Performance %'], errors='coerce')
            
            # Calculate quadrant statistics
            quadrant_stats = df.groupby('Quadrant').agg({
                'Performance %': ['count', 'mean', 'std'],
                'Sections': 'nunique',
                'Present Operations': 'nunique'
            }).round(2)
            
            # Calculate quadrant distribution
            quadrant_dist = df['Quadrant'].value_counts().sort_index()
            
            print("\nQuadrant Statistics:")
            print(quadrant_stats)
            print("\nQuadrant Distribution:")
            print(quadrant_dist)
            
            return {
                'quadrant_stats': quadrant_stats,
                'quadrant_dist': quadrant_dist
            }
        except Exception as e:
            print(f"Error analyzing quadrants: {str(e)}")
            return None

    def identify_improvement_areas(self):
        """Identify areas needing improvement."""
        try:
            print("\nIdentifying improvement areas...")
            
            # Get Competency Matrix data
            df = self.product_data['Competency Matrix']
            
            # Clean and prepare data
            df = df.dropna(subset=['Sections', 'Present Operations', 'Performance %'])
            df['Performance %'] = pd.to_numeric(df['Performance %'], errors='coerce')
            
            # Identify low-performing sections
            section_performance = df.groupby('Sections')['Performance %'].mean().sort_values()
            low_performing_sections = section_performance[section_performance < 80]
            
            # Identify low-performing operations
            operation_performance = df.groupby('Present Operations')['Performance %'].mean().sort_values()
            low_performing_operations = operation_performance[operation_performance < 80]
            
            print("\nLow-performing Sections (Performance < 80%):")
            for section, perf in low_performing_sections.items():
                print(f"- {section}: {perf:.2f}%")
            
            print("\nLow-performing Operations (Performance < 80%):")
            for operation, perf in low_performing_operations.items():
                print(f"- {operation}: {perf:.2f}%")
            
            return {
                'low_performing_sections': low_performing_sections,
                'low_performing_operations': low_performing_operations
            }
        except Exception as e:
            print(f"Error identifying improvement areas: {str(e)}")
            return None

    def generate_recommendations(self):
        """Generate improvement recommendations."""
        try:
            print("\nGenerating recommendations...")
            
            # Get Competency Matrix data
            df = self.product_data['Competency Matrix']
            
            # Clean and prepare data
            df = df.dropna(subset=['Sections', 'Present Operations', 'Performance %', 'SMV', 'Target'])
            df['Performance %'] = pd.to_numeric(df['Performance %'], errors='coerce')
            df['SMV'] = pd.to_numeric(df['SMV'], errors='coerce')
            df['Target'] = pd.to_numeric(df['Target'], errors='coerce')
            
            recommendations = []
            
            # Analyze section performance
            section_stats = df.groupby('Sections').agg({
                'Performance %': ['mean', 'std', 'count'],
                'SMV': 'mean',
                'Target': 'mean'
            })
            
            for section, stats in section_stats.iterrows():
                mean_perf = stats[('Performance %', 'mean')]
                std_perf = stats[('Performance %', 'std')]
                count = stats[('Performance %', 'count')]
                
                if mean_perf < 80:
                    recommendations.append(
                        f"Section '{section}' has low average performance ({mean_perf:.2f}%) with {count} operators. "
                        f"Consider additional training or process optimization."
                    )
                elif std_perf > 15:
                    recommendations.append(
                        f"Section '{section}' shows high performance variation (std: {std_perf:.2f}%). "
                        f"Consider standardizing work methods and sharing best practices."
                    )
            
            # Analyze operation performance
            operation_stats = df.groupby('Present Operations').agg({
                'Performance %': ['mean', 'std', 'count'],
                'SMV': 'mean',
                'Target': 'mean'
            })
            
            for operation, stats in operation_stats.iterrows():
                mean_perf = stats[('Performance %', 'mean')]
                std_perf = stats[('Performance %', 'std')]
                count = stats[('Performance %', 'count')]
                
                if mean_perf < 80 and count > 2:
                    recommendations.append(
                        f"Operation '{operation}' has consistently low performance ({mean_perf:.2f}%) across {count} operators. "
                        f"Review work method and consider process improvement."
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
            report_file = os.path.join(output_dir, f'product_analysis_report_{timestamp}.txt')
            
            with open(report_file, 'w') as f:
                f.write("Product Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                
                # Write performance analysis
                f.write("Performance Analysis\n")
                f.write("-" * 30 + "\n")
                performance_results = self.analyze_performance()
                if performance_results:
                    f.write("\nSection-wise Statistics:\n")
                    f.write(str(performance_results['section_stats']))
                    f.write("\n\nOverall Statistics:\n")
                    f.write(str(performance_results['overall_stats']))
                
                # Write quadrant analysis
                f.write("\n\nQuadrant Analysis\n")
                f.write("-" * 30 + "\n")
                quadrant_results = self.analyze_quadrants()
                if quadrant_results:
                    f.write("\nQuadrant Statistics:\n")
                    f.write(str(quadrant_results['quadrant_stats']))
                    f.write("\n\nQuadrant Distribution:\n")
                    f.write(str(quadrant_results['quadrant_dist']))
                
                # Write improvement areas
                f.write("\n\nImprovement Areas\n")
                f.write("-" * 30 + "\n")
                improvement_results = self.identify_improvement_areas()
                if improvement_results:
                    f.write("\nLow-performing Sections:\n")
                    for section, perf in improvement_results['low_performing_sections'].items():
                        f.write(f"- {section}: {perf:.2f}%\n")
                    f.write("\nLow-performing Operations:\n")
                    for operation, perf in improvement_results['low_performing_operations'].items():
                        f.write(f"- {operation}: {perf:.2f}%\n")
                
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

def main():
    # Initialize the analyzer
    analyzer = ProductAnalyzer('Quadrant data - AI.xlsx')
    
    # Load data
    if not analyzer.load_data():
        return
    
    # Analyze performance
    print("\nAnalyzing performance...")
    performance_results = analyzer.analyze_performance()
    
    # Analyze quadrants
    print("\nAnalyzing quadrants...")
    quadrant_results = analyzer.analyze_quadrants()
    
    # Identify improvement areas
    print("\nIdentifying improvement areas...")
    improvement_results = analyzer.identify_improvement_areas()
    
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