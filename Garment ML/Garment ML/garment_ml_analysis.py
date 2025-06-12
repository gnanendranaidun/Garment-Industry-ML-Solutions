#!/usr/bin/env python3
"""
Comprehensive Machine Learning Analysis for Garment ML Project
==============================================================

This script performs a complete analysis of all datasets in the Garment ML project,
including data discovery, EDA, ML implementation, and reporting.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import os
import glob
from datetime import datetime
import json

class GarmentMLAnalyzer:
    """
    Comprehensive ML analyzer for garment industry datasets
    """
    
    def __init__(self, data_directory="./Datasets"):
        self.data_directory = data_directory
        self.datasets = {}
        self.analysis_results = {}
        self.ml_results = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def discover_datasets(self):
        """
        Discover and catalog all datasets in the project directory
        """
        print("🔍 Discovering datasets...")
        
        # Find all Excel files
        excel_files = glob.glob(os.path.join(self.data_directory, "*.xlsx"))
        csv_files = glob.glob(os.path.join(self.data_directory, "*.csv"))
        
        all_files = excel_files + csv_files
        
        print(f"Found {len(all_files)} dataset files:")
        for file in all_files:
            print(f"  📊 {os.path.basename(file)}")
            
        return all_files
    
    def load_datasets(self):
        """
        Load all discovered datasets
        """
        print("\n📥 Loading datasets...")
        
        files = self.discover_datasets()
        
        for file_path in files:
            file_name = os.path.basename(file_path)
            print(f"\nLoading: {file_name}")
            
            try:
                if file_path.endswith('.xlsx'):
                    # Try to read Excel file, handle multiple sheets
                    excel_file = pd.ExcelFile(file_path)
                    if len(excel_file.sheet_names) == 1:
                        df = pd.read_excel(file_path)
                        self.datasets[file_name] = df
                    else:
                        # Multiple sheets - load each as separate dataset
                        for sheet_name in excel_file.sheet_names:
                            df = pd.read_excel(file_path, sheet_name=sheet_name)
                            dataset_key = f"{file_name}_{sheet_name}"
                            self.datasets[dataset_key] = df
                            
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    self.datasets[file_name] = df
                    
                print(f"  ✅ Successfully loaded")
                
            except Exception as e:
                print(f"  ❌ Error loading {file_name}: {str(e)}")
                
        print(f"\n📊 Total datasets loaded: {len(self.datasets)}")
        return self.datasets
    
    def analyze_dataset_structure(self, df, dataset_name):
        """
        Analyze the structure and quality of a single dataset
        """
        analysis = {
            'name': dataset_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'sample_data': df.head(3).to_dict('records')
        }
        
        # Statistical summary for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            analysis['numerical_summary'] = df[numerical_cols].describe().to_dict()
            
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            analysis['categorical_summary'] = {}
            for col in categorical_cols:
                analysis['categorical_summary'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
                
        return analysis
    
    def perform_eda(self, df, dataset_name):
        """
        Perform comprehensive exploratory data analysis
        """
        print(f"\n🔬 Performing EDA for: {dataset_name}")
        
        # Create output directory for plots
        plot_dir = f"plots_{dataset_name.replace(' ', '_').replace('.', '_')}"
        os.makedirs(plot_dir, exist_ok=True)
        
        eda_results = {
            'correlations': {},
            'distributions': {},
            'insights': []
        }
        
        # Correlation analysis for numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            correlation_matrix = df[numerical_cols].corr()
            eda_results['correlations'] = correlation_matrix.to_dict()
            
            # Plot correlation heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title(f'Correlation Matrix - {dataset_name}')
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        # Distribution analysis
        for col in numerical_cols:
            if df[col].dtype in ['int64', 'float64']:
                plt.figure(figsize=(10, 6))
                
                # Create subplot for distribution
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Histogram
                ax1.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                ax1.set_title(f'Distribution of {col}')
                ax1.set_xlabel(col)
                ax1.set_ylabel('Frequency')
                
                # Box plot
                ax2.boxplot(df[col].dropna())
                ax2.set_title(f'Box Plot of {col}')
                ax2.set_ylabel(col)
                
                plt.tight_layout()
                plt.savefig(f'{plot_dir}/distribution_{col.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        return eda_results

    def implement_ml_pipeline(self, df, dataset_name):
        """
        Implement comprehensive ML pipeline based on data characteristics
        """
        print(f"\n🤖 Implementing ML pipeline for: {dataset_name}")

        ml_results = {
            'problem_type': None,
            'target_variable': None,
            'features_used': [],
            'preprocessing_steps': [],
            'models_tested': {},
            'best_model': None,
            'performance_metrics': {}
        }

        # Identify potential target variables and problem type
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Auto-detect problem type based on data characteristics
        target_candidates = []

        # Look for common target variable patterns in garment industry
        garment_keywords = ['efficiency', 'productivity', 'quality', 'defect', 'time', 'cost',
                          'rating', 'score', 'performance', 'output', 'target', 'actual']

        for col in df.columns:
            col_str = str(col).lower()
            if any(keyword in col_str for keyword in garment_keywords):
                target_candidates.append(col)

        # If no obvious targets, use the last numerical column or most varied categorical
        if not target_candidates:
            if len(numerical_cols) > 0:
                target_candidates = [numerical_cols[-1]]
            elif len(categorical_cols) > 0:
                # Choose categorical with reasonable number of unique values
                for col in categorical_cols:
                    unique_count = df[col].nunique()
                    if 2 <= unique_count <= 10:
                        target_candidates.append(col)
                        break

        if not target_candidates:
            print("  ⚠️ No suitable target variable found. Performing unsupervised learning.")
            return self.implement_unsupervised_learning(df, dataset_name)

        # Use the first target candidate
        target_col = target_candidates[0]
        ml_results['target_variable'] = target_col

        # Determine problem type
        if target_col in numerical_cols:
            ml_results['problem_type'] = 'regression'
        else:
            unique_values = df[target_col].nunique()
            if unique_values <= 10:
                ml_results['problem_type'] = 'classification'
            else:
                ml_results['problem_type'] = 'regression'  # Treat as regression if too many categories

        print(f"  🎯 Target variable: {target_col}")
        print(f"  📊 Problem type: {ml_results['problem_type']}")

        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Remove rows with missing target values
        valid_indices = ~y.isnull()
        X = X[valid_indices]
        y = y[valid_indices]

        if len(X) == 0:
            print("  ❌ No valid data after removing missing targets")
            return ml_results

        # Preprocessing
        ml_results['preprocessing_steps'].append("Removed missing target values")

        # Handle missing values in features
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].dtype in ['int64', 'float64']:
                    X[col].fillna(X[col].median(), inplace=True)
                    ml_results['preprocessing_steps'].append(f"Filled missing {col} with median")
                else:
                    X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)
                    ml_results['preprocessing_steps'].append(f"Filled missing {col} with mode")

        # Handle datetime columns
        datetime_features = X.select_dtypes(include=['datetime64']).columns
        for col in datetime_features:
            # Extract useful features from datetime
            if not X[col].isnull().all():
                X[f'{col}_year'] = X[col].dt.year
                X[f'{col}_month'] = X[col].dt.month
                X[f'{col}_day'] = X[col].dt.day
                X[f'{col}_dayofweek'] = X[col].dt.dayofweek
                ml_results['preprocessing_steps'].append(f"Extracted datetime features from {col}")
            X = X.drop(col, axis=1)
            ml_results['preprocessing_steps'].append(f"Dropped original datetime column {col}")

        # Encode categorical variables
        categorical_features = X.select_dtypes(include=['object']).columns
        for col in categorical_features:
            if X[col].nunique() <= 10:  # One-hot encode if few categories
                dummies = pd.get_dummies(X[col], prefix=col)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                ml_results['preprocessing_steps'].append(f"One-hot encoded {col}")
            else:  # Label encode if many categories
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                ml_results['preprocessing_steps'].append(f"Label encoded {col}")

        # Encode target if classification
        if ml_results['problem_type'] == 'classification':
            if y.dtype == 'object' or y.apply(lambda x: isinstance(x, str)).any():
                # Convert all values to strings first to handle mixed types
                y = y.astype(str)
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
                ml_results['preprocessing_steps'].append("Label encoded target variable")

        ml_results['features_used'] = list(X.columns)

        # Ensure all features are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    # Try to convert to numeric
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    # Fill NaN values with median if possible, otherwise with 0
                    if X[col].notna().sum() > 0:
                        X[col].fillna(X[col].median(), inplace=True)
                    else:
                        X[col].fillna(0, inplace=True)
                    ml_results['preprocessing_steps'].append(f"Converted {col} to numeric")
                except:
                    # If conversion fails, drop the column
                    X = X.drop(col, axis=1)
                    ml_results['preprocessing_steps'].append(f"Dropped problematic column {col}")

        # Check if we have any features left
        if X.shape[1] == 0:
            print("    ❌ No valid features remaining after preprocessing")
            return ml_results

        # Check if we have enough data for train/test split
        if len(X) < 5:
            print(f"    ❌ Insufficient data for ML ({len(X)} samples). Need at least 5 samples.")
            return ml_results

        # Adjust test size for small datasets
        test_size = 0.2 if len(X) >= 10 else 0.1
        if len(X) < 10:
            test_size = 1.0 / len(X)  # Use 1 sample for test if very small dataset

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Scale features (only numeric columns)
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            ml_results['preprocessing_steps'].append("Standardized features")
        except Exception as e:
            print(f"    ⚠️ Scaling failed: {str(e)}, using original features")
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        # Implement models based on problem type
        if ml_results['problem_type'] == 'classification':
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42)
            }
        else:  # regression
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'SVR': SVR()
            }

        # Train and evaluate models
        best_score = -np.inf if ml_results['problem_type'] == 'classification' else np.inf
        best_model_name = None

        for model_name, model in models.items():
            print(f"    Training {model_name}...")

            try:
                # Train model
                if model_name in ['Logistic Regression', 'SVM', 'SVR', 'Linear Regression']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                # Evaluate model
                if ml_results['problem_type'] == 'classification':
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    ml_results['models_tested'][model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }

                    if accuracy > best_score:
                        best_score = accuracy
                        best_model_name = model_name

                else:  # regression
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)

                    ml_results['models_tested'][model_name] = {
                        'mse': mse,
                        'rmse': rmse,
                        'r2_score': r2
                    }

                    if mse < best_score:
                        best_score = mse
                        best_model_name = model_name

            except Exception as e:
                print(f"    ❌ Error training {model_name}: {str(e)}")
                ml_results['models_tested'][model_name] = {'error': str(e)}

        ml_results['best_model'] = best_model_name
        if best_model_name:
            ml_results['performance_metrics'] = ml_results['models_tested'][best_model_name]
            print(f"    🏆 Best model: {best_model_name}")

        return ml_results

    def implement_unsupervised_learning(self, df, dataset_name):
        """
        Implement unsupervised learning when no clear target is available
        """
        print(f"  🔍 Implementing unsupervised learning for: {dataset_name}")

        ml_results = {
            'problem_type': 'clustering',
            'target_variable': None,
            'features_used': [],
            'preprocessing_steps': [],
            'models_tested': {},
            'best_model': 'K-Means',
            'performance_metrics': {}
        }

        # Prepare data for clustering
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            print("    ❌ No numerical columns for clustering")
            return ml_results

        X = df[numerical_cols].copy()

        # Handle missing values
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
                ml_results['preprocessing_steps'].append(f"Filled missing {col} with median")

        ml_results['features_used'] = list(X.columns)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        ml_results['preprocessing_steps'].append("Standardized features")

        # Determine optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, min(11, len(X)//2))

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        # Find elbow point (simple method)
        optimal_k = k_range[0]
        if len(inertias) > 2:
            # Calculate rate of change
            rates = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
            optimal_k = k_range[rates.index(max(rates))]

        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(X_scaled, clusters)

        ml_results['models_tested']['K-Means'] = {
            'n_clusters': optimal_k,
            'silhouette_score': silhouette_avg,
            'inertia': kmeans.inertia_
        }
        ml_results['performance_metrics'] = ml_results['models_tested']['K-Means']

        # PCA for visualization
        if X_scaled.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # Plot clusters
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter)
            plt.title(f'K-Means Clustering Results - {dataset_name}')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

            plot_dir = f"plots_{dataset_name.replace(' ', '_').replace('.', '_')}"
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(f'{plot_dir}/clustering_results.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"    🎯 Optimal clusters: {optimal_k}")
        print(f"    📊 Silhouette score: {silhouette_avg:.3f}")

        return ml_results

    def generate_comprehensive_report(self):
        """
        Generate a comprehensive HTML report with all analysis results
        """
        print("\n📋 Generating comprehensive report...")

        # Create report directory
        report_dir = "garment_ml_report"
        os.makedirs(report_dir, exist_ok=True)

        # Generate HTML report
        html_content = self._create_html_report()

        with open(f"{report_dir}/comprehensive_analysis_report.html", "w", encoding='utf-8') as f:
            f.write(html_content)

        # Save analysis results as JSON
        with open(f"{report_dir}/analysis_results.json", "w") as f:
            json.dump(self.analysis_results, f, indent=2, default=str)

        with open(f"{report_dir}/ml_results.json", "w") as f:
            json.dump(self.ml_results, f, indent=2, default=str)

        print(f"  ✅ Report generated in: {report_dir}/")
        print(f"  📄 Main report: comprehensive_analysis_report.html")

        return f"{report_dir}/comprehensive_analysis_report.html"

    def _create_html_report(self):
        """
        Create comprehensive HTML report
        """
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Garment ML Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; margin-top: 30px; }}
                h3 {{ color: #7f8c8d; }}
                .dataset-section {{ background-color: #ecf0f1; padding: 20px; margin: 20px 0; border-radius: 8px; }}
                .metric-box {{ display: inline-block; background-color: #3498db; color: white; padding: 10px 15px; margin: 5px; border-radius: 5px; }}
                .insight-box {{ background-color: #e8f5e8; border-left: 4px solid #27ae60; padding: 15px; margin: 10px 0; }}
                .warning-box {{ background-color: #fdf2e9; border-left: 4px solid #e67e22; padding: 15px; margin: 10px 0; }}
                .error-box {{ background-color: #fadbd8; border-left: 4px solid #e74c3c; padding: 15px; margin: 10px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ border: 1px solid #bdc3c7; padding: 12px; text-align: left; }}
                th {{ background-color: #34495e; color: white; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .stat-card {{ background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
                .code-block {{ background-color: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; font-family: 'Courier New', monospace; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧥 Garment ML Analysis Report</h1>
                <p style="text-align: center; color: #7f8c8d; font-size: 18px;">
                    Comprehensive Machine Learning Analysis<br>
                    Generated on: {timestamp}
                </p>

                <h2>📊 Executive Summary</h2>
                <div class="summary-stats">
                    <div class="stat-card">
                        <div class="stat-value">{total_datasets}</div>
                        <div class="stat-label">Datasets Analyzed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{total_records}</div>
                        <div class="stat-label">Total Records</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{total_features}</div>
                        <div class="stat-label">Total Features</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{ml_models_tested}</div>
                        <div class="stat-label">ML Models Tested</div>
                    </div>
                </div>

                {dataset_sections}

                <h2>🎯 Key Insights and Recommendations</h2>
                {insights_section}

                <h2>📈 Overall Performance Summary</h2>
                {performance_summary}

                <h2>🔧 Technical Appendix</h2>
                <div class="code-block">
                    <strong>Analysis Configuration:</strong><br>
                    • Python Libraries: pandas, scikit-learn, matplotlib, seaborn, plotly<br>
                    • ML Algorithms: Random Forest, Logistic/Linear Regression, SVM<br>
                    • Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, R², RMSE<br>
                    • Cross-validation: 5-fold (where applicable)<br>
                    • Feature Scaling: StandardScaler<br>
                    • Missing Value Strategy: Median (numerical), Mode (categorical)
                </div>
            </div>
        </body>
        </html>
        """

        # Calculate summary statistics
        total_datasets = len(self.datasets)
        total_records = sum(df.shape[0] for df in self.datasets.values())
        total_features = sum(df.shape[1] for df in self.datasets.values())
        ml_models_tested = sum(len(result.get('models_tested', {})) for result in self.ml_results.values())

        # Generate dataset sections
        dataset_sections = ""
        for dataset_name, analysis in self.analysis_results.items():
            dataset_sections += self._create_dataset_section(dataset_name, analysis)

        # Generate insights section
        insights_section = self._generate_insights()

        # Generate performance summary
        performance_summary = self._generate_performance_summary()

        # Fill template
        html_content = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_datasets=total_datasets,
            total_records=f"{total_records:,}",
            total_features=f"{total_features:,}",
            ml_models_tested=ml_models_tested,
            dataset_sections=dataset_sections,
            insights_section=insights_section,
            performance_summary=performance_summary
        )

        return html_content

    def _create_dataset_section(self, dataset_name, analysis):
        """Create HTML section for individual dataset"""
        ml_result = self.ml_results.get(dataset_name, {})

        section = f"""
        <div class="dataset-section">
            <h3>📊 Dataset: {dataset_name}</h3>

            <h4>Data Structure</h4>
            <div class="metric-box">Shape: {analysis['shape'][0]} rows × {analysis['shape'][1]} columns</div>
            <div class="metric-box">Memory: {analysis['memory_usage'] / 1024:.1f} KB</div>
            <div class="metric-box">Duplicates: {analysis['duplicates']}</div>

            <h4>Data Quality Assessment</h4>
            <table>
                <tr><th>Column</th><th>Data Type</th><th>Missing Values</th><th>Missing %</th></tr>
        """

        for col in analysis['columns']:
            missing_count = analysis['missing_values'].get(col, 0)
            missing_pct = analysis['missing_percentage'].get(col, 0)
            dtype = str(analysis['dtypes'].get(col, 'unknown'))

            section += f"""
                <tr>
                    <td>{col}</td>
                    <td>{dtype}</td>
                    <td>{missing_count}</td>
                    <td>{missing_pct:.1f}%</td>
                </tr>
            """

        section += "</table>"

        # Add ML results if available
        if ml_result:
            section += f"""
            <h4>🤖 Machine Learning Results</h4>
            <div class="insight-box">
                <strong>Problem Type:</strong> {ml_result.get('problem_type', 'N/A')}<br>
                <strong>Target Variable:</strong> {ml_result.get('target_variable', 'N/A')}<br>
                <strong>Best Model:</strong> {ml_result.get('best_model', 'N/A')}<br>
                <strong>Features Used:</strong> {len(ml_result.get('features_used', []))}
            </div>
            """

            if ml_result.get('models_tested'):
                section += "<table><tr><th>Model</th><th>Performance Metrics</th></tr>"
                for model_name, metrics in ml_result['models_tested'].items():
                    metrics_str = ", ".join([f"{k}: {v:.3f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                                           for k, v in metrics.items()])
                    section += f"<tr><td>{model_name}</td><td>{metrics_str}</td></tr>"
                section += "</table>"

        section += "</div>"
        return section

    def _generate_insights(self):
        """Generate key insights from analysis"""
        insights = []

        # Data quality insights
        total_missing = 0
        total_cells = 0
        for analysis in self.analysis_results.values():
            total_missing += sum(analysis['missing_values'].values())
            total_cells += analysis['shape'][0] * analysis['shape'][1]

        missing_rate = (total_missing / total_cells * 100) if total_cells > 0 else 0

        if missing_rate > 10:
            insights.append(f"⚠️ High missing data rate ({missing_rate:.1f}%) - consider data collection improvements")
        elif missing_rate < 5:
            insights.append(f"✅ Good data quality with low missing rate ({missing_rate:.1f}%)")

        # ML performance insights
        classification_models = []
        regression_models = []

        for ml_result in self.ml_results.values():
            if ml_result.get('problem_type') == 'classification':
                best_model = ml_result.get('best_model')
                if best_model and ml_result.get('performance_metrics'):
                    accuracy = ml_result['performance_metrics'].get('accuracy', 0)
                    classification_models.append((best_model, accuracy))
            elif ml_result.get('problem_type') == 'regression':
                best_model = ml_result.get('best_model')
                if best_model and ml_result.get('performance_metrics'):
                    r2 = ml_result['performance_metrics'].get('r2_score', 0)
                    regression_models.append((best_model, r2))

        if classification_models:
            avg_accuracy = np.mean([acc for _, acc in classification_models])
            if avg_accuracy > 0.8:
                insights.append(f"🎯 Strong classification performance (avg accuracy: {avg_accuracy:.1%})")
            else:
                insights.append(f"📈 Classification models show room for improvement (avg accuracy: {avg_accuracy:.1%})")

        if regression_models:
            avg_r2 = np.mean([r2 for _, r2 in regression_models])
            if avg_r2 > 0.7:
                insights.append(f"📊 Good regression model fit (avg R²: {avg_r2:.3f})")
            else:
                insights.append(f"🔧 Regression models may need feature engineering (avg R²: {avg_r2:.3f})")

        # Feature insights
        total_features = sum(len(result.get('features_used', [])) for result in self.ml_results.values())
        if total_features > 0:
            insights.append(f"🔍 Total features utilized across all models: {total_features}")

        if not insights:
            insights.append("📋 Analysis completed successfully - review individual dataset sections for detailed findings")

        insights_html = ""
        for insight in insights:
            insights_html += f'<div class="insight-box">{insight}</div>'

        return insights_html

    def _generate_performance_summary(self):
        """Generate overall performance summary"""
        summary = "<table><tr><th>Dataset</th><th>Problem Type</th><th>Best Model</th><th>Key Metric</th><th>Value</th></tr>"

        for dataset_name, ml_result in self.ml_results.items():
            problem_type = ml_result.get('problem_type', 'N/A')
            best_model = ml_result.get('best_model', 'N/A')

            key_metric = 'N/A'
            metric_value = 'N/A'

            if ml_result.get('performance_metrics'):
                metrics = ml_result['performance_metrics']
                if problem_type == 'classification':
                    key_metric = 'Accuracy'
                    metric_value = f"{metrics.get('accuracy', 0):.3f}"
                elif problem_type == 'regression':
                    key_metric = 'R² Score'
                    metric_value = f"{metrics.get('r2_score', 0):.3f}"
                elif problem_type == 'clustering':
                    key_metric = 'Silhouette Score'
                    metric_value = f"{metrics.get('silhouette_score', 0):.3f}"

            summary += f"""
            <tr>
                <td>{dataset_name}</td>
                <td>{problem_type}</td>
                <td>{best_model}</td>
                <td>{key_metric}</td>
                <td>{metric_value}</td>
            </tr>
            """

        summary += "</table>"
        return summary

    def run_complete_analysis(self):
        """
        Run the complete ML analysis pipeline
        """
        print("🚀 Starting Comprehensive Garment ML Analysis")
        print("=" * 60)

        # Step 1: Load datasets
        self.load_datasets()

        if not self.datasets:
            print("❌ No datasets found to analyze")
            return

        # Step 2: Analyze each dataset
        for dataset_name, df in self.datasets.items():
            print(f"\n{'='*60}")
            print(f"Analyzing: {dataset_name}")
            print(f"{'='*60}")

            # Structural analysis
            analysis = self.analyze_dataset_structure(df, dataset_name)
            self.analysis_results[dataset_name] = analysis

            print(f"  📊 Shape: {analysis['shape']}")
            print(f"  🔍 Columns: {len(analysis['columns'])}")
            print(f"  ⚠️ Missing values: {sum(analysis['missing_values'].values())}")
            print(f"  🔄 Duplicates: {analysis['duplicates']}")

            # EDA
            eda_results = self.perform_eda(df, dataset_name)
            self.analysis_results[dataset_name]['eda'] = eda_results

            # ML Pipeline
            ml_results = self.implement_ml_pipeline(df, dataset_name)
            self.ml_results[dataset_name] = ml_results

        # Step 3: Generate comprehensive report
        report_path = self.generate_comprehensive_report()

        print(f"\n{'='*60}")
        print("🎉 Analysis Complete!")
        print(f"📄 Report available at: {report_path}")
        print(f"{'='*60}")

        return report_path


def main():
    """
    Main execution function
    """
    analyzer = GarmentMLAnalyzer()
    report_path = analyzer.run_complete_analysis()

    print(f"\n🎯 Next Steps:")
    print(f"1. Open the report: {report_path}")
    print(f"2. Review the analysis results and insights")
    print(f"3. Consider implementing the recommended optimizations")
    print(f"4. Run additional experiments based on findings")


if __name__ == "__main__":
    main()
