"""
Comprehensive Garment Industry ML Dashboard
A production-ready Streamlit application for garment industry stakeholders
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import datetime

# Page configuration
st.set_page_config(
    page_title="Garment Industry Analytics Dashboard",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #A23B72;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #f0f8ff;
        border-left: 5px solid #2E86AB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .recommendation-box {
        background-color: #f0fff0;
        border-left: 5px solid #32cd32;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff8dc;
        border-left: 5px solid #ffa500;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_garment_data():
    """Load and process all garment industry datasets"""
    csv_dir = Path('csv')
    data = {}

    # Check if CSV directory exists
    if not csv_dir.exists():
        st.warning("CSV directory not found. Please create a 'csv' folder and add your data files.")
        return {}

    try:
        # Load inventory and stores data
        stores_files = {
            'stock_entry': 'Stores_-_Data_sets_for_AI_training_program__Stock_Entry_&_Location.csv',
            'material_outward': 'Stores_-_Data_sets_for_AI_training_program__Material_outward.csv',
            'material_inhouse': 'Stores_-_Data_sets_for_AI_training_program__Material_Inhouse.csv',
            'fdr_fra': 'Stores_-_Data_sets_for_AI_training_program__FDR_&_FRA_tracker.csv',
            'inward_grn': 'Stores_-_Data_sets_for_AI_training_program__Inward_&_GRN.csv'
        }

        for key, filename in stores_files.items():
            file_path = csv_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    # Clean and convert Stock Qty to numeric
                    if 'Stock Qty' in df.columns:
                        df['Stock Qty'] = pd.to_numeric(df['Stock Qty'], errors='coerce').fillna(0)
                    data[key] = df
                except Exception as e:
                    st.warning(f"Could not load {filename}: {str(e)}")

        # Load capacity and production data
        capacity_files = [f for f in csv_dir.glob('Capacity_study_*') if f.suffix == '.csv']
        data['capacity'] = {}
        for file in capacity_files:
            try:
                # First, try reading normally
                df = pd.read_csv(file)

                # Check if first row contains "Unnamed" columns (indicating headers are in row 2)
                if df.columns[0].startswith('Unnamed'):
                    # Check if row 1 contains actual column names
                    if len(df) > 1 and not pd.isna(df.iloc[1, 0]):
                        # Use row 1 as headers and skip first two rows
                        df = pd.read_csv(file, header=1)
                        # Remove any completely empty rows
                        df = df.dropna(how='all')

                if not df.empty and len(df.columns) > 0:
                    # Clean column names
                    df.columns = df.columns.astype(str).str.strip()

                    # Remove rows that are mostly NaN or contain metadata
                    df = df.dropna(thresh=len(df.columns)//2)  # Keep rows with at least half non-NaN values

                    if not df.empty:
                        data['capacity'][file.stem] = df
            except Exception as e:
                continue

        # Load loss time and quality data
        loss_time_files = [f for f in csv_dir.glob('CCL_loss_time_*') if f.suffix == '.csv']
        if not loss_time_files:
            # Try alternative pattern
            loss_time_files = [f for f in csv_dir.glob('*loss*') if f.suffix == '.csv']

        data['loss_time'] = {}
        for file in loss_time_files:
            try:
                df = pd.read_csv(file)
                if not df.empty and len(df.columns) > 0:
                    # Clean column names
                    df.columns = df.columns.astype(str).str.strip()
                    # Convert numeric columns
                    for col in df.columns:
                        if 'actual' in col.lower() or 'target' in col.lower():
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    data['loss_time'][file.stem] = df
            except Exception as e:
                continue

        # Load competency and workforce data
        quadrant_files = [f for f in csv_dir.glob('Quadrant_data_*') if f.suffix == '.csv']
        if not quadrant_files:
            # Try alternative pattern
            quadrant_files = [f for f in csv_dir.glob('*Quadrant*') if f.suffix == '.csv']

        data['workforce'] = {}
        for file in quadrant_files:
            try:
                df = pd.read_csv(file)
                if not df.empty and len(df.columns) > 0:
                    # Clean column names
                    df.columns = df.columns.astype(str).str.strip()
                    # Clean performance data
                    if 'Performance %' in df.columns:
                        df['Performance %'] = pd.to_numeric(df['Performance %'], errors='coerce').fillna(0)
                    if 'Quadrant' in df.columns:
                        df['Quadrant'] = pd.to_numeric(df['Quadrant'], errors='coerce').fillna(4)
                    data['workforce'][file.stem] = df
            except Exception as e:
                continue

        # If no specific files found, try to load any CSV files
        if not any([data.get('capacity'), data.get('loss_time'), data.get('workforce')]):
            all_csv_files = list(csv_dir.glob('*.csv'))
            if all_csv_files:
                st.info(f"Found {len(all_csv_files)} CSV files. Loading available data...")
                for file in all_csv_files[:10]:  # Limit to first 10 files
                    try:
                        df = pd.read_csv(file)
                        if not df.empty:
                            # Categorize based on content
                            if any(col.lower() in ['stock', 'material', 'inventory'] for col in df.columns):
                                if 'inventory' not in data:
                                    data['inventory'] = {}
                                data['inventory'][file.stem] = df
                            elif any(col.lower() in ['performance', 'quadrant', 'operator'] for col in df.columns):
                                if 'workforce' not in data:
                                    data['workforce'] = {}
                                data['workforce'][file.stem] = df
                            elif any(col.lower() in ['capacity', 'production', 'line'] for col in df.columns):
                                if 'capacity' not in data:
                                    data['capacity'] = {}
                                data['capacity'][file.stem] = df
                            else:
                                if 'general' not in data:
                                    data['general'] = {}
                                data['general'][file.stem] = df
                    except Exception as e:
                        continue

        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}

class GarmentMLPredictor:
    """Machine Learning predictor for garment industry analytics"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_metrics = {}
    
    def train_production_efficiency_model(self, capacity_data):
        """Train model to predict production efficiency with enhanced data handling"""
        try:
            if not capacity_data:
                st.warning("No capacity data available for training production efficiency model")
                return False

            # Find suitable capacity dataset with proper structure
            best_dataset = None
            best_score = 0

            for name, df in capacity_data.items():
                if df is None or df.empty:
                    continue

                # Score dataset based on structure and content
                score = 0

                # Check for key production columns
                key_columns = ['SMV', 'Eff%', 'Capacity', 'CT', 'Cycle Time', 'TGT', 'Production']
                for col in df.columns:
                    col_str = str(col)
                    if any(key in col_str for key in key_columns):
                        score += 10

                # Prefer datasets with more rows
                score += min(len(df), 50)

                # Prefer datasets with more numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                score += len(numeric_cols) * 2

                if score > best_score:
                    best_score = score
                    best_dataset = (name, df)

            if not best_dataset:
                st.warning("No suitable capacity dataset found for production efficiency training")
                return False

            name, df = best_dataset
            st.info(f"Training production efficiency model using dataset: {name}")

            # Clean and prepare the data
            df_clean = df.copy()

            # Remove rows that are mostly empty or contain headers
            df_clean = df_clean.dropna(how='all')

            # Identify and use the most relevant columns for production efficiency
            feature_candidates = []
            target_candidates = []

            for col in df_clean.columns:
                col_str = str(col).lower()

                # Convert to numeric if possible
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

                # Skip columns with too many NaN values
                if df_clean[col].isna().sum() > len(df_clean) * 0.7:
                    continue

                # Identify potential features and targets based on garment industry terminology
                if any(keyword in col_str for keyword in ['smv', 'cycle time', 'ct', 'ct1', 'ct2', 'ct3', 'ct4', 'ct5']):
                    feature_candidates.append(col)
                elif any(keyword in col_str for keyword in ['tgt', 'target']):
                    feature_candidates.append(col)
                elif any(keyword in col_str for keyword in ['eff%', 'efficiency', 'capacity/hr', 'capacity']):
                    target_candidates.append(col)
                elif any(keyword in col_str for keyword in ['avg prodn', 'production']):
                    target_candidates.append(col)

            # If no specific columns found, use any numeric columns
            if not feature_candidates and not target_candidates:
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    feature_candidates = numeric_cols[:-1]
                    target_candidates = [numeric_cols[-1]]

            # Select best features and target
            if not feature_candidates or not target_candidates:
                st.warning("Could not identify suitable features and target columns for production efficiency model")
                return False

            # Use up to 3 best features
            features = feature_candidates[:3]
            target = target_candidates[0]

            # Prepare final dataset
            columns_to_use = features + [target]
            df_model = df_clean[columns_to_use].dropna()

            if len(df_model) < 5:
                st.warning(f"Insufficient clean data for training: only {len(df_model)} samples available")
                return False

            X = df_model[features]
            y = df_model[target]

            # Check for variance in target
            if y.std() == 0:
                st.warning("Target variable has no variance - cannot train model")
                return False

            # Remove outliers (values beyond 3 standard deviations)
            z_scores = np.abs((y - y.mean()) / y.std())
            mask = z_scores < 3
            X = X[mask]
            y = y[mask]

            if len(X) < 5:
                st.warning("Insufficient data after outlier removal")
                return False

            # Split data
            test_size = min(0.3, max(0.2, 1.0 / len(X)))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model with appropriate parameters
            n_estimators = min(100, max(10, len(X_train) // 2))
            max_depth = min(10, len(features) + 2)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                min_samples_split=max(2, len(X_train) // 10),
                min_samples_leaf=max(1, len(X_train) // 20)
            )
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 0
            mse = mean_squared_error(y_test, y_pred)

            # Store model and metadata
            self.models['production_efficiency'] = model
            self.scalers['production_efficiency'] = scaler
            self.model_metrics['production_efficiency'] = {
                'r2_score': max(0, r2),
                'mse': mse,
                'features': features,
                'target': target,
                'dataset': name,
                'n_samples': len(X),
                'feature_importance': dict(zip(features, model.feature_importances_))
            }

            st.success(f"✅ Production efficiency model trained successfully!")
            st.info(f"Model performance: R² = {r2:.3f}, MSE = {mse:.3f}")
            st.info(f"Features used: {', '.join(features)}")
            st.info(f"Target: {target}")

            return True

        except Exception as e:
            st.error(f"Error training production efficiency model: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            return False
    
    def train_quality_prediction_model(self, workforce_data):
        """Train model to predict quality based on workforce competency"""
        try:
            if not workforce_data:
                return False

            for name, df in workforce_data.items():
                if df is None or df.empty:
                    continue

                # Check for required columns
                has_performance = any('performance' in str(col).lower() for col in df.columns)
                has_quadrant = any('quadrant' in str(col).lower() for col in df.columns)

                if has_performance and has_quadrant:
                    # Prepare features for quality prediction
                    feature_cols = []
                    target_col = None

                    for col in df.columns:
                        col_lower = str(col).lower()
                        if any(keyword in col_lower for keyword in ['smv', 'target', 'production']):
                            if pd.api.types.is_numeric_dtype(df[col]):
                                feature_cols.append(col)
                        elif 'performance' in col_lower and pd.api.types.is_numeric_dtype(df[col]):
                            feature_cols.append(col)
                        elif 'quadrant' in col_lower:
                            target_col = col

                    if len(feature_cols) >= 1 and target_col is not None:
                        # Clean data
                        X = df[feature_cols].copy()
                        y = df[target_col].copy()

                        # Handle missing values
                        for col in feature_cols:
                            X[col] = pd.to_numeric(X[col], errors='coerce')
                            X[col] = X[col].fillna(X[col].median())

                        y = pd.to_numeric(y, errors='coerce')
                        y = y.fillna(y.mode()[0] if not y.mode().empty else 1)

                        # Remove rows with all NaN
                        valid_idx = ~(X.isna().all(axis=1) | y.isna())
                        X = X.loc[valid_idx]
                        y = y.loc[valid_idx]

                        if len(X) > 5 and len(y.unique()) > 1:
                            # Split data
                            test_size = min(0.3, max(0.1, len(X) // 10))
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=42, stratify=y if len(y.unique()) > 1 else None
                            )

                            # Scale features
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)

                            # Train classifier
                            model = RandomForestClassifier(
                                n_estimators=min(100, max(10, len(X_train))),
                                random_state=42,
                                max_depth=min(10, len(feature_cols) * 2)
                            )
                            model.fit(X_train_scaled, y_train)

                            # Evaluate
                            y_pred = model.predict(X_test_scaled)
                            accuracy = accuracy_score(y_test, y_pred) if len(y_test) > 0 else 0

                            # Store model
                            self.models['quality_prediction'] = model
                            self.scalers['quality_prediction'] = scaler
                            self.model_metrics['quality_prediction'] = {
                                'accuracy': accuracy,
                                'features': feature_cols,
                                'target': target_col,
                                'dataset': name,
                                'n_samples': len(X),
                                'n_classes': len(y.unique())
                            }

                            return True
            return False
        except Exception as e:
            st.warning(f"Could not train quality prediction model: {str(e)}")
            return False
    
    def predict_production_efficiency(self, feature_values):
        """Make production efficiency prediction"""
        if 'production_efficiency' not in self.models:
            return None

        try:
            # Validate input
            if not feature_values or len(feature_values) == 0:
                return None

            # Convert to numpy array and handle missing values
            feature_array = []
            for val in feature_values:
                try:
                    feature_array.append(float(val))
                except (ValueError, TypeError):
                    feature_array.append(0.0)

            X = np.array(feature_array).reshape(1, -1)

            # Check if we have the right number of features
            expected_features = len(self.model_metrics['production_efficiency']['features'])
            if X.shape[1] != expected_features:
                st.warning(f"Expected {expected_features} features, got {X.shape[1]}")
                return None

            X_scaled = self.scalers['production_efficiency'].transform(X)
            prediction = self.models['production_efficiency'].predict(X_scaled)[0]
            return float(prediction)
        except Exception as e:
            st.warning(f"Prediction error: {str(e)}")
            return None

    def predict_quality_quadrant(self, feature_values):
        """Make quality quadrant prediction"""
        if 'quality_prediction' not in self.models:
            return None

        try:
            # Validate input
            if not feature_values or len(feature_values) == 0:
                return None

            # Convert to numpy array and handle missing values
            feature_array = []
            for val in feature_values:
                try:
                    feature_array.append(float(val))
                except (ValueError, TypeError):
                    feature_array.append(0.0)

            X = np.array(feature_array).reshape(1, -1)

            # Check if we have the right number of features
            expected_features = len(self.model_metrics['quality_prediction']['features'])
            if X.shape[1] != expected_features:
                st.warning(f"Expected {expected_features} features, got {X.shape[1]}")
                return None

            X_scaled = self.scalers['quality_prediction'].transform(X)
            prediction = self.models['quality_prediction'].predict(X_scaled)[0]
            return int(prediction)
        except Exception as e:
            st.warning(f"Prediction error: {str(e)}")
            return None

def main():
    """Main application function"""

    try:
        # Header
        st.markdown('<h1 class="main-header">🏭 Garment Industry Analytics Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("**Empowering garment manufacturers with data-driven insights and machine learning predictions**")

        # Load data
        with st.spinner("Loading garment industry data..."):
            data = load_garment_data()

        if not data:
            st.error("Unable to load data. Please check if CSV files are available in the 'csv' directory.")
            st.info("Please ensure you have CSV files in a 'csv' folder in the same directory as this application.")
            return

        # Show data loading summary
        data_summary = []
        for category, items in data.items():
            if isinstance(items, dict):
                data_summary.append(f"✅ {category.title()}: {len(items)} datasets")
            else:
                data_summary.append(f"✅ {category.title()}: 1 dataset")

        with st.expander("📋 Data Loading Summary", expanded=False):
            for summary in data_summary:
                st.markdown(summary)

        # Initialize ML predictor
        ml_predictor = GarmentMLPredictor()

        # Sidebar navigation
        st.sidebar.title("📊 Navigation")
        page = st.sidebar.radio(
            "Select Analysis Section:",
            ["🏠 Executive Summary", "📈 Analysis Dashboard", "🤖 ML Predictions", "💡 Business Insights"]
        )

        # Route to appropriate page
        try:
            if page == "🏠 Executive Summary":
                show_executive_summary(data)
            elif page == "📈 Analysis Dashboard":
                show_analysis_dashboard(data)
            elif page == "🤖 ML Predictions":
                show_ml_predictions(data, ml_predictor)
            elif page == "💡 Business Insights":
                show_business_insights(data)
        except Exception as e:
            st.error(f"Error displaying page: {str(e)}")
            st.info("Please try refreshing the page or selecting a different section.")

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check your data files and try again.")

def show_executive_summary(data):
    """Display executive summary with key metrics and overview"""
    st.markdown('<h2 class="section-header">📊 Executive Summary</h2>', unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if 'stock_entry' in data:
            total_items = len(data['stock_entry'])
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_items:,}</h3>
                <p>Inventory Items</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        if 'workforce' in data:
            total_workers = 0
            for df in data['workforce'].values():
                if 'Operator Name' in df.columns:
                    total_workers += df['Operator Name'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_workers}</h3>
                <p>Active Workers</p>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        if 'capacity' in data:
            production_lines = len(data['capacity'])
            st.markdown(f"""
            <div class="metric-card">
                <h3>{production_lines}</h3>
                <p>Production Lines</p>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        if 'loss_time' in data:
            total_loss_hours = 0
            for df in data['loss_time'].values():
                if df is not None and not df.empty and 'Actual' in df.columns:
                    actual_values = pd.to_numeric(df['Actual'], errors='coerce').fillna(0)
                    total_loss_hours += actual_values.sum() / 60  # Convert to hours
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_loss_hours:,.0f}h</h3>
                <p>Total Loss Time</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Overview sections
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏭 Production Overview")
        if 'stock_entry' in data:
            df = data['stock_entry']

            # Material distribution
            if 'Material' in df.columns:
                material_counts = df['Material'].value_counts().head(10)
                fig = px.pie(
                    values=material_counts.values,
                    names=material_counts.index,
                    title="Top 10 Materials by Volume"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 👥 Workforce Performance")
        if 'workforce' in data:
            for _, df in data['workforce'].items():
                if df is not None and not df.empty and 'Performance %' in df.columns and 'Quadrant' in df.columns:
                    # Performance by quadrant
                    try:
                        perf_by_quadrant = df.groupby('Quadrant')['Performance %'].mean()

                        if not perf_by_quadrant.empty:
                            fig = px.bar(
                                x=perf_by_quadrant.index,
                                y=perf_by_quadrant.values,
                                title="Average Performance by Quadrant",
                                labels={'x': 'Quadrant', 'y': 'Performance %'}
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            break
                    except Exception as e:
                        st.warning(f"Could not create workforce performance chart: {str(e)}")
                        continue

    # Key insights
    st.markdown("### 🔍 Key Insights")

    insights = []

    # Inventory insights
    if 'stock_entry' in data:
        df = data['stock_entry']
        if df is not None and not df.empty and 'Stock Qty' in df.columns:
            stock_qty = pd.to_numeric(df['Stock Qty'], errors='coerce').fillna(0)
            low_stock_items = len(stock_qty[stock_qty < 100])
            insights.append(f"📦 {low_stock_items} items have low stock levels (< 100 units)")

    # Loss time insights
    if 'loss_time' in data:
        for df in data['loss_time'].values():
            if df is not None and not df.empty and 'Reason Category' in df.columns:
                reason_mode = df['Reason Category'].mode()
                if not reason_mode.empty:
                    top_reason = reason_mode.iloc[0]
                    insights.append(f"⚠️ Main cause of production loss: {top_reason}")
                    break

    # Performance insights
    if 'workforce' in data:
        for df in data['workforce'].values():
            if df is not None and not df.empty and 'Performance %' in df.columns:
                performance_values = pd.to_numeric(df['Performance %'], errors='coerce').dropna()
                if not performance_values.empty:
                    avg_performance = performance_values.mean()
                    insights.append(f"👥 Average workforce performance: {avg_performance:.1f}%")
                    break

    for insight in insights:
        st.markdown(f"""
        <div class="insight-box">
            {insight}
        </div>
        """, unsafe_allow_html=True)

def show_analysis_dashboard(data):
    """Display comprehensive data analysis dashboard"""
    st.markdown('<h2 class="section-header">📈 Analysis Dashboard</h2>', unsafe_allow_html=True)

    # Analysis type selector
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Inventory Analysis", "Production Capacity", "Quality Control", "Workforce Performance"]
    )

    if analysis_type == "Inventory Analysis":
        show_inventory_analysis(data)
    elif analysis_type == "Production Capacity":
        show_production_analysis(data)
    elif analysis_type == "Quality Control":
        show_quality_analysis(data)
    elif analysis_type == "Workforce Performance":
        show_workforce_analysis(data)

def show_inventory_analysis(data):
    """Show detailed inventory analysis"""
    st.markdown("### 📦 Inventory Analysis")

    if 'stock_entry' not in data:
        st.warning("No inventory data available")
        return

    df = data['stock_entry']

    col1, col2 = st.columns(2)

    with col1:
        # Stock quantity distribution
        if 'Stock Qty' in df.columns:
            fig = px.histogram(
                df, x='Stock Qty',
                title="Stock Quantity Distribution",
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Brand distribution
        if 'Brand' in df.columns:
            brand_counts = df['Brand'].value_counts()
            fig = px.pie(
                values=brand_counts.values,
                names=brand_counts.index,
                title="Inventory by Brand"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Stock level alerts
    if 'Stock Qty' in df.columns:
        st.markdown("### 🚨 Stock Level Alerts")

        low_stock = df[df['Stock Qty'] < 100]
        zero_stock = df[df['Stock Qty'] == 0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Low Stock Items", len(low_stock))
        with col2:
            st.metric("Out of Stock Items", len(zero_stock))
        with col3:
            st.metric("Total Stock Value", f"{df['Stock Qty'].sum():,}")

        if len(low_stock) > 0:
            st.markdown("**Items requiring immediate attention:**")
            st.dataframe(low_stock[['Material', 'Description', 'Stock Qty']].head(10))

def show_production_analysis(data):
    """Show production capacity analysis"""
    st.markdown("### 🏭 Production Capacity Analysis")

    if 'capacity' not in data or not data['capacity']:
        st.warning("No production capacity data available")
        return

    # Select capacity dataset
    capacity_file = st.selectbox("Select Production Line:", list(data['capacity'].keys()))
    df = data['capacity'][capacity_file]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Data Overview**")
        st.write(f"Records: {len(df)}")
        st.write(f"Columns: {len(df.columns)}")

        # Show sample data
        st.markdown("**Sample Data:**")
        st.dataframe(df.head())

    with col2:
        # Find numeric columns for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            selected_metric = st.selectbox("Select Metric to Analyze:", numeric_cols)

            if selected_metric:
                # Create visualization
                fig = px.histogram(
                    df, x=selected_metric,
                    title=f"Distribution of {selected_metric}",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show statistics
                st.markdown("**Statistics:**")
                st.write(f"Mean: {df[selected_metric].mean():.2f}")
                st.write(f"Median: {df[selected_metric].median():.2f}")
                st.write(f"Std Dev: {df[selected_metric].std():.2f}")

def show_quality_analysis(data):
    """Show quality control analysis"""
    st.markdown("### 🔍 Quality Control Analysis")

    if 'loss_time' not in data or not data['loss_time']:
        st.warning("No quality/loss time data available")
        return

    # Select loss time dataset
    loss_file = st.selectbox("Select Loss Time Report:", list(data['loss_time'].keys()))
    df = data['loss_time'][loss_file]

    col1, col2 = st.columns(2)

    with col1:
        # Loss time by category
        if 'Reason Category' in df.columns:
            reason_counts = df['Reason Category'].value_counts()
            fig = px.bar(
                x=reason_counts.values,
                y=reason_counts.index,
                orientation='h',
                title="Loss Time by Category"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Loss time trends
        if 'Actual' in df.columns:
            # Convert to hours for better readability
            df['Loss Hours'] = df['Actual'] / 60

            fig = px.histogram(
                df, x='Loss Hours',
                title="Loss Time Distribution (Hours)",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    if 'Actual' in df.columns:
        total_loss_minutes = df['Actual'].sum()
        total_loss_hours = total_loss_minutes / 60
        avg_loss_per_incident = df['Actual'].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Loss Time", f"{total_loss_hours:.1f} hours")
        with col2:
            st.metric("Total Incidents", len(df))
        with col3:
            st.metric("Avg Loss per Incident", f"{avg_loss_per_incident:.1f} min")

def show_workforce_analysis(data):
    """Show workforce performance analysis"""
    st.markdown("### 👥 Workforce Performance Analysis")

    if 'workforce' not in data or not data['workforce']:
        st.warning("No workforce data available")
        return

    # Select workforce dataset
    workforce_file = st.selectbox("Select Workforce Report:", list(data['workforce'].keys()))
    df = data['workforce'][workforce_file]

    if 'Performance %' in df.columns and 'Quadrant' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Performance distribution by quadrant
            fig = px.box(
                df, x='Quadrant', y='Performance %',
                title="Performance Distribution by Quadrant"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Section-wise performance
            if 'Sections' in df.columns:
                section_perf = df.groupby('Sections')['Performance %'].mean().sort_values(ascending=False)
                fig = px.bar(
                    x=section_perf.values,
                    y=section_perf.index,
                    orientation='h',
                    title="Average Performance by Section"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Performance insights
        st.markdown("### 📊 Performance Insights")

        try:
            performance_values = pd.to_numeric(df['Performance %'], errors='coerce').fillna(0)
            high_performers = performance_values[performance_values > 1.0]
            low_performers = performance_values[performance_values < 0.7]

            total_workers = len(performance_values)
            high_pct = (len(high_performers) / total_workers * 100) if total_workers > 0 else 0
            low_pct = (len(low_performers) / total_workers * 100) if total_workers > 0 else 0
            avg_performance = performance_values.mean() if not performance_values.empty else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("High Performers", f"{len(high_performers)} ({high_pct:.1f}%)")
            with col2:
                st.metric("Average Performance", f"{avg_performance:.2f}")
            with col3:
                st.metric("Low Performers", f"{len(low_performers)} ({low_pct:.1f}%)")
        except Exception as e:
            st.warning(f"Could not calculate performance insights: {str(e)}")

def show_ml_predictions(data, ml_predictor):
    """Show machine learning predictions interface"""
    st.markdown('<h2 class="section-header">🤖 Machine Learning Predictions</h2>', unsafe_allow_html=True)

    # Model Training Section
    st.subheader("🔧 Model Training & Status")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Production Efficiency Model")

        # Check if production model exists
        if 'production_efficiency' in ml_predictor.models:
            metrics = ml_predictor.model_metrics['production_efficiency']
            st.success("✅ Model trained and ready!")
            st.info(f"R² Score: {metrics['r2_score']:.3f}")
            st.info(f"Dataset: {metrics['dataset']}")
            st.info(f"Features: {', '.join(metrics['features'])}")
        else:
            st.warning("⚠️ Model not trained yet")

        # Training button
        if st.button("🔄 Train Production Model", key="train_prod"):
            if 'capacity' in data and data['capacity']:
                with st.spinner("Training production efficiency model..."):
                    success = ml_predictor.train_production_efficiency_model(data['capacity'])
                    if success:
                        st.rerun()  # Refresh to show updated status
                    else:
                        st.error("❌ Training failed - check data quality")
            else:
                st.error("❌ No capacity data available for training")

    with col2:
        st.markdown("### Quality Prediction Model")

        # Check if quality model exists
        if 'quality_prediction' in ml_predictor.models:
            metrics = ml_predictor.model_metrics['quality_prediction']
            st.success("✅ Model trained and ready!")
            st.info(f"Accuracy: {metrics['accuracy']:.3f}")
            st.info(f"Dataset: {metrics['dataset']}")
            st.info(f"Features: {', '.join(metrics['features'])}")
        else:
            st.warning("⚠️ Model not trained yet")

        # Training button
        if st.button("🔄 Train Quality Model", key="train_qual"):
            if 'workforce' in data and data['workforce']:
                with st.spinner("Training quality prediction model..."):
                    success = ml_predictor.train_quality_prediction_model(data['workforce'])
                    if success:
                        st.rerun()  # Refresh to show updated status
                    else:
                        st.error("❌ Training failed - check data quality")
            else:
                st.error("❌ No workforce data available for training")

    # Auto-train models if not already trained
    if not ml_predictor.models:
        st.info("🤖 No models trained yet. Use the buttons above to train models, or they will be trained automatically when you make predictions.")

        # Auto-train if data is available
        auto_train = st.checkbox("Auto-train models with available data", value=True)
        if auto_train:
            with st.spinner("Auto-training models with available data..."):
                trained_any = False

                # Try to train production efficiency model
                if 'capacity' in data and data['capacity']:
                    success = ml_predictor.train_production_efficiency_model(data['capacity'])
                    if success:
                        trained_any = True

                # Try to train quality prediction model
                if 'workforce' in data and data['workforce']:
                    success = ml_predictor.train_quality_prediction_model(data['workforce'])
                    if success:
                        trained_any = True

                if trained_any:
                    st.rerun()  # Refresh to show updated status

    st.divider()

    # Model selection and predictions
    available_models = list(ml_predictor.models.keys())

    if not available_models:
        st.warning("⚠️ No machine learning models are available.")
        st.info("Please train models using the buttons above, or check that you have the required data files:")
        st.markdown("- **Production Efficiency Model**: Requires capacity study data (CSV files with 'Capacity' in filename)")
        st.markdown("- **Quality Prediction Model**: Requires workforce/quadrant data (CSV files with 'Quadrant' in filename)")
        return

    st.subheader("🎯 Make Predictions")
    selected_model = st.selectbox("Select Prediction Model:", available_models)

    if selected_model == 'production_efficiency':
        show_production_prediction(ml_predictor)
    elif selected_model == 'quality_prediction':
        show_quality_prediction(ml_predictor)

def show_production_prediction(ml_predictor):
    """Show production efficiency prediction interface"""
    st.markdown("### 🏭 Production Efficiency Prediction")

    model_info = ml_predictor.model_metrics.get('production_efficiency', {})

    if model_info:
        st.markdown(f"""
        <div class="insight-box">
            <strong>Model Performance:</strong><br>
            • R² Score: {model_info.get('r2_score', 0):.3f}<br>
            • Mean Squared Error: {model_info.get('mse', 0):.3f}<br>
            • Features: {', '.join(model_info.get('features', []))}<br>
            • Target: {model_info.get('target', 'Unknown')}
        </div>
        """, unsafe_allow_html=True)

    # Input interface
    st.markdown("**Enter Production Parameters:**")

    features = model_info.get('features', ['Feature 1', 'Feature 2'])
    feature_values = []

    cols = st.columns(len(features))
    for i, feature in enumerate(features):
        with cols[i]:
            value = st.number_input(f"{feature}:", value=1.0, step=0.1, key=f"prod_{i}")
            feature_values.append(value)

    if st.button("🔮 Predict Production Efficiency", type="primary"):
        prediction = ml_predictor.predict_production_efficiency(feature_values)

        if prediction is not None:
            st.markdown(f"""
            <div class="recommendation-box">
                <h3>🎯 Prediction Result</h3>
                <p><strong>Predicted {model_info.get('target', 'Value')}: {prediction:.2f}</strong></p>
                <p>This prediction is based on the input parameters and historical production data.</p>
            </div>
            """, unsafe_allow_html=True)

            # Provide interpretation
            if prediction > 100:
                st.success("🟢 Excellent efficiency predicted! Production is likely to exceed targets.")
            elif prediction > 80:
                st.info("🟡 Good efficiency predicted. Production should meet most targets.")
            else:
                st.warning("🔴 Low efficiency predicted. Consider optimizing production parameters.")

def show_quality_prediction(ml_predictor):
    """Show quality prediction interface"""
    st.markdown("### 🔍 Quality Quadrant Prediction")

    model_info = ml_predictor.model_metrics.get('quality_prediction', {})

    if model_info:
        st.markdown(f"""
        <div class="insight-box">
            <strong>Model Performance:</strong><br>
            • Accuracy: {model_info.get('accuracy', 0):.3f}<br>
            • Features: {', '.join(model_info.get('features', []))}<br>
            • Target: {model_info.get('target', 'Unknown')}
        </div>
        """, unsafe_allow_html=True)

    # Quadrant explanation
    st.markdown("""
    **Quality Quadrants:**
    - **Quadrant 1**: High Performance, High Skill
    - **Quadrant 2**: Moderate Performance, Developing Skills
    - **Quadrant 3**: Needs Improvement, Training Required
    - **Quadrant 4**: Low Performance, Immediate Attention Needed
    """)

    # Input interface
    st.markdown("**Enter Worker Performance Parameters:**")

    features = model_info.get('features', ['SMV', 'Target', 'Production'])
    feature_values = []

    cols = st.columns(len(features))
    for i, feature in enumerate(features):
        with cols[i]:
            if 'performance' in feature.lower():
                value = st.slider(f"{feature}:", 0.0, 2.0, 1.0, 0.1, key=f"qual_{i}")
            else:
                value = st.number_input(f"{feature}:", value=50.0, step=1.0, key=f"qual_{i}")
            feature_values.append(value)

    if st.button("🔮 Predict Quality Quadrant", type="primary"):
        prediction = ml_predictor.predict_quality_quadrant(feature_values)

        if prediction is not None:
            quadrant_descriptions = {
                1: ("🟢 Quadrant 1", "High Performance - Excellent worker performance!"),
                2: ("🟡 Quadrant 2", "Good Performance - Solid contributor with room for growth"),
                3: ("🟠 Quadrant 3", "Needs Improvement - Training and support recommended"),
                4: ("🔴 Quadrant 4", "Low Performance - Immediate intervention required")
            }

            color, description = quadrant_descriptions.get(prediction, ("Unknown", "Unknown quadrant"))

            st.markdown(f"""
            <div class="recommendation-box">
                <h3>🎯 Quality Prediction Result</h3>
                <p><strong>{color}</strong></p>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

def show_business_insights(data):
    """Show business insights and recommendations"""
    st.markdown('<h2 class="section-header">💡 Business Insights & Recommendations</h2>', unsafe_allow_html=True)

    # Generate insights based on data analysis
    insights = generate_business_insights(data)

    # Display insights in organized sections
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Key Recommendations")
        for i, insight in enumerate(insights['recommendations'][:5]):
            st.markdown(f"""
            <div class="recommendation-box">
                <strong>{i+1}. {insight['title']}</strong><br>
                {insight['description']}
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### ⚠️ Areas of Concern")
        for i, concern in enumerate(insights['concerns'][:5]):
            st.markdown(f"""
            <div class="warning-box">
                <strong>{i+1}. {concern['title']}</strong><br>
                {concern['description']}
            </div>
            """, unsafe_allow_html=True)

    # Action plan
    st.markdown("### 📋 Suggested Action Plan")

    action_plan = [
        "🔍 **Immediate Actions (1-2 weeks)**",
        "• Address critical stock shortages identified in inventory analysis",
        "• Implement training programs for low-performing workers",
        "• Investigate and resolve top causes of production loss time",
        "",
        "📈 **Short-term Goals (1-3 months)**",
        "• Optimize production line efficiency based on capacity analysis",
        "• Establish performance monitoring systems for all quadrants",
        "• Implement predictive maintenance to reduce machine breakdowns",
        "",
        "🚀 **Long-term Strategy (3-12 months)**",
        "• Deploy machine learning models for real-time decision making",
        "• Establish data-driven quality control processes",
        "• Create comprehensive workforce development programs"
    ]

    for item in action_plan:
        if item.startswith(('🔍', '📈', '🚀')):
            st.markdown(f"**{item}**")
        else:
            st.markdown(item)

def generate_business_insights(data):
    """Generate business insights from data analysis"""
    insights = {
        'recommendations': [],
        'concerns': []
    }

    # Inventory insights
    if 'stock_entry' in data:
        df = data['stock_entry']
        if 'Stock Qty' in df.columns:
            low_stock_count = len(df[df['Stock Qty'] < 100])
            if low_stock_count > 0:
                insights['concerns'].append({
                    'title': 'Inventory Management',
                    'description': f'{low_stock_count} items have critically low stock levels. Implement automated reorder points to prevent stockouts.'
                })

            insights['recommendations'].append({
                'title': 'Inventory Optimization',
                'description': 'Implement ABC analysis to categorize inventory items and optimize stock levels based on demand patterns.'
            })

    # Workforce insights
    if 'workforce' in data:
        for df in data['workforce'].values():
            if 'Performance %' in df.columns:
                low_performers = len(df[df['Performance %'] < 0.7])
                if low_performers > 0:
                    insights['concerns'].append({
                        'title': 'Workforce Performance',
                        'description': f'{low_performers} workers are underperforming. Targeted training and support programs are needed.'
                    })

                insights['recommendations'].append({
                    'title': 'Performance Management',
                    'description': 'Establish regular performance reviews and skill development programs to improve overall workforce efficiency.'
                })
                break

    # Quality insights
    if 'loss_time' in data:
        insights['recommendations'].append({
            'title': 'Quality Control Enhancement',
            'description': 'Implement real-time monitoring systems to identify and address quality issues before they impact production.'
        })

        insights['concerns'].append({
            'title': 'Production Losses',
            'description': 'Significant time losses detected in production. Root cause analysis and preventive measures are essential.'
        })

    return insights

if __name__ == "__main__":
    main()
