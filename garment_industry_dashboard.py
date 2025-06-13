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
                df = pd.read_csv(file_path)
                # Clean and convert Stock Qty to numeric
                if 'Stock Qty' in df.columns:
                    df['Stock Qty'] = pd.to_numeric(df['Stock Qty'], errors='coerce').fillna(0)
                data[key] = df
        
        # Load capacity and production data
        capacity_files = [f for f in csv_dir.glob('Capacity_study_*') if f.suffix == '.csv']
        data['capacity'] = {}
        for file in capacity_files:
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    data['capacity'][file.stem] = df
            except:
                continue
        
        # Load loss time and quality data
        loss_time_files = [f for f in csv_dir.glob('CCL_loss_time_*') if f.suffix == '.csv']
        data['loss_time'] = {}
        for file in loss_time_files:
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    data['loss_time'][file.stem] = df
            except:
                continue
        
        # Load competency and workforce data
        quadrant_files = [f for f in csv_dir.glob('Quadrant_data_*') if f.suffix == '.csv']
        data['workforce'] = {}
        for file in quadrant_files:
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    # Clean performance data
                    if 'Performance %' in df.columns:
                        df['Performance %'] = pd.to_numeric(df['Performance %'], errors='coerce').fillna(0)
                    if 'Quadrant' in df.columns:
                        df['Quadrant'] = pd.to_numeric(df['Quadrant'], errors='coerce').fillna(4)
                    data['workforce'][file.stem] = df
            except:
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
        """Train model to predict production efficiency"""
        try:
            # Find suitable capacity dataset
            for name, df in capacity_data.items():
                if len(df) > 10:  # Ensure sufficient data
                    # Look for numeric columns that could represent time/efficiency
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) >= 2:
                        # Use first two numeric columns as features and target
                        feature_cols = numeric_cols[:2]
                        target_col = numeric_cols[0] if len(numeric_cols) > 2 else numeric_cols[1]
                        
                        # Prepare data
                        X = df[feature_cols].dropna()
                        y = df[target_col].dropna()
                        
                        # Align indices
                        common_idx = X.index.intersection(y.index)
                        X = X.loc[common_idx]
                        y = y.loc[common_idx]
                        
                        if len(X) > 5:
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.3, random_state=42
                            )
                            
                            # Scale features
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            # Train model
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(X_train_scaled, y_train)
                            
                            # Evaluate
                            y_pred = model.predict(X_test_scaled)
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            
                            # Store model
                            self.models['production_efficiency'] = model
                            self.scalers['production_efficiency'] = scaler
                            self.model_metrics['production_efficiency'] = {
                                'r2_score': r2,
                                'mse': mse,
                                'features': list(feature_cols),
                                'target': target_col,
                                'dataset': name
                            }
                            
                            return True
            return False
        except Exception as e:
            st.error(f"Error training production efficiency model: {str(e)}")
            return False
    
    def train_quality_prediction_model(self, workforce_data):
        """Train model to predict quality based on workforce competency"""
        try:
            for name, df in workforce_data.items():
                if 'Performance' in str(df.columns) and 'Quadrant' in str(df.columns):
                    # Prepare features for quality prediction
                    feature_cols = []
                    for col in df.columns:
                        if any(keyword in col.lower() for keyword in ['smv', 'target', 'production', 'performance']):
                            if df[col].dtype in ['int64', 'float64']:
                                feature_cols.append(col)
                    
                    if len(feature_cols) >= 2 and 'Quadrant' in df.columns:
                        X = df[feature_cols].fillna(df[feature_cols].mean())
                        y = df['Quadrant'].fillna(df['Quadrant'].mode()[0] if not df['Quadrant'].mode().empty else 1)
                        
                        if len(X) > 5:
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.3, random_state=42
                            )
                            
                            # Scale features
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            # Train classifier
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                            model.fit(X_train_scaled, y_train)
                            
                            # Evaluate
                            y_pred = model.predict(X_test_scaled)
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            # Store model
                            self.models['quality_prediction'] = model
                            self.scalers['quality_prediction'] = scaler
                            self.model_metrics['quality_prediction'] = {
                                'accuracy': accuracy,
                                'features': feature_cols,
                                'target': 'Quadrant',
                                'dataset': name
                            }
                            
                            return True
            return False
        except Exception as e:
            st.error(f"Error training quality prediction model: {str(e)}")
            return False
    
    def predict_production_efficiency(self, feature_values):
        """Make production efficiency prediction"""
        if 'production_efficiency' in self.models:
            try:
                X = np.array(feature_values).reshape(1, -1)
                X_scaled = self.scalers['production_efficiency'].transform(X)
                prediction = self.models['production_efficiency'].predict(X_scaled)[0]
                return prediction
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        return None
    
    def predict_quality_quadrant(self, feature_values):
        """Make quality quadrant prediction"""
        if 'quality_prediction' in self.models:
            try:
                X = np.array(feature_values).reshape(1, -1)
                X_scaled = self.scalers['quality_prediction'].transform(X)
                prediction = self.models['quality_prediction'].predict(X_scaled)[0]
                return int(prediction)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        return None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏭 Garment Industry Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Empowering garment manufacturers with data-driven insights and machine learning predictions**")
    
    # Load data
    with st.spinner("Loading garment industry data..."):
        data = load_garment_data()
    
    if not data:
        st.error("Unable to load data. Please check if CSV files are available in the 'csv' directory.")
        return
    
    # Initialize ML predictor
    ml_predictor = GarmentMLPredictor()
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Analysis Section:",
        ["🏠 Executive Summary", "📈 Analysis Dashboard", "🤖 ML Predictions", "💡 Business Insights"]
    )
    
    if page == "🏠 Executive Summary":
        show_executive_summary(data)
    elif page == "📈 Analysis Dashboard":
        show_analysis_dashboard(data)
    elif page == "🤖 ML Predictions":
        show_ml_predictions(data, ml_predictor)
    elif page == "💡 Business Insights":
        show_business_insights(data)

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
                if 'Actual' in df.columns:
                    total_loss_hours += df['Actual'].sum() / 60  # Convert to hours
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
            for name, df in data['workforce'].items():
                if 'Performance %' in df.columns and 'Quadrant' in df.columns:
                    # Performance by quadrant
                    perf_by_quadrant = df.groupby('Quadrant')['Performance %'].mean()

                    fig = px.bar(
                        x=perf_by_quadrant.index,
                        y=perf_by_quadrant.values,
                        title="Average Performance by Quadrant",
                        labels={'x': 'Quadrant', 'y': 'Performance %'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    break

    # Key insights
    st.markdown("### 🔍 Key Insights")

    insights = []

    # Inventory insights
    if 'stock_entry' in data:
        df = data['stock_entry']
        if 'Stock Qty' in df.columns:
            low_stock_items = len(df[df['Stock Qty'] < 100])
            insights.append(f"📦 {low_stock_items} items have low stock levels (< 100 units)")

    # Loss time insights
    if 'loss_time' in data:
        for df in data['loss_time'].values():
            if 'Reason Category' in df.columns:
                top_reason = df['Reason Category'].mode()[0] if not df['Reason Category'].mode().empty else "Unknown"
                insights.append(f"⚠️ Main cause of production loss: {top_reason}")
                break

    # Performance insights
    if 'workforce' in data:
        for df in data['workforce'].values():
            if 'Performance %' in df.columns:
                avg_performance = df['Performance %'].mean()
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

        high_performers = df[df['Performance %'] > 1.0]
        low_performers = df[df['Performance %'] < 0.7]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High Performers", f"{len(high_performers)} ({len(high_performers)/len(df)*100:.1f}%)")
        with col2:
            st.metric("Average Performance", f"{df['Performance %'].mean():.2f}")
        with col3:
            st.metric("Low Performers", f"{len(low_performers)} ({len(low_performers)/len(df)*100:.1f}%)")

def show_ml_predictions(data, ml_predictor):
    """Show machine learning predictions interface"""
    st.markdown('<h2 class="section-header">🤖 Machine Learning Predictions</h2>', unsafe_allow_html=True)

    # Train models if not already trained
    if not ml_predictor.models:
        with st.spinner("Training machine learning models..."):
            # Train production efficiency model
            if 'capacity' in data:
                success = ml_predictor.train_production_efficiency_model(data['capacity'])
                if success:
                    st.success("✅ Production efficiency model trained successfully!")
                else:
                    st.warning("⚠️ Could not train production efficiency model with available data")

            # Train quality prediction model
            if 'workforce' in data:
                success = ml_predictor.train_quality_prediction_model(data['workforce'])
                if success:
                    st.success("✅ Quality prediction model trained successfully!")
                else:
                    st.warning("⚠️ Could not train quality prediction model with available data")

    # Model selection
    available_models = list(ml_predictor.models.keys())

    if not available_models:
        st.error("No machine learning models are available. Please check your data.")
        return

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
