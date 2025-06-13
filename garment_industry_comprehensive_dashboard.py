import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

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
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 8px;
        color: #333;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class GarmentAnalytics:
    def __init__(self):
        self.data = {}
        self.models = {}
        self.scalers = {}
        self.load_data()
        
    @st.cache_data
    def load_data(_self):
        """Load all CSV data files"""
        csv_dir = 'csv'
        data = {}
        
        if not os.path.exists(csv_dir):
            st.error("CSV directory not found. Please ensure data files are in the 'csv' folder.")
            return data
            
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        
        for file in csv_files:
            try:
                df = pd.read_csv(os.path.join(csv_dir, file))
                # Clean column names
                df.columns = df.columns.str.strip()
                data[file] = df
            except Exception as e:
                st.warning(f"Could not load {file}: {str(e)}")
                
        return data
    
    def get_inventory_data(self):
        """Get consolidated inventory data"""
        inventory_files = [f for f in self.data.keys() if 'Stores' in f or 'Stock' in f or 'Material' in f]
        inventory_data = []
        
        for file in inventory_files:
            df = self.data[file].copy()
            df['source_file'] = file
            inventory_data.append(df)
            
        if inventory_data:
            return pd.concat(inventory_data, ignore_index=True, sort=False)
        return pd.DataFrame()
    
    def get_production_data(self):
        """Get consolidated production capacity data"""
        production_files = [f for f in self.data.keys() if 'Capacity' in f or 'Line_balance' in f]
        production_data = []
        
        for file in production_files:
            df = self.data[file].copy()
            df['source_file'] = file
            production_data.append(df)
            
        if production_data:
            return pd.concat(production_data, ignore_index=True, sort=False)
        return pd.DataFrame()
    
    def get_quality_data(self):
        """Get consolidated quality and loss time data"""
        quality_files = [f for f in self.data.keys() if 'CCL' in f or 'loss' in f.lower()]
        quality_data = []
        
        for file in quality_files:
            df = self.data[file].copy()
            df['source_file'] = file
            quality_data.append(df)
            
        if quality_data:
            return pd.concat(quality_data, ignore_index=True, sort=False)
        return pd.DataFrame()
    
    def get_workforce_data(self):
        """Get workforce and competency data"""
        workforce_files = [f for f in self.data.keys() if 'Quadrant' in f]
        
        if workforce_files:
            # Use the main quadrant data file
            main_file = [f for f in workforce_files if f == 'Quadrant data - AI.csv']
            if main_file:
                return self.data[main_file[0]]
                
        return pd.DataFrame()

class MLModels:
    def __init__(self, analytics):
        self.analytics = analytics
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        
    def train_production_model(self):
        """Train production efficiency prediction model"""
        try:
            production_data = self.analytics.get_production_data()
            
            if production_data.empty:
                return None, "No production data available"
                
            # Find numeric columns for features
            numeric_cols = production_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return None, "Insufficient numeric data for training"
                
            # Remove any columns with all NaN values
            numeric_cols = [col for col in numeric_cols if not production_data[col].isna().all()]
            
            if len(numeric_cols) < 2:
                return None, "Insufficient valid numeric data"
                
            # Use the first column as target, rest as features
            target_col = numeric_cols[0]
            feature_cols = numeric_cols[1:]
            
            # Prepare data
            X = production_data[feature_cols].dropna()
            y = production_data[target_col].loc[X.index]
            
            if len(X) < 10:
                return None, "Insufficient samples for training"
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
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
            self.models['production'] = model
            self.scalers['production'] = scaler
            self.feature_names['production'] = feature_cols
            
            return {
                'r2_score': r2,
                'mse': mse,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
                'n_samples': len(X)
            }, None
            
        except Exception as e:
            return None, f"Error training production model: {str(e)}"
    
    def train_quality_model(self):
        """Train quality quadrant classification model"""
        try:
            workforce_data = self.analytics.get_workforce_data()
            
            if workforce_data.empty:
                return None, "No workforce data available"
                
            # Check for required columns
            required_cols = ['SMV', 'Target', 'production', 'Performance %', 'Quadrant']
            available_cols = [col for col in required_cols if col in workforce_data.columns]
            
            if len(available_cols) < 3:
                return None, "Insufficient columns for quality model training"
                
            # Prepare features and target
            feature_cols = [col for col in available_cols if col != 'Quadrant']
            target_col = 'Quadrant' if 'Quadrant' in workforce_data.columns else available_cols[-1]
            
            # Clean data
            df_clean = workforce_data[feature_cols + [target_col]].dropna()
            
            if len(df_clean) < 10:
                return None, "Insufficient clean samples for training"
                
            X = df_clean[feature_cols]
            y = df_clean[target_col]
            
            # Convert to numeric if needed
            for col in feature_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.dropna()
            y = y.loc[X.index]
            
            if len(X) < 10:
                return None, "Insufficient numeric samples"
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model
            self.models['quality'] = model
            self.feature_names['quality'] = feature_cols
            
            return {
                'accuracy': accuracy,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
                'n_samples': len(X),
                'classes': list(model.classes_)
            }, None
            
        except Exception as e:
            return None, f"Error training quality model: {str(e)}"
    
    def predict_production(self, features):
        """Make production prediction"""
        if 'production' not in self.models:
            return None, "Production model not trained"
            
        try:
            # Scale features
            features_scaled = self.scalers['production'].transform([features])
            prediction = self.models['production'].predict(features_scaled)[0]
            return prediction, None
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def predict_quality(self, features):
        """Make quality quadrant prediction"""
        if 'quality' not in self.models:
            return None, "Quality model not trained"
            
        try:
            prediction = self.models['quality'].predict([features])[0]
            probabilities = self.models['quality'].predict_proba([features])[0]
            return {
                'prediction': prediction,
                'probabilities': dict(zip(self.models['quality'].classes_, probabilities))
            }, None
        except Exception as e:
            return None, f"Prediction error: {str(e)}"

# Initialize analytics and models
@st.cache_resource
def initialize_system():
    analytics = GarmentAnalytics()
    ml_models = MLModels(analytics)
    return analytics, ml_models

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>👔 Garment Industry Analytics Dashboard</h1>
        <p>Comprehensive Data Analysis & Machine Learning Insights for Garment Manufacturing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    analytics, ml_models = initialize_system()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Section",
        [
            "🏠 Executive Summary",
            "📊 Analysis Dashboard", 
            "🤖 ML Predictions",
            "💡 Business Insights"
        ]
    )
    
    # Route to appropriate page
    if page == "🏠 Executive Summary":
        executive_summary(analytics)
    elif page == "📊 Analysis Dashboard":
        analysis_dashboard(analytics)
    elif page == "🤖 ML Predictions":
        ml_predictions(analytics, ml_models)
    elif page == "💡 Business Insights":
        business_insights(analytics)

def executive_summary(analytics):
    """Executive Summary Dashboard"""
    st.header("🏠 Executive Summary")

    # Get data summaries
    inventory_data = analytics.get_inventory_data()
    production_data = analytics.get_production_data()
    quality_data = analytics.get_quality_data()
    workforce_data = analytics.get_workforce_data()

    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        inventory_items = len(inventory_data) if not inventory_data.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>📦 Inventory Items</h3>
            <h2>{inventory_items:,}</h2>
            <p>Total tracked items</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        active_workers = len(workforce_data) if not workforce_data.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Active Workers</h3>
            <h2>{active_workers}</h2>
            <p>Performance tracked</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        production_lines = len(production_data['source_file'].unique()) if not production_data.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏭 Production Lines</h3>
            <h2>{production_lines}</h2>
            <p>Capacity analyzed</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        # Calculate total loss time if available
        total_loss_time = 0
        if not quality_data.empty:
            numeric_cols = quality_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                total_loss_time = quality_data[numeric_cols].sum().sum()

        st.markdown(f"""
        <div class="metric-card">
            <h3>⏱️ Total Loss Time</h3>
            <h2>{total_loss_time:,.0f}</h2>
            <p>Minutes tracked</p>
        </div>
        """, unsafe_allow_html=True)

    # Visual Overview Section
    st.subheader("📈 Visual Overview")

    col1, col2 = st.columns(2)

    with col1:
        # Workforce Performance Distribution
        if not workforce_data.empty and 'Quadrant' in workforce_data.columns:
            quadrant_counts = workforce_data['Quadrant'].value_counts().sort_index()

            fig = px.pie(
                values=quadrant_counts.values,
                names=[f"Quadrant {i}" for i in quadrant_counts.index],
                title="Workforce Performance Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Workforce performance data not available for visualization")

    with col2:
        # Production Efficiency Trend
        if not production_data.empty:
            numeric_cols = production_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Create a simple efficiency visualization
                efficiency_data = production_data[numeric_cols].mean()

                fig = px.bar(
                    x=efficiency_data.index[:10],  # Top 10 metrics
                    y=efficiency_data.values[:10],
                    title="Production Metrics Overview",
                    color=efficiency_data.values[:10],
                    color_continuous_scale="Blues"
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Production efficiency data not available for visualization")
        else:
            st.info("Production data not available")

    # Critical Insights
    st.subheader("🔍 Critical Insights")

    insights = []

    # Workforce insights
    if not workforce_data.empty and 'Quadrant' in workforce_data.columns:
        quadrant_4_count = len(workforce_data[workforce_data['Quadrant'] == 4.0])
        if quadrant_4_count > 0:
            insights.append(f"⚠️ {quadrant_4_count} workers in Quadrant 4 need immediate attention")

        high_performers = len(workforce_data[workforce_data['Quadrant'] == 1.0])
        if high_performers > 0:
            insights.append(f"⭐ {high_performers} high-performing workers identified for mentoring roles")

    # Quality insights
    if not quality_data.empty:
        if 'Reason Category' in quality_data.columns:
            top_issue = quality_data['Reason Category'].mode().iloc[0] if len(quality_data['Reason Category'].mode()) > 0 else "Unknown"
            insights.append(f"🔧 Most common issue: {top_issue}")

    # Display insights
    if insights:
        for insight in insights:
            st.markdown(f"""
            <div class="insight-box">
                {insight}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Analyzing data for insights...")

def analysis_dashboard(analytics):
    """Comprehensive Analysis Dashboard"""
    st.header("📊 Analysis Dashboard")

    # Analysis section selector
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Inventory Analysis", "Production Capacity", "Quality Control", "Workforce Performance"]
    )

    if analysis_type == "Inventory Analysis":
        inventory_analysis(analytics)
    elif analysis_type == "Production Capacity":
        production_analysis(analytics)
    elif analysis_type == "Quality Control":
        quality_analysis(analytics)
    elif analysis_type == "Workforce Performance":
        workforce_analysis(analytics)

def inventory_analysis(analytics):
    """Inventory Analysis Section"""
    st.subheader("📦 Inventory Analysis")

    inventory_data = analytics.get_inventory_data()

    if inventory_data.empty:
        st.warning("No inventory data available for analysis")
        return

    # Basic inventory statistics
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Items", len(inventory_data))

        # Find numeric columns for analysis
        numeric_cols = inventory_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            total_value = inventory_data[numeric_cols].sum().sum()
            st.metric("Total Value (All Metrics)", f"{total_value:,.0f}")

    with col2:
        st.metric("Data Sources", len(inventory_data['source_file'].unique()))

        # Show data completeness
        completeness = (1 - inventory_data.isnull().sum().sum() / (len(inventory_data) * len(inventory_data.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")

    # Inventory distribution by source
    if 'source_file' in inventory_data.columns:
        source_counts = inventory_data['source_file'].value_counts()

        fig = px.bar(
            x=source_counts.index,
            y=source_counts.values,
            title="Inventory Items by Data Source",
            labels={'x': 'Data Source', 'y': 'Number of Items'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Show sample data
    st.subheader("Sample Inventory Data")
    st.dataframe(inventory_data.head(10))

def production_analysis(analytics):
    """Production Capacity Analysis"""
    st.subheader("🏭 Production Capacity Analysis")

    production_data = analytics.get_production_data()

    if production_data.empty:
        st.warning("No production data available for analysis")
        return

    # Production metrics
    numeric_cols = production_data.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        col1, col2 = st.columns(2)

        with col1:
            # Average metrics
            avg_metrics = production_data[numeric_cols].mean()

            fig = px.bar(
                x=avg_metrics.index[:10],
                y=avg_metrics.values[:10],
                title="Average Production Metrics",
                color=avg_metrics.values[:10],
                color_continuous_scale="Viridis"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Efficiency distribution
            if len(numeric_cols) > 1:
                efficiency_col = numeric_cols[0]  # Use first numeric column as efficiency proxy

                fig = px.histogram(
                    production_data,
                    x=efficiency_col,
                    title=f"Distribution of {efficiency_col}",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)

    # Production line comparison
    if 'source_file' in production_data.columns and len(numeric_cols) > 0:
        line_performance = production_data.groupby('source_file')[numeric_cols[0]].mean().sort_values(ascending=False)

        st.subheader("Production Line Performance Comparison")

        fig = px.bar(
            x=line_performance.index,
            y=line_performance.values,
            title=f"Average {numeric_cols[0]} by Production Line",
            color=line_performance.values,
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

def quality_analysis(analytics):
    """Quality Control Analysis"""
    st.subheader("🔍 Quality Control Analysis")

    quality_data = analytics.get_quality_data()

    if quality_data.empty:
        st.warning("No quality data available for analysis")
        return

    # Loss time analysis
    if 'Reason Category' in quality_data.columns:
        reason_counts = quality_data['Reason Category'].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                values=reason_counts.values,
                names=reason_counts.index,
                title="Loss Time by Reason Category"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                x=reason_counts.values,
                y=reason_counts.index,
                orientation='h',
                title="Loss Time Frequency by Category",
                color=reason_counts.values,
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Numeric analysis
    numeric_cols = quality_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.subheader("Loss Time Metrics")

        # Summary statistics
        summary_stats = quality_data[numeric_cols].describe()
        st.dataframe(summary_stats)

        # Time series if date column exists
        date_cols = [col for col in quality_data.columns if 'date' in col.lower() or 'Date' in col]
        if date_cols and len(numeric_cols) > 0:
            try:
                quality_data[date_cols[0]] = pd.to_datetime(quality_data[date_cols[0]], errors='coerce')
                daily_loss = quality_data.groupby(date_cols[0])[numeric_cols[0]].sum().reset_index()

                fig = px.line(
                    daily_loss,
                    x=date_cols[0],
                    y=numeric_cols[0],
                    title="Loss Time Trend Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Could not create time series visualization")

def workforce_analysis(analytics):
    """Workforce Performance Analysis"""
    st.subheader("👥 Workforce Performance Analysis")

    workforce_data = analytics.get_workforce_data()

    if workforce_data.empty:
        st.warning("No workforce data available for analysis")
        return

    # Performance quadrant analysis
    if 'Quadrant' in workforce_data.columns:
        quadrant_summary = workforce_data['Quadrant'].value_counts().sort_index()

        col1, col2 = st.columns(2)

        with col1:
            # Quadrant distribution
            quadrant_labels = {
                1.0: "High Performance",
                2.0: "Good Performance",
                3.0: "Needs Improvement",
                4.0: "Low Performance"
            }

            labels = [quadrant_labels.get(q, f"Quadrant {q}") for q in quadrant_summary.index]

            fig = px.pie(
                values=quadrant_summary.values,
                names=labels,
                title="Workforce Performance Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Performance metrics by quadrant
            if 'Performance %' in workforce_data.columns:
                avg_performance = workforce_data.groupby('Quadrant')['Performance %'].mean()

                fig = px.bar(
                    x=[quadrant_labels.get(q, f"Q{q}") for q in avg_performance.index],
                    y=avg_performance.values,
                    title="Average Performance % by Quadrant",
                    color=avg_performance.values,
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig, use_container_width=True)

    # Section-wise analysis
    if 'Sections' in workforce_data.columns:
        section_performance = workforce_data.groupby('Sections').agg({
            'Performance %': 'mean',
            'Quadrant': 'count'
        }).round(2)
        section_performance.columns = ['Avg Performance %', 'Worker Count']

        st.subheader("Performance by Section")
        st.dataframe(section_performance)

        # Section performance visualization
        fig = px.scatter(
            section_performance.reset_index(),
            x='Worker Count',
            y='Avg Performance %',
            text='Sections',
            title="Section Performance vs Worker Count",
            size='Worker Count'
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

def ml_predictions(analytics, ml_models):
    """Machine Learning Predictions Interface"""
    st.header("🤖 Machine Learning Predictions")

    # Model training section
    st.subheader("🔧 Model Training & Status")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Production Efficiency Model")
        if st.button("Train Production Model"):
            with st.spinner("Training production efficiency model..."):
                result, error = ml_models.train_production_model()

                if result:
                    st.success("✅ Production model trained successfully!")
                    st.json(result)
                else:
                    st.error(f"❌ Training failed: {error}")

    with col2:
        st.markdown("### Quality Quadrant Model")
        if st.button("Train Quality Model"):
            with st.spinner("Training quality classification model..."):
                result, error = ml_models.train_quality_model()

                if result:
                    st.success("✅ Quality model trained successfully!")
                    st.json(result)
                else:
                    st.error(f"❌ Training failed: {error}")

    st.divider()

    # Prediction interfaces
    prediction_type = st.selectbox(
        "Select Prediction Type",
        ["Production Efficiency Prediction", "Quality Quadrant Prediction"]
    )

    if prediction_type == "Production Efficiency Prediction":
        production_prediction_interface(ml_models)
    elif prediction_type == "Quality Quadrant Prediction":
        quality_prediction_interface(ml_models)

def production_prediction_interface(ml_models):
    """Production efficiency prediction interface"""
    st.subheader("📈 Production Efficiency Prediction")

    if 'production' not in ml_models.models:
        st.warning("⚠️ Production model not trained. Please train the model first.")
        return

    st.markdown("""
    **Business Purpose**: Predict production efficiency based on operational parameters to optimize
    resource allocation and identify potential bottlenecks before they occur.
    """)

    # Get feature names
    feature_names = ml_models.feature_names.get('production', [])

    if not feature_names:
        st.error("No feature information available for production model")
        return

    st.markdown("### Input Parameters")

    # Create input fields for each feature
    feature_values = []
    cols = st.columns(min(3, len(feature_names)))

    for i, feature in enumerate(feature_names):
        with cols[i % len(cols)]:
            value = st.number_input(
                f"{feature}",
                value=0.0,
                help=f"Enter value for {feature}"
            )
            feature_values.append(value)

    if st.button("🔮 Predict Production Efficiency", type="primary"):
        prediction, error = ml_models.predict_production(feature_values)

        if prediction is not None:
            st.markdown(f"""
            <div class="recommendation-box">
                <h3>🎯 Predicted Production Value: {prediction:.2f}</h3>
                <p>Based on the input parameters, this is the expected production outcome.</p>
            </div>
            """, unsafe_allow_html=True)

            # Business interpretation
            if prediction > 50:  # Adjust threshold based on your data
                st.success("✅ **High Efficiency Expected** - Production parameters are optimized")
            elif prediction > 25:
                st.warning("⚠️ **Moderate Efficiency** - Consider optimization opportunities")
            else:
                st.error("❌ **Low Efficiency Alert** - Immediate attention required")

        else:
            st.error(f"Prediction failed: {error}")

def quality_prediction_interface(ml_models):
    """Quality quadrant prediction interface"""
    st.subheader("🎯 Quality Quadrant Prediction")

    if 'quality' not in ml_models.models:
        st.warning("⚠️ Quality model not trained. Please train the model first.")
        return

    st.markdown("""
    **Business Purpose**: Classify worker performance into improvement quadrants to identify
    training needs, recognize top performers, and optimize workforce development strategies.
    """)

    # Get feature names
    feature_names = ml_models.feature_names.get('quality', [])

    if not feature_names:
        st.error("No feature information available for quality model")
        return

    st.markdown("### Worker Performance Parameters")

    # Create input fields
    feature_values = []
    cols = st.columns(min(3, len(feature_names)))

    for i, feature in enumerate(feature_names):
        with cols[i % len(cols)]:
            if feature == 'SMV':
                value = st.number_input(f"{feature} (Standard Minute Value)", value=1.0, min_value=0.1)
            elif feature == 'Target':
                value = st.number_input(f"{feature} (Target Production)", value=100.0, min_value=1.0)
            elif feature == 'production':
                value = st.number_input(f"{feature} (Actual Production)", value=80.0, min_value=0.0)
            elif 'Performance' in feature:
                value = st.number_input(f"{feature}", value=0.8, min_value=0.0, max_value=2.0)
            else:
                value = st.number_input(f"{feature}", value=0.0)
            feature_values.append(value)

    if st.button("🎯 Predict Performance Quadrant", type="primary"):
        result, error = ml_models.predict_quality(feature_values)

        if result:
            prediction = result['prediction']
            probabilities = result['probabilities']

            # Quadrant interpretations
            quadrant_info = {
                1.0: {"name": "High Performance", "color": "success", "action": "Mentoring Role"},
                2.0: {"name": "Good Performance", "color": "info", "action": "Maintain Standards"},
                3.0: {"name": "Needs Improvement", "color": "warning", "action": "Training Required"},
                4.0: {"name": "Low Performance", "color": "error", "action": "Immediate Intervention"}
            }

            info = quadrant_info.get(prediction, {"name": f"Quadrant {prediction}", "color": "info", "action": "Review Required"})

            st.markdown(f"""
            <div class="recommendation-box">
                <h3>🎯 Predicted Quadrant: {info['name']}</h3>
                <p><strong>Recommended Action:</strong> {info['action']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Show probability distribution
            st.markdown("### Prediction Confidence")
            prob_df = pd.DataFrame(list(probabilities.items()), columns=['Quadrant', 'Probability'])
            prob_df['Probability'] = prob_df['Probability'] * 100

            fig = px.bar(
                prob_df,
                x='Quadrant',
                y='Probability',
                title="Prediction Probability by Quadrant",
                color='Probability',
                color_continuous_scale="Viridis"
            )
            fig.update_layout(yaxis_title="Probability (%)")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"Prediction failed: {error}")

def business_insights(analytics):
    """Business Insights and Recommendations"""
    st.header("💡 Business Insights & Recommendations")

    # Get all data for comprehensive analysis
    inventory_data = analytics.get_inventory_data()
    production_data = analytics.get_production_data()
    quality_data = analytics.get_quality_data()
    workforce_data = analytics.get_workforce_data()

    # Key Recommendations
    st.subheader("🎯 Key Recommendations")

    recommendations = []

    # Workforce recommendations
    if not workforce_data.empty and 'Quadrant' in workforce_data.columns:
        quadrant_4_count = len(workforce_data[workforce_data['Quadrant'] == 4.0])
        quadrant_1_count = len(workforce_data[workforce_data['Quadrant'] == 1.0])

        if quadrant_4_count > 0:
            recommendations.append({
                "priority": "High",
                "area": "Workforce Development",
                "issue": f"{quadrant_4_count} workers in low performance quadrant",
                "action": "Implement immediate training programs and performance improvement plans",
                "impact": "15-25% productivity improvement expected"
            })

        if quadrant_1_count > 0:
            recommendations.append({
                "priority": "Medium",
                "area": "Knowledge Transfer",
                "issue": f"{quadrant_1_count} high performers available for mentoring",
                "action": "Establish mentorship programs and best practice sharing sessions",
                "impact": "10-15% overall team performance improvement"
            })

    # Quality recommendations
    if not quality_data.empty and 'Reason Category' in quality_data.columns:
        top_issues = quality_data['Reason Category'].value_counts().head(3)
        for issue, count in top_issues.items():
            recommendations.append({
                "priority": "High",
                "area": "Quality Control",
                "issue": f"Frequent {issue} issues ({count} occurrences)",
                "action": f"Investigate root causes and implement preventive measures for {issue}",
                "impact": "20-30% reduction in loss time expected"
            })

    # Production recommendations
    if not production_data.empty:
        recommendations.append({
            "priority": "Medium",
            "area": "Production Optimization",
            "issue": "Multiple production lines with varying efficiency",
            "action": "Implement line balancing optimization and capacity planning",
            "impact": "10-20% increase in overall production efficiency"
        })

    # Display recommendations
    for i, rec in enumerate(recommendations[:6]):  # Show top 6 recommendations
        priority_color = {"High": "warning-box", "Medium": "recommendation-box", "Low": "insight-box"}

        st.markdown(f"""
        <div class="{priority_color.get(rec['priority'], 'insight-box')}">
            <h4>🎯 {rec['area']} - {rec['priority']} Priority</h4>
            <p><strong>Issue:</strong> {rec['issue']}</p>
            <p><strong>Recommended Action:</strong> {rec['action']}</p>
            <p><strong>Expected Impact:</strong> {rec['impact']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Areas of Concern
    st.subheader("⚠️ Areas Requiring Attention")

    concerns = []

    # Data quality concerns
    total_files = len(analytics.data)
    if total_files < 5:
        concerns.append("Limited data sources - Consider expanding data collection")

    # Workforce concerns
    if not workforce_data.empty and 'Performance %' in workforce_data.columns:
        low_performers = len(workforce_data[workforce_data['Performance %'] < 0.7])
        if low_performers > len(workforce_data) * 0.2:  # More than 20% low performers
            concerns.append(f"High percentage of low performers ({low_performers}/{len(workforce_data)}) - Systematic training needed")

    # Quality concerns
    if not quality_data.empty:
        numeric_cols = quality_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            total_loss = quality_data[numeric_cols].sum().sum()
            if total_loss > 100000:  # Adjust threshold as needed
                concerns.append(f"High total loss time ({total_loss:,.0f} minutes) - Process improvement critical")

    if concerns:
        for concern in concerns:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ {concern}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No critical concerns identified in current analysis")

    # Strategic Action Plan
    st.subheader("📋 Strategic Action Plan")

    action_plan = {
        "Immediate (1-2 weeks)": [
            "Train low-performing workers identified in Quadrant 4",
            "Address top 3 quality issues causing loss time",
            "Implement daily performance monitoring"
        ],
        "Short-term (1-3 months)": [
            "Establish mentorship programs with high performers",
            "Optimize production line balancing",
            "Implement predictive maintenance schedules"
        ],
        "Long-term (3-12 months)": [
            "Deploy real-time monitoring systems",
            "Expand data collection and analytics capabilities",
            "Develop advanced predictive models for demand forecasting"
        ]
    }

    for timeframe, actions in action_plan.items():
        st.markdown(f"### {timeframe}")
        for action in actions:
            st.markdown(f"- {action}")

if __name__ == "__main__":
    main()
