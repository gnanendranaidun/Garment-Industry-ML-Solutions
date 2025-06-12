import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

# Set page config
st.set_page_config(
    page_title="Garment Industry ML Dashboard",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
    }
    .insight-box {
        background-color: #e0ffe0;
        color: #1a5e20;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .model-info {
        background-color: #fff3e0;
        color: #e65100;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_csv_data():
    csv_dir = Path('csv')
    data = {}
    
    # Load Stores data
    stores_files = {
        'material_outward': 'Stores_-_Data_sets_for_AI_training_program__Material_outward.csv',
        'material_inhouse': 'Stores_-_Data_sets_for_AI_training_program__Material_Inhouse.csv',
        'stock_entry': 'Stores_-_Data_sets_for_AI_training_program__Stock_Entry_&_Location.csv',
        'fdr_fra': 'Stores_-_Data_sets_for_AI_training_program__FDR_&_FRA_tracker.csv',
        'inward_grn': 'Stores_-_Data_sets_for_AI_training_program__Inward_&_GRN.csv'
    }
    
    for key, file in stores_files.items():
        try:
            data[key] = pd.read_csv(csv_dir / file)
            st.success(f"Successfully loaded {file}")
        except Exception as e:
            st.warning(f"Error loading {file}: {str(e)}")
    
    # Load Loss Time data
    loss_time_files = [f for f in os.listdir(csv_dir) if 'CCL_loss_time' in f]
    data['loss_time'] = {}
    for file in loss_time_files:
        try:
            data['loss_time'][file] = pd.read_csv(csv_dir / file)
            st.success(f"Successfully loaded {file}")
        except Exception as e:
            st.warning(f"Error loading {file}: {str(e)}")
    
    # Load Capacity Study data
    capacity_files = [f for f in os.listdir(csv_dir) if 'Capacity_study' in f]
    data['capacity'] = {}
    for file in capacity_files:
        try:
            data['capacity'][file] = pd.read_csv(csv_dir / file)
            st.success(f"Successfully loaded {file}")
        except Exception as e:
            st.warning(f"Error loading {file}: {str(e)}")
    
    # Load Quadrant data
    quadrant_files = [f for f in os.listdir(csv_dir) if 'Quadrant_data' in f]
    data['quadrant'] = {}
    for file in quadrant_files:
        try:
            data['quadrant'][file] = pd.read_csv(csv_dir / file)
            st.success(f"Successfully loaded {file}")
        except Exception as e:
            st.warning(f"Error loading {file}: {str(e)}")
    
    return data

class MLPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.model_info = {}
    
    def train_production_prediction(self, data):
        try:
            # Prepare features
            X = data[['Operation tack time', 'Req.Tack time']].dropna()
            y = data['Time (min)'].dropna()
            
            if len(X) > 0 and len(y) > 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                # Train model
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                model.fit(X_train_scaled, y_train)
                
                # Store model and scaler
                self.models['production'] = model
                self.scalers['production'] = scaler
                
                # Calculate and store model info
                y_pred = model.predict(scaler.transform(X_test))
                mse = mean_squared_error(y_test, y_pred)
                feature_importance = dict(zip(X.columns, model.feature_importances_))
                
                self.model_info['production'] = {
                    'mse': mse,
                    'feature_importance': feature_importance,
                    'n_samples': len(X),
                    'features': list(X.columns),
                    'target': 'Time (min)'
                }
                
                return model.score(scaler.transform(X_test), y_test)
        except Exception as e:
            st.error(f"Error in production prediction training: {str(e)}")
        return None
    
    def train_quality_prediction(self, data):
        try:
            # Prepare features
            X = data[['Operation', 'Operator', 'Time']].dropna()
            y = data['Defects'].dropna()
            
            if len(X) > 0 and len(y) > 0:
                # Encode categorical features
                label_encoders = {}
                for col in ['Operation', 'Operator']:
                    label_encoders[col] = LabelEncoder()
                    X[col] = label_encoders[col].fit_transform(X[col])
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Store model and encoders
                self.models['quality'] = model
                self.label_encoders['quality'] = label_encoders
                
                # Calculate and store model info
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                feature_importance = dict(zip(X.columns, model.feature_importances_))
                
                self.model_info['quality'] = {
                    'accuracy': accuracy,
                    'feature_importance': feature_importance,
                    'n_samples': len(X),
                    'features': list(X.columns),
                    'target': 'Defects'
                }
                
                return accuracy
        except Exception as e:
            st.error(f"Error in quality prediction training: {str(e)}")
        return None
    
    def predict_production_time(self, operation_time, req_time):
        if 'production' in self.models:
            try:
                X = np.array([[operation_time, req_time]])
                X_scaled = self.scalers['production'].transform(X)
                prediction = self.models['production'].predict(X_scaled)[0]
                return prediction, self.model_info['production']
            except Exception as e:
                st.error(f"Error in production prediction: {str(e)}")
        return None, None
    
    def predict_quality(self, operation, operator, time):
        if 'quality' in self.models:
            try:
                # Encode categorical features
                X = pd.DataFrame({
                    'Operation': [operation],
                    'Operator': [operator],
                    'Time': [time]
                })
                
                for col, encoder in self.label_encoders['quality'].items():
                    X[col] = encoder.transform(X[col])
                
                prediction = self.models['quality'].predict(X)[0]
                return prediction, self.model_info['quality']
            except Exception as e:
                st.error(f"Error in quality prediction: {str(e)}")
        return None, None

def show_model_info(model_info):
    if model_info:
        st.markdown("""
        <div class="model-info">
            <h3>Model Information</h3>
            <ul>
                <li>Number of samples: {}</li>
                <li>Features: {}</li>
                <li>Target variable: {}</li>
            </ul>
            <h4>Model Performance</h4>
            <ul>
                {}
            </ul>
            <h4>Feature Importance</h4>
            <ul>
                {}
            </ul>
        </div>
        """.format(
            model_info['n_samples'],
            ', '.join(model_info['features']),
            model_info['target'],
            ''.join([f'<li>{k}: {v:.4f}</li>' for k, v in model_info.items() if k not in ['n_samples', 'features', 'target', 'feature_importance']]),
            ''.join([f'<li>{k}: {v:.4f}</li>' for k, v in model_info['feature_importance'].items()])
        ), unsafe_allow_html=True)

def production_live_demo(data, ml_predictor):
    st.subheader("Production Time Prediction Demo")
    
    col1, col2 = st.columns(2)
    with col1:
        operation_time = st.number_input("Operation Tack Time (min)", min_value=0.0, value=1.0)
        req_time = st.number_input("Required Tack Time (min)", min_value=0.0, value=1.0)
    
    if st.button("Predict Production Time"):
        prediction, model_info = ml_predictor.predict_production_time(operation_time, req_time)
        if prediction is not None:
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Predicted Production Time: {prediction:.2f} minutes</h3>
            </div>
            """, unsafe_allow_html=True)
            show_model_info(model_info)

def quality_live_demo(data, ml_predictor):
    st.subheader("Quality Prediction Demo")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        operation = st.selectbox("Operation", data['capacity'].keys())
    with col2:
        operator = st.text_input("Operator ID")
    with col3:
        time = st.number_input("Operation Time (min)", min_value=0.0, value=1.0)
    
    if st.button("Predict Quality"):
        prediction, model_info = ml_predictor.predict_quality(operation, operator, time)
        if prediction is not None:
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Predicted Quality: {'Good' if prediction == 1 else 'Needs Improvement'}</h3>
            </div>
            """, unsafe_allow_html=True)
            show_model_info(model_info)

def inventory_live_demo(data):
    st.subheader("Inventory Analysis Demo")
    
    if 'stock_entry' in data:
        df = data['stock_entry']
        
        # Show data info
        st.markdown("""
        <div class="model-info">
            <h3>Data Information</h3>
            <ul>
                <li>Number of records: {}</li>
                <li>Columns: {}</li>
            </ul>
        </div>
        """.format(
            len(df),
            ', '.join(df.columns)
        ), unsafe_allow_html=True)
        
        # Material type distribution
        fig = px.pie(df, names='Material Type', title='Material Type Distribution')
        st.plotly_chart(fig)
        
        # Stock level trends
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            stock_trend = df.groupby('Date')['Quantity'].sum().reset_index()
            fig = px.line(stock_trend, x='Date', y='Quantity', title='Stock Level Trends')
            st.plotly_chart(fig)

def production_optimization_page(data):
    st.title("Production Optimization")
    
    if 'capacity' in data:
        # Select capacity study file
        capacity_file = st.selectbox("Select Capacity Study File", list(data['capacity'].keys()))
        df = data['capacity'][capacity_file]
        
        # Show data info
        st.markdown("""
        <div class="model-info">
            <h3>Data Information</h3>
            <ul>
                <li>Number of records: {}</li>
                <li>Columns: {}</li>
            </ul>
        </div>
        """.format(
            len(df),
            ', '.join(df.columns)
        ), unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_operations = len(df)
            st.metric("Total Operations", total_operations)
        with col2:
            avg_time = df['Time (min)'].mean()
            st.metric("Average Time", f"{avg_time:.2f} min")
        with col3:
            total_time = df['Time (min)'].sum()
            st.metric("Total Time", f"{total_time:.2f} min")
        
        # Operation time distribution
        st.subheader("Operation Time Distribution")
        fig = px.histogram(df, x='Time (min)', title='Operation Time Distribution')
        st.plotly_chart(fig)
        
        # Bottleneck analysis
        st.subheader("Bottleneck Analysis")
        bottlenecks = df[df['Time (min)'] > df['Time (min)'].mean() * 1.5]
        if not bottlenecks.empty:
            st.dataframe(bottlenecks[['Operation', 'Time (min)']])
        else:
            st.info("No significant bottlenecks identified.")
        
        # Line balancing visualization
        st.subheader("Line Balancing")
        fig = px.bar(df, x='Operation', y='Time (min)', title='Operation Times')
        fig.add_hline(y=df['Time (min)'].mean(), line_dash="dash", line_color="red",
                     annotation_text="Average Time")
        st.plotly_chart(fig)

def quality_control_page(data):
    st.title("Quality Control")
    
    if 'loss_time' in data:
        # Select loss time file
        loss_time_file = st.selectbox("Select Loss Time File", list(data['loss_time'].keys()))
        df = data['loss_time'][loss_time_file]
        
        # Show data info
        st.markdown("""
        <div class="model-info">
            <h3>Data Information</h3>
            <ul>
                <li>Number of records: {}</li>
                <li>Columns: {}</li>
            </ul>
        </div>
        """.format(
            len(df),
            ', '.join(df.columns)
        ), unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_issues = len(df)
            st.metric("Total Issues", total_issues)
        with col2:
            if 'Time' in df.columns:
                total_time = df['Time'].sum()
                st.metric("Total Loss Time", f"{total_time:.2f} min")
        with col3:
            if 'Defects' in df.columns:
                total_defects = df['Defects'].sum()
                st.metric("Total Defects", total_defects)
        
        # Issue type distribution
        if 'Issue_Type' in df.columns:
            st.subheader("Issue Type Distribution")
            fig = px.pie(df, names='Issue_Type', title='Issue Type Distribution')
            st.plotly_chart(fig)
        
        # Time trend
        if 'Date' in df.columns and 'Time' in df.columns:
            st.subheader("Loss Time Trend")
            df['Date'] = pd.to_datetime(df['Date'])
            time_trend = df.groupby('Date')['Time'].sum().reset_index()
            fig = px.line(time_trend, x='Date', y='Time', title='Loss Time Trend')
            st.plotly_chart(fig)

def main():
    # Load data
    data = load_csv_data()
    
    # Initialize ML predictor
    ml_predictor = MLPredictor()
    
    # Train models if data is available
    if 'capacity' in data:
        for file, df in data['capacity'].items():
            if 'Operation tack time' in df.columns and 'Time (min)' in df.columns:
                ml_predictor.train_production_prediction(df)
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Live Demos", "Production", "Quality"])
    
    if page == "Home":
        st.title("Garment Industry ML Dashboard")
        st.markdown("""
        Welcome to the Garment Industry ML Dashboard. This application provides insights and recommendations
        across various aspects of garment manufacturing using machine learning.

        ### Key Features:
        - **Live Demos:** Interactive demonstrations of ML predictions
        - **Production Optimization:** Line balancing and efficiency analysis
        - **Quality Control:** Defect analysis and prediction
        - **Inventory Management:** Stock level analysis and optimization
        """)
    
    elif page == "Live Demos":
        st.title("Live Demos")
        
        demo_type = st.selectbox("Select Demo Type", ["Production", "Quality", "Inventory"])
        
        if demo_type == "Production":
            production_live_demo(data, ml_predictor)
        elif demo_type == "Quality":
            quality_live_demo(data, ml_predictor)
        elif demo_type == "Inventory":
            inventory_live_demo(data)
    
    elif page == "Production":
        production_optimization_page(data)
    
    elif page == "Quality":
        quality_control_page(data)

if __name__ == "__main__":
    main() 