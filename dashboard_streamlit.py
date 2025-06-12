import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os
from sklearn.preprocessing import StandardScaler
import json
import time
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="Production Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# Load ML model and scaler
def load_ml_model():
    try:
        model_path = os.path.join('models', 'production_model.joblib')
        scaler_path = os.path.join('models', 'scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            st.session_state.ml_model = joblib.load(model_path)
            st.session_state.scaler = joblib.load(scaler_path)
            return True
        return False
    except Exception as e:
        st.error(f"Error loading ML model: {str(e)}")
        return False

# Load data with real-time updates
def load_data():
    try:
        data_path = os.path.join('data', 'production_data.csv')
        if os.path.exists(data_path):
            # Check if file has been modified
            file_mod_time = os.path.getmtime(data_path)
            if st.session_state.last_update is None or file_mod_time > st.session_state.last_update:
                st.session_state.data = pd.read_csv(data_path)
                st.session_state.last_update = file_mod_time
                return True
        return False
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False

# Export data to Excel
def get_download_link(df, filename, text):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Production Optimization Functions
def calculate_optimal_parameters(data):
    """Calculate optimal production parameters based on historical data"""
    if data is None or data.empty:
        return None
    
    # Calculate optimal parameters for each product
    optimal_params = {}
    for product in data['product_id'].unique():
        product_data = data[data['product_id'] == product]
        
        # Calculate optimal temperature using weighted average
        temp_stats = product_data['temperature'].describe()
        quality_weights = product_data['quality_score']
        optimal_temp = np.average(product_data['temperature'], weights=quality_weights)
        
        # Calculate optimal pressure using weighted average
        pressure_stats = product_data['pressure'].describe()
        optimal_pressure = np.average(product_data['pressure'], weights=quality_weights)
        
        # Calculate optimal speed using weighted average
        speed_stats = product_data['speed'].describe()
        optimal_speed = np.average(product_data['speed'], weights=quality_weights)
        
        optimal_params[product] = {
            'temperature': optimal_temp,
            'pressure': optimal_pressure,
            'speed': optimal_speed,
            'confidence': quality_weights.mean()  # Add confidence score
        }
    
    return optimal_params

def calculate_efficiency_metrics(data):
    """Calculate production efficiency metrics"""
    if data is None or data.empty:
        return None
    
    # Calculate OEE (Overall Equipment Effectiveness)
    total_time = data['end_time'] - data['start_time']
    planned_production_time = total_time.sum()
    actual_production_time = data['actual_production_time'].sum()
    good_units = data['good_units'].sum()
    total_units = data['total_units'].sum()
    
    availability = actual_production_time / planned_production_time
    performance = (good_units / actual_production_time) / (total_units / planned_production_time)
    quality = good_units / total_units
    
    oee = availability * performance * quality
    
    # Calculate additional metrics
    downtime = planned_production_time - actual_production_time
    downtime_percentage = downtime / planned_production_time
    
    return {
        'oee': oee,
        'availability': availability,
        'performance': performance,
        'quality': quality,
        'downtime': downtime,
        'downtime_percentage': downtime_percentage
    }

# Quality Monitoring Functions
def calculate_quality_metrics(data):
    """Calculate quality metrics"""
    if data is None or data.empty:
        return None
    
    # Calculate defect rate
    total_units = data['total_units'].sum()
    defect_units = data['defect_units'].sum()
    defect_rate = defect_units / total_units
    
    # Calculate quality score
    quality_score = 1 - defect_rate
    
    # Calculate quality trends
    quality_trends = data.groupby('date')['quality_score'].agg(['mean', 'std']).reset_index()
    
    # Calculate defect types distribution
    defect_types = data['defect_type'].value_counts()
    
    return {
        'defect_rate': defect_rate,
        'quality_score': quality_score,
        'quality_trends': quality_trends,
        'defect_types': defect_types
    }

# ML Prediction Functions
def prepare_features(data):
    """Prepare features for ML prediction"""
    if data is None or data.empty:
        return None
    
    # Select relevant features
    features = ['temperature', 'pressure', 'speed', 'humidity']
    
    # Scale features
    if st.session_state.scaler is not None:
        scaled_features = st.session_state.scaler.transform(data[features])
        return scaled_features
    return None

def predict_quality(data):
    """Predict quality using ML model"""
    if data is None or data.empty or st.session_state.ml_model is None:
        return None
    
    # Prepare features
    features = prepare_features(data)
    if features is None:
        return None
    
    # Make predictions
    predictions = st.session_state.ml_model.predict(features)
    probabilities = st.session_state.ml_model.predict_proba(features)
    
    # Calculate prediction confidence
    confidence = np.max(probabilities, axis=1)
    
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'confidence': confidence
    }

# Dashboard Layout
def main():
    st.title("🏭 Production Dashboard")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Production Optimization", "Quality Monitoring", "ML Predictions"])
    
    # Auto-refresh toggle
    st.sidebar.header("Settings")
    st.session_state.auto_refresh = st.sidebar.checkbox("Enable Auto-refresh", value=st.session_state.auto_refresh)
    if st.session_state.auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 30)
        time.sleep(refresh_interval)
        st.experimental_rerun()
    
    # Load data and model
    if st.session_state.data is None:
        load_data()
    if st.session_state.ml_model is None:
        load_ml_model()
    
    if page == "Overview":
        show_overview()
    elif page == "Production Optimization":
        show_production_optimization()
    elif page == "Quality Monitoring":
        show_quality_monitoring()
    elif page == "ML Predictions":
        show_ml_predictions()

def show_overview():
    st.header("Production Overview")
    
    if st.session_state.data is not None:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Production", f"{st.session_state.data['total_units'].sum():,.0f}")
        with col2:
            st.metric("Good Units", f"{st.session_state.data['good_units'].sum():,.0f}")
        with col3:
            st.metric("Defect Rate", f"{(st.session_state.data['defect_units'].sum() / st.session_state.data['total_units'].sum() * 100):.1f}%")
        with col4:
            st.metric("Average Quality Score", f"{st.session_state.data['quality_score'].mean():.2f}")
        
        # Production trends
        st.subheader("Production Trends")
        fig = px.line(st.session_state.data, x='date', y='total_units', 
                     title='Daily Production Volume',
                     labels={'total_units': 'Production Volume', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Quality distribution
        st.subheader("Quality Distribution")
        fig = px.histogram(st.session_state.data, x='quality_score', 
                          title='Quality Score Distribution',
                          nbins=20,
                          labels={'quality_score': 'Quality Score', 'count': 'Frequency'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Export data
        st.subheader("Export Data")
        if st.button("Export to Excel"):
            st.markdown(get_download_link(st.session_state.data, 'production_data.xlsx', 'Download Excel File'), unsafe_allow_html=True)

def show_production_optimization():
    st.header("Production Optimization")
    
    if st.session_state.data is not None:
        # Calculate optimal parameters
        optimal_params = calculate_optimal_parameters(st.session_state.data)
        efficiency_metrics = calculate_efficiency_metrics(st.session_state.data)
        
        if optimal_params and efficiency_metrics:
            # Display optimal parameters
            st.subheader("Optimal Production Parameters")
            for product, params in optimal_params.items():
                st.write(f"Product {product}:")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Temperature", f"{params['temperature']:.1f}°C")
                with col2:
                    st.metric("Pressure", f"{params['pressure']:.1f} bar")
                with col3:
                    st.metric("Speed", f"{params['speed']:.1f} units/min")
                with col4:
                    st.metric("Confidence", f"{params['confidence']:.1%}")
            
            # Display efficiency metrics
            st.subheader("Efficiency Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("OEE", f"{efficiency_metrics['oee']:.1%}")
            with col2:
                st.metric("Availability", f"{efficiency_metrics['availability']:.1%}")
            with col3:
                st.metric("Performance", f"{efficiency_metrics['performance']:.1%}")
            with col4:
                st.metric("Quality", f"{efficiency_metrics['quality']:.1%}")
            with col5:
                st.metric("Downtime", f"{efficiency_metrics['downtime_percentage']:.1%}")
            
            # Parameter trends
            st.subheader("Parameter Trends")
            fig = px.line(st.session_state.data, x='date', y=['temperature', 'pressure', 'speed'],
                         title='Production Parameters Over Time',
                         labels={'value': 'Parameter Value', 'variable': 'Parameter', 'date': 'Date'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Export optimization report
            st.subheader("Export Optimization Report")
            if st.button("Export Report"):
                report_data = pd.DataFrame({
                    'Product': list(optimal_params.keys()),
                    'Temperature': [params['temperature'] for params in optimal_params.values()],
                    'Pressure': [params['pressure'] for params in optimal_params.values()],
                    'Speed': [params['speed'] for params in optimal_params.values()],
                    'Confidence': [params['confidence'] for params in optimal_params.values()]
                })
                st.markdown(get_download_link(report_data, 'optimization_report.xlsx', 'Download Optimization Report'), unsafe_allow_html=True)

def show_quality_monitoring():
    st.header("Quality Monitoring")
    
    if st.session_state.data is not None:
        # Calculate quality metrics
        quality_metrics = calculate_quality_metrics(st.session_state.data)
        
        if quality_metrics:
            # Display quality metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Defect Rate", f"{quality_metrics['defect_rate']:.1%}")
            with col2:
                st.metric("Quality Score", f"{quality_metrics['quality_score']:.2f}")
            
            # Quality trends with confidence intervals
            st.subheader("Quality Trends")
            fig = px.line(quality_metrics['quality_trends'], x='date', y='mean',
                         error_y='std',
                         title='Quality Score Over Time with Standard Deviation',
                         labels={'mean': 'Quality Score', 'date': 'Date'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Defect analysis
            st.subheader("Defect Analysis")
            defect_data = st.session_state.data[st.session_state.data['defect_units'] > 0]
            if not defect_data.empty:
                # Defect distribution by parameters
                fig = px.scatter(defect_data, x='temperature', y='pressure',
                               color='defect_units', size='defect_units',
                               title='Defect Distribution by Parameters',
                               labels={'temperature': 'Temperature (°C)', 'pressure': 'Pressure (bar)', 'defect_units': 'Number of Defects'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Defect types distribution
                st.subheader("Defect Types Distribution")
                fig = px.pie(values=quality_metrics['defect_types'].values,
                           names=quality_metrics['defect_types'].index,
                           title='Distribution of Defect Types')
                st.plotly_chart(fig, use_container_width=True)
            
            # Export quality report
            st.subheader("Export Quality Report")
            if st.button("Export Quality Report"):
                report_data = pd.DataFrame({
                    'Date': quality_metrics['quality_trends']['date'],
                    'Quality Score': quality_metrics['quality_trends']['mean'],
                    'Standard Deviation': quality_metrics['quality_trends']['std']
                })
                st.markdown(get_download_link(report_data, 'quality_report.xlsx', 'Download Quality Report'), unsafe_allow_html=True)

def show_ml_predictions():
    st.header("ML Predictions")
    
    if st.session_state.data is not None and st.session_state.ml_model is not None:
        # Make predictions
        predictions = predict_quality(st.session_state.data)
        
        if predictions is not None:
            # Display prediction results
            st.subheader("Quality Predictions")
            
            # Prediction distribution
            fig = px.histogram(predictions['predictions'], 
                             title='Predicted Quality Distribution',
                             labels={'value': 'Predicted Quality', 'count': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction confidence
            st.subheader("Prediction Confidence")
            confidence_data = pd.DataFrame({
                'confidence': predictions['confidence']
            })
            fig = px.histogram(confidence_data, x='confidence',
                             title='Prediction Confidence Distribution',
                             labels={'confidence': 'Confidence Score', 'count': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            if hasattr(st.session_state.ml_model, 'feature_importances_'):
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'feature': ['temperature', 'pressure', 'speed', 'humidity'],
                    'importance': st.session_state.ml_model.feature_importances_
                })
                fig = px.bar(feature_importance, x='feature', y='importance',
                           title='Feature Importance',
                           labels={'feature': 'Feature', 'importance': 'Importance Score'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Export predictions
            st.subheader("Export Predictions")
            if st.button("Export Predictions"):
                prediction_data = pd.DataFrame({
                    'Predicted Quality': predictions['predictions'],
                    'Confidence': predictions['confidence']
                })
                st.markdown(get_download_link(prediction_data, 'ml_predictions.xlsx', 'Download Predictions'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
