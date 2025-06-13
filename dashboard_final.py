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
            # Check if data has the required columns
            required_cols = ['Operation tack time', 'Req.Tack time', 'Time (min)']
            available_cols = data.columns.tolist()

            # Find alternative column names for time data
            time_cols = [col for col in available_cols if 'time' in col.lower() or 'min' in col.lower()]

            if not any(col in available_cols for col in required_cols):
                st.warning(f"Required columns not found. Available columns: {available_cols}")
                return None

            # Try to find suitable columns
            feature_cols = [col for col in ['Operation tack time', 'Req.Tack time'] if col in available_cols]
            target_col = None

            for col in ['Time (min)', 'Time  (min)'] + time_cols:
                if col in available_cols:
                    target_col = col
                    break

            if len(feature_cols) < 2 or target_col is None:
                st.warning("Insufficient data columns for production prediction training")
                return None

            # Prepare features
            X = data[feature_cols].dropna()
            y = data[target_col].dropna()

            # Align X and y indices
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]

            if len(X) > 10 and len(y) > 10:
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
                    'target': target_col
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

    st.info("Quality prediction demo - This feature requires training data with Operation, Operator, and Defects columns.")

    col1, col2, col3 = st.columns(3)
    with col1:
        # Get available operations from capacity data
        operations = ["Sewing", "Cutting", "Pressing", "Finishing"]  # Default options
        if 'capacity' in data and data['capacity']:
            for filename, df in data['capacity'].items():
                if 'Operation' in df.columns:
                    operations = df['Operation'].dropna().unique().tolist()
                    break
        operation = st.selectbox("Operation", operations)
    with col2:
        operator = st.text_input("Operator ID", value="OP001")
    with col3:
        time = st.number_input("Operation Time (min)", min_value=0.0, value=1.0)

    if st.button("Predict Quality"):
        if 'quality' in ml_predictor.models:
            try:
                prediction, model_info = ml_predictor.predict_quality(operation, operator, time)
                if prediction is not None:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Predicted Quality: {'Good' if prediction == 1 else 'Needs Improvement'}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    show_model_info(model_info)
                else:
                    st.warning("Could not generate prediction with current inputs.")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("Quality prediction model is not trained. Training requires data with Operation, Operator, Time, and Defects columns.")

def inventory_live_demo(data):
    st.subheader("Inventory Analysis Demo")

    if 'stock_entry' in data and data['stock_entry'] is not None:
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

        # Show data sample
        st.subheader("Data Sample")
        st.dataframe(df.head(10))

        # Find suitable columns for visualization
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = [col for col in df.columns if 'date' in col.lower()]

        # Material type distribution
        if categorical_cols:
            material_col = categorical_cols[0]  # Use first categorical column
            if 'material' in material_col.lower() or len(categorical_cols) == 1:
                st.subheader("Material Distribution")
                fig = px.pie(df, names=material_col, title=f'{material_col} Distribution')
                st.plotly_chart(fig)

        # Stock level trends
        if date_cols and numeric_cols:
            try:
                st.subheader("Trends Over Time")
                date_col = date_cols[0]
                qty_col = numeric_cols[0]  # Use first numeric column as quantity

                df_trend = df.copy()
                df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
                df_trend = df_trend.dropna(subset=[date_col])

                if not df_trend.empty:
                    stock_trend = df_trend.groupby(date_col)[qty_col].sum().reset_index()
                    fig = px.line(stock_trend, x=date_col, y=qty_col, title=f'{qty_col} Trends Over Time')
                    st.plotly_chart(fig)
            except Exception as e:
                st.info(f"Could not create trend analysis: {str(e)}")

        # Summary statistics
        if numeric_cols:
            st.subheader("Summary Statistics")
            st.dataframe(df[numeric_cols].describe())
    else:
        st.warning("No stock entry data available for inventory analysis.")

def production_optimization_page(data):
    st.title("Production Optimization")

    if 'capacity' in data and data['capacity']:
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

        # Find numeric columns for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        time_related_cols = [col for col in df.columns if 'time' in col.lower() or 'min' in col.lower() or 'cycle' in col.lower()]

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_operations = len(df)
            st.metric("Total Operations", total_operations)

        # Try to find time-related data
        if numeric_cols:
            with col2:
                # Use first numeric column as proxy for time data
                time_col = numeric_cols[0]
                if time_related_cols:
                    time_col = time_related_cols[0]

                avg_time = df[time_col].mean()
                st.metric(f"Average {time_col}", f"{avg_time:.2f}")
            with col3:
                total_time = df[time_col].sum()
                st.metric(f"Total {time_col}", f"{total_time:.2f}")

            # Operation time distribution
            st.subheader("Time Distribution")
            fig = px.histogram(df, x=time_col, title=f'{time_col} Distribution')
            st.plotly_chart(fig)

            # Show data sample
            st.subheader("Data Sample")
            st.dataframe(df.head(10))
        else:
            with col2:
                st.metric("Numeric Columns", len(numeric_cols))
            with col3:
                st.metric("Text Columns", len(df.columns) - len(numeric_cols))

            # Show data sample
            st.subheader("Data Sample")
            st.dataframe(df.head(10))
            st.info("No numeric time data found for detailed analysis.")
    else:
        st.warning("No capacity data available for analysis.")

def quality_control_page(data):
    st.title("Quality Control")

    if 'loss_time' in data and data['loss_time']:
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

        # Find relevant columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'min' in col.lower()]
        date_cols = [col for col in df.columns if 'date' in col.lower()]

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_records = len(df)
            st.metric("Total Records", total_records)

        if time_cols and numeric_cols:
            with col2:
                time_col = time_cols[0]
                if time_col in numeric_cols:
                    total_time = df[time_col].sum()
                    st.metric(f"Total {time_col}", f"{total_time:.2f}")

        if numeric_cols:
            with col3:
                # Use first numeric column as a metric
                metric_col = numeric_cols[0]
                total_value = df[metric_col].sum()
                st.metric(f"Total {metric_col}", f"{total_value:.2f}")

        # Show data sample
        st.subheader("Data Sample")
        st.dataframe(df.head(10))

        # Create visualizations if we have suitable data
        if numeric_cols:
            st.subheader("Data Distribution")
            fig = px.histogram(df, x=numeric_cols[0], title=f'{numeric_cols[0]} Distribution')
            st.plotly_chart(fig)

        # Time trend if date column exists
        if date_cols and numeric_cols:
            try:
                st.subheader("Trend Analysis")
                date_col = date_cols[0]
                value_col = numeric_cols[0]
                df_trend = df.copy()
                df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
                df_trend = df_trend.dropna(subset=[date_col])

                if not df_trend.empty:
                    trend_data = df_trend.groupby(date_col)[value_col].sum().reset_index()
                    fig = px.line(trend_data, x=date_col, y=value_col, title=f'{value_col} Trend Over Time')
                    st.plotly_chart(fig)
            except Exception as e:
                st.info(f"Could not create trend analysis: {str(e)}")
    else:
        st.warning("No loss time data available for analysis.")

def main():
    # Load data
    data = load_csv_data()
    
    # Initialize ML predictor
    ml_predictor = MLPredictor()
    
    # Train models if data is available
    if 'capacity' in data and data['capacity']:
        for filename, df in data['capacity'].items():
            # Try to train production prediction with available data
            try:
                result = ml_predictor.train_production_prediction(df)
                if result is not None:
                    st.success(f"Production model trained successfully with {filename}")
                    break  # Only train with first successful dataset
            except Exception as e:
                continue  # Try next dataset
    
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