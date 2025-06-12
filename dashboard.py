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
            # st.success(f"Successfully loaded {file}") # Removed logging
        except Exception as e:
            # st.warning(f"Error loading {file}: {str(e)}") # Removed logging
            pass # Keep pass to suppress errors silently during loading
    
    # Load Loss Time data
    loss_time_files = [f for f in os.listdir(csv_dir) if 'CCL_loss_time' in f]
    data['loss_time'] = {}
    for file in loss_time_files:
        try:
            data['loss_time'][file] = pd.read_csv(csv_dir / file)
            # st.success(f"Successfully loaded {file}") # Removed logging
        except Exception as e:
            # st.warning(f"Error loading {file}: {str(e)}") # Removed logging
            pass # Keep pass to suppress errors silently during loading
    
    # Load Capacity Study data
    capacity_files = [f for f in os.listdir(csv_dir) if 'Capacity_study' in f]
    data['capacity'] = {}
    for file in capacity_files:
        try:
            data['capacity'][file] = pd.read_csv(csv_dir / file, header=7)
            # st.success(f"Successfully loaded {file}") # Removed logging
        except Exception as e:
            # st.warning(f"Error loading {file}: {str(e)}") # Removed logging
            pass # Keep pass to suppress errors silently during loading
    
    # Load Quadrant data
    quadrant_files = [f for f in os.listdir(csv_dir) if 'Quadrant_data' in f]
    data['quadrant'] = {}
    for file in quadrant_files:
        try:
            data['quadrant'][file] = pd.read_csv(csv_dir / file)
            # st.success(f"Successfully loaded {file}") # Removed logging
        except Exception as e:
            # st.warning(f"Error loading {file}: {str(e)}") # Removed logging
            pass # Keep pass to suppress errors silently during loading
    
    return data

class MLPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.model_info = {}
    
    def train_production_prediction(self, data):
        try:
            # Define potential target columns for time data
            time_cols = ['Cycle Time(CT)', 'SMV', 'Time (min)']
            target_col = None
            for col in time_cols:
                if col in data.columns:
                    target_col = col
                    break
            
            if target_col is None:
                st.warning("Could not find a suitable time-related column for production prediction training. Looked for: " + ', '.join(time_cols))
                return None

            # Prepare features
            X = data[['Operation tack time', 'Req.Tack time']].dropna()
            y = data[target_col].dropna()
            
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
        st.markdown("""
    This demo predicts the production time based on operation tack time and required tack time.
    It uses a **Random Forest Regressor** model. The model learns the relationship between
    input features and actual production time from historical data.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        operation_time = st.number_input("Operation Tack Time (min)", min_value=0.0, value=1.0)
        req_time = st.number_input("Required Tack Time (min)", min_value=0.0, value=1.0)
    
    if st.button("Predict Production Time"):
        if 'production' in ml_predictor.models:
            prediction, model_info = ml_predictor.predict_production_time(operation_time, req_time)
            if prediction is not None:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Predicted Production Time: {prediction:.2f} minutes</h3>
                    <p>Calculation: The Random Forest Regressor combines predictions from multiple decision trees. Each tree considers the input features (Operation Tack Time, Required Tack Time) to estimate the production time. The final prediction is an average of all individual tree predictions.</p>
                </div>
                """, unsafe_allow_html=True)
                show_model_info(model_info)
            else:
                st.warning("Failed to get a production time prediction. Model might not be trained or input values are invalid.")
        else:
            st.warning("Production prediction model is not trained. Please ensure relevant data is available.")

def quality_live_demo(data, ml_predictor):
    st.subheader("Quality Prediction Demo")
    st.markdown("""
    This demo predicts the quality outcome (e.g., Good/Needs Improvement) based on operation, operator, and time.
    It uses a **Random Forest Classifier** model. This model classifies the quality based on patterns learned from historical data of operations, operators, and their associated defects.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Ensure 'Operation' column exists in some relevant data for selection
        # Using 'Operation' from the first capacity file for demonstration purposes
        operations = []
        if 'capacity' in data and data['capacity']:
            first_capacity_file_key = list(data['capacity'].keys())[0]
            if 'Operation' in data['capacity'][first_capacity_file_key].columns:
                operations = data['capacity'][first_capacity_file_key]['Operation'].dropna().unique().tolist()
        
        if operations:
            operation = st.selectbox("Operation", operations)
        else:
            operation = st.text_input("Operation (e.g., Sewing)")
            st.warning("'Operation' column not found in capacity data. Please enter manually.")

    with col2:
        operator = st.text_input("Operator ID")
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
                        <p>Calculation: The Random Forest Classifier uses input features (Operation, Operator, Time) to classify the outcome. Each tree in the forest votes on the quality category, and the final prediction is based on the majority vote.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    show_model_info(model_info)
                else:
                    st.warning("Failed to get a quality prediction. Model might not be trained or input values are invalid.")
            except Exception as e:
                st.error(f"An error occurred during quality prediction: {str(e)}")
        else:
            st.warning("Quality prediction model is not trained. Please ensure relevant data is available.")

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
        fig = px.pie(df, names='Material', title='Material Type Distribution')
        st.plotly_chart(fig)
        
        # Stock level trends
        if 'Updated Date' in df.columns:
            df['Updated Date'] = pd.to_datetime(df['Updated Date'])
            stock_trend = df.groupby('Updated Date')['Stock Qty'].sum().reset_index()
            fig = px.line(stock_trend, x='Updated Date', y='Stock Qty', title='Stock Level Trends')
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
            # Define potential target columns for time data
            time_cols = ['Cycle Time(CT)', 'SMV', 'Time (min)']
            selected_time_col = None
            for col in time_cols:
                if col in df.columns:
                    selected_time_col = col
                    break
            
            if selected_time_col is not None:
                avg_time = df[selected_time_col].mean()
                st.metric("Average Time", f"{avg_time:.2f} min")
            else:
                st.warning("Could not find a suitable time-related column for average time calculation. Looked for: " + ', '.join(time_cols))
                avg_time = None
        with col3:
            if selected_time_col is not None:
                total_time = df[selected_time_col].sum()
                st.metric("Total Time", f"{total_time:.2f} min")
            else:
                total_time = None
        
        # Operation time distribution
        st.subheader("Operation Time Distribution")
        if selected_time_col is not None:
            fig = px.histogram(df, x=selected_time_col, title=f'Operation Time Distribution ({selected_time_col})')
            st.plotly_chart(fig)
        else:
            st.info("Cannot display Operation Time Distribution: no suitable time column found.")
        
        # Bottleneck analysis
        st.subheader("Bottleneck Analysis")
        if selected_time_col is not None and avg_time is not None:
            bottlenecks = df[df[selected_time_col] > avg_time * 1.5] 
            if not bottlenecks.empty:
                st.dataframe(bottlenecks[['Operation', selected_time_col]]) 
            else:
                st.info("No significant bottlenecks identified.")
        else:
            st.info("Cannot perform Bottleneck Analysis: no suitable time column or average time found.")
        
        # Line balancing visualization
        st.subheader("Line Balancing")
        if selected_time_col is not None and avg_time is not None:
            fig = px.bar(df, x='Operation', y=selected_time_col, title='Operation Times') 
            fig.add_hline(y=avg_time, line_dash="dash", line_color="red", 
                         annotation_text="Average Time")
            st.plotly_chart(fig)
        else:
            st.info("Cannot display Line Balancing: no suitable time column or average time found.")

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

def worker_performance_page(data):
    st.title("Worker Performance Analysis")

    if 'quadrant' in data:
        # Select Quadrant data file
        competency_file = 'Quadrant_data_-_AI__Competency_Matrix.csv'
        quadrant_details_file = 'Quadrant_data_-_AI__Quadrant_details.csv'

        if competency_file in data['quadrant'] and quadrant_details_file in data['quadrant']:
            df_competency = data['quadrant'][competency_file]
            df_details = data['quadrant'][quadrant_details_file]

            st.subheader("Competency Matrix Data Overview")
            st.dataframe(df_competency.head())

            st.subheader("Quadrant Details Data Overview")
            st.dataframe(df_details.head())

            # Example: Visualize skill distribution (requires appropriate column names)
            st.subheader("Skill Distribution (Example)")
            if 'Skill' in df_competency.columns:
                skill_dist = df_competency['Skill'].value_counts().reset_index()
                skill_dist.columns = ['Skill', 'Count']
                fig = px.bar(skill_dist, x='Skill', y='Count', title='Distribution of Skills')
                st.plotly_chart(fig)
            else:
                st.info("'Skill' column not found in Competency Matrix for visualization.")

            st.subheader("Insights and ML Algorithm Suggestions")
            st.markdown("""
            Based on the Quadrant data, we can infer insights related to worker skill levels, training needs, and overall workforce capabilities.
            
            **Applicable Machine Learning Algorithms:**
            -   **Classification:** To categorize workers into different performance tiers or identify those who might need specific training.
            -   **Clustering:** To group workers with similar skill sets or performance patterns, which can aid in team formation and targeted development programs.
            
            **How it helps the Garment Industry:**
            -   **Optimized Training Programs:** Identify specific skill gaps and tailor training to maximize impact.
            -   **Improved Resource Allocation:** Place the right workers in the right operations based on their strengths.
            -   **Performance Benchmarking:** Establish internal benchmarks for different roles and identify high-performers.
            -   **Succession Planning:** Identify potential leaders and prepare them for future roles.
            """)

        else:
            st.warning(f"Required Quadrant data files ({competency_file}, {quadrant_details_file}) not found.")

def worker_allocation_page():
    st.title("Worker Allocation")

    st.markdown("""
    This section focuses on optimizing worker deployment based on various factors such as skill, performance, and task requirements.
    This aims to maximize productivity and minimize idle time.

    ### ML Algorithms:
    For real-world worker allocation, this can involve complex optimization algorithms (e.g., Linear Programming, Integer Programming)
    to find the best assignment of workers to tasks under various constraints. More advanced scenarios might use Reinforcement Learning
    to dynamically adapt allocations based on real-time feedback. Predictive models (e.g., regression) could estimate task completion times.

    ### How Workers are Allocated (Simulated Logic):
    In this simulation, workers are assigned to tasks based on a simple matching algorithm. Tasks are prioritized (High > Medium > Low),
    and then by required skill level. Workers are selected based on meeting or exceeding the required skill level, with higher performance
    workers being prioritized first. Once a worker is assigned, they are no longer available for other tasks.

    ### Features Used:
    -   **Worker Data:** 'Worker ID', 'Skill Level (1-10)', 'Experience (Years)', 'Current Performance (%)', 'Task Preference'.
    -   **Task Data:** 'Task ID', 'Required Skill Level (1-10)', 'Estimated Time (hours)', 'Priority'.

    ### Further Predictions/Optimizations:
    -   **Dynamic Re-allocation:** Predict the need for re-allocating workers based on real-time production changes or unexpected delays.
    -   **Fatigue Prediction:** Use sensor data or historical work patterns to predict worker fatigue and optimize breaks or shift rotations.
    -   **Skill Gap Identification:** Analyze allocation results to identify skill gaps in the workforce and suggest targeted training programs.
    """)

    st.subheader("Simulate Worker Allocation")

    num_workers = st.number_input("Number of Workers", min_value=10, max_value=500, value=50, step=10)
    num_tasks = st.number_input("Number of Tasks", min_value=5, max_value=100, value=20, step=5)

    if st.button("Run Simulation"):
        # Generate sample worker data
        worker_data = {
            'Worker ID': [f'W{i+1:03d}' for i in range(num_workers)],
            'Skill Level (1-10)': np.random.randint(1, 11, num_workers),
            'Experience (Years)': np.random.randint(1, 20, num_workers),
            'Current Performance (%)': np.random.randint(70, 101, num_workers),
            'Task Preference': np.random.choice(['Cutting', 'Sewing', 'Finishing', 'Packing'], num_workers)
        }
        df_workers = pd.DataFrame(worker_data)

        # Generate sample task data
        task_data = {
            'Task ID': [f'T{i+1:03d}' for i in range(num_tasks)],
            'Required Skill Level (1-10)': np.random.randint(1, 11, num_tasks),
            'Estimated Time (hours)': np.round(np.random.uniform(1, 8, num_tasks), 1),
            'Priority': np.random.choice(['High', 'Medium', 'Low'], num_tasks, p=[0.3, 0.4, 0.3])
        }
        df_tasks = pd.DataFrame(task_data)

        st.subheader("Generated Worker Data (Sample)")
        st.dataframe(df_workers.head())

        st.subheader("Generated Task Data (Sample)")
        st.dataframe(df_tasks.head())

        st.subheader("Worker Allocation Results")

        # Correlation analysis (example)
        st.subheader("Correlation Analysis of Worker Data")
        worker_numeric_cols = ['Skill Level (1-10)', 'Experience (Years)', 'Current Performance (%)']
        if not df_workers[worker_numeric_cols].empty:
            fig_corr = px.imshow(df_workers[worker_numeric_cols].corr(),
                                  text_auto=True,
                                  color_continuous_scale='Viridis',
                                  title='Worker Data Correlation Heatmap')
            st.plotly_chart(fig_corr)
        else:
            st.info("Not enough data to generate worker correlation heatmap.")

        # Worker Data Distributions
        st.subheader("Worker Data Distributions")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            fig_skill_dist = px.histogram(df_workers, x='Skill Level (1-10)', title='Worker Skill Level Distribution')
            st.plotly_chart(fig_skill_dist)
        with col_w2:
            fig_perf_dist = px.histogram(df_workers, x='Current Performance (%)', title='Worker Performance Distribution')
            st.plotly_chart(fig_perf_dist)

        # Task Data Distributions
        st.subheader("Task Data Distributions")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_req_skill_dist = px.histogram(df_tasks, x='Required Skill Level (1-10)', title='Required Skill Level Distribution')
            st.plotly_chart(fig_req_skill_dist)
        with col_t2:
            fig_est_time_dist = px.histogram(df_tasks, x='Estimated Time (hours)', title='Estimated Task Time Distribution')
            st.plotly_chart(fig_est_time_dist)
        
        fig_priority_dist = px.pie(df_tasks, names='Priority', title='Task Priority Distribution')
        st.plotly_chart(fig_priority_dist)

        st.subheader("Detailed ML Algorithms for Worker Allocation")
        st.markdown("""
        Beyond the simulated logic, real-world worker allocation can leverage various ML algorithms:
        -   **Clustering (e.g., K-Means, DBSCAN):** To segment workers into groups based on their skills, experience, and performance.
            This helps in understanding the workforce composition and identifying natural teams or skill gaps.
            *Example: Grouping workers with similar skill sets for specialized tasks.*
        -   **Regression (e.g., Linear Regression, Random Forest Regressor):** To predict task completion times based on task complexity, worker skill, and historical data.
            This helps in more accurate scheduling and workload balancing.
            *Example: Predicting how long a specific sewing operation will take based on the assigned worker's skill and experience.*
        -   **Optimization Algorithms (e.g., Linear Programming, Integer Programming):** While not strictly ML, these are often used in conjunction with ML predictions.
            They find the optimal assignment of workers to tasks to minimize idle time, maximize output, or balance workload, given various constraints.
            *Example: Assigning available workers to production lines to meet a daily target while minimizing overtime.*
        -   **Reinforcement Learning (e.g., Q-Learning, Deep Q-Networks):** For dynamic allocation scenarios where the environment changes (e.g., worker absenteeism, machine breakdown).
            An RL agent can learn optimal allocation policies through trial and error, adapting to real-time feedback.
            *Example: Dynamically re-assigning workers when a machine breaks down to minimize disruption and maintain production flow.*
        """)

        st.subheader("Simulated Worker Allocation")

        # Sort tasks by priority (High > Medium > Low) and then by required skill level
        priority_order = {'High': 3, 'Medium': 2, 'Low': 1}
        df_tasks['Priority_Rank'] = df_tasks['Priority'].map(priority_order)
        df_tasks = df_tasks.sort_values(by=['Priority_Rank', 'Required Skill Level (1-10)'], ascending=[False, False])
            
            allocated_tasks = []
        available_workers = df_workers.copy()

        for index, task in df_tasks.iterrows():
                suitable_workers = available_workers[
                (available_workers['Skill Level (1-10)'] >= task['Required Skill Level (1-10)'])
                ].sort_values(by='Current Performance (%)', ascending=False)

                if not suitable_workers.empty:
                chosen_worker = suitable_workers.iloc[0]
                allocated_tasks.append({
                    'Task ID': task['Task ID'],
                    'Worker ID': chosen_worker['Worker ID'],
                    'Task Priority': task['Priority'],
                    'Task Required Skill': task['Required Skill Level (1-10)'],
                    'Worker Skill': chosen_worker['Skill Level (1-10)'],
                    'Worker Performance': chosen_worker['Current Performance (%)'],
                    'Estimated Time (hours)': task['Estimated Time (hours)']
                })
                # Remove allocated worker from available list
                available_workers = available_workers[available_workers['Worker ID'] != chosen_worker['Worker ID']]
            else:
                    allocated_tasks.append({
                        'Task ID': task['Task ID'],
                    'Worker ID': 'N/A',
                        'Task Priority': task['Priority'],
                    'Task Required Skill': task['Required Skill Level (1-10)'],
                    'Worker Skill': 'N/A',
                    'Worker Performance': 'N/A',
                    'Estimated Time (hours)': task['Estimated Time (hours)']
                })

        df_allocated = pd.DataFrame(allocated_tasks)
        st.dataframe(df_allocated)

        # Visualization of Allocation Results
        st.subheader("Allocation Summary")
        allocated_count = len(df_allocated[df_allocated['Worker ID'] != 'N/A'])
        unallocated_count = len(df_allocated[df_allocated['Worker ID'] == 'N/A'])

        col_alloc1, col_alloc2 = st.columns(2)
        with col_alloc1:
            st.metric("Allocated Tasks", allocated_count)
        with col_alloc2:
            st.metric("Unallocated Tasks", unallocated_count)

        fig_allocation = px.bar(df_allocated, x='Task ID', y='Estimated Time (hours)',
                                color='Worker ID', title='Task Allocation by Worker',
                                labels={'Worker ID': 'Assigned Worker'})
        st.plotly_chart(fig_allocation)

        unallocated_tasks = df_allocated[df_allocated['Worker ID'] == 'N/A']
        if not unallocated_tasks.empty:
            st.warning(f"{len(unallocated_tasks)} tasks could not be allocated due to insufficient suitable workers.")
            st.dataframe(unallocated_tasks)
        else:
            st.success("All tasks allocated successfully!")

def main():
    # Load data
    data = load_csv_data()
    
    # Initialize ML predictor
    ml_predictor = MLPredictor()
    
    # Train models if data is available
    if 'capacity' in data:
        st.subheader("Training Machine Learning Models...")
        production_model_trained = False
        for file, df in data['capacity'].items():
            if 'Operation tack time' in df.columns:
                # Check for any of the potential time columns
                time_cols = ['Cycle Time(CT)', 'SMV', 'Time (min)']
                found_time_col = False
                for col in time_cols:
                    if col in df.columns:
                        found_time_col = True
                        break

                if found_time_col:
                    st.info(f"Attempting to train Production Prediction model using {file}...")
                    score = ml_predictor.train_production_prediction(df)
                    if score is not None:
                        st.success(f"Production Prediction model trained successfully with score: {score:.4f}")
                        production_model_trained = True
                    else:
                        st.warning(f"Failed to train Production Prediction model using {file}.")
                else:
                    st.warning(f"Required time column (any of {', '.join(time_cols)}) not found in {file} for Production Prediction training.")
            else:
                st.warning(f"Required column 'Operation tack time' not found in {file} for Production Prediction training.")

        if not production_model_trained:
            st.error("Production Prediction model could not be trained due to missing required data in any capacity file.")
    else:
        st.warning("No capacity data found for Production Prediction model training.")

    # Train quality model if data is available
    if 'loss_time' in data:
        quality_model_trained = False
        for file, df in data['loss_time'].items():
            # Assuming 'Operation', 'Operator', 'Time', 'Defects' are critical for quality training
            required_quality_cols = ['Operation', 'Operator', 'Time', 'Defects']
            if all(col in df.columns for col in required_quality_cols):
                st.info(f"Attempting to train Quality Prediction model using {file}...")
                accuracy = ml_predictor.train_quality_prediction(df)
                if accuracy is not None:
                    st.success(f"Quality Prediction model trained successfully with accuracy: {accuracy:.4f}")
                    quality_model_trained = True
                else:
                    st.warning(f"Failed to train Quality Prediction model using {file}.")
            else:
                st.warning(f"Required columns ({', '.join(required_quality_cols)}) not found in {file} for Quality Prediction training.")
        
        if not quality_model_trained:
            st.error("Quality Prediction model could not be trained due to missing required data in any loss time file.")
    else:
        st.warning("No loss time data found for Quality Prediction model training.")

    # Navigation
st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Live Demos", "Production", "Quality", "Worker Performance", "Worker Allocation"])
    
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
        - **Worker Performance Analysis:** Skill distribution and performance insights
        - **Worker Allocation:** Optimized worker deployment
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
    
    elif page == "Worker Performance":
        worker_performance_page(data)
    
    elif page == "Worker Allocation":
        worker_allocation_page()

if __name__ == "__main__":
    main()