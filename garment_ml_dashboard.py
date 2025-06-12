import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

st.set_page_config(page_title="Garment Industry ML Dashboard", layout="wide")

# --- Data Loading ---
csv_dir = 'csv'
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

st.sidebar.title("Garment Industry ML Dashboard")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "📊 Explore Datasets",
    "💡 Dataset Info & ML Use Cases",
    "🤖 ML Live Demos"
])

# --- Home Page ---
def home():
    st.title("Garment Industry ML Dashboard")
    st.markdown("""
    Welcome to the unified dashboard for garment industry analytics and machine learning demos.
    - **Explore Datasets**: Browse, filter, and visualize your data.
    - **Dataset Info & ML Use Cases**: Learn about each dataset, its features, and possible ML applications.
    - **ML Live Demos**: Try out multiple ML algorithms live on your data!
    """)
    st.info("All ML runs are performed in your virtual environment.")

# --- Dataset Exploration Page ---
def explore_datasets():
    st.title("Explore Datasets")
    selected_csv = st.selectbox("Choose a dataset", csv_files)
    df = pd.read_csv(os.path.join(csv_dir, selected_csv))
    st.header(f"Dataset: {selected_csv}")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    with st.expander("Show Columns and Types"):
        st.write(pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str)}))
    with st.expander("Show First 10 Rows"):
        st.dataframe(df.head(10))
    with st.expander("Show Basic Statistics"):
        st.write(df.describe(include='all'))

# --- Dataset Info & ML Use Cases Page ---
def dataset_info():
    st.title("Dataset Info & ML Use Cases")
    selected_csv = st.selectbox("Choose a dataset", csv_files)
    df = pd.read_csv(os.path.join(csv_dir, selected_csv))
    st.header(f"Dataset: {selected_csv}")
    st.write("### Columns:")
    st.write(list(df.columns))
    st.write("### What can it be used to predict?")
    st.write("- Predict production delays, fabric needs, operator performance, quality issues, etc., depending on the dataset.")
    st.write("### ML Algorithms Suitable for This Data:")
    st.markdown("""
- **Regression (Linear, Ridge, Random Forest):** Predict continuous variables like production time, fabric consumption, or defect rates.
- **Classification (Logistic Regression, Random Forest):** Predict categorical outcomes such as delay reason, defect type, or operator performance class.
- **Clustering (KMeans):** Group similar products, operators, or defect patterns.
- **Time Series Forecasting:** Predict future production/output (with appropriate time-indexed data).
- **Anomaly Detection:** Identify unusual production or quality events.
    """)
    st.write("### Example ML Applications:")
    st.markdown("""
- **Production Forecasting**
- **Quality Control/Defect Prediction**
- **Inventory Optimization**
- **Operator Performance Classification**
- **Worker Allocation Optimization**
    """)

# --- ML Live Demo Page ---
def ml_live_demo():
    st.title("ML Live Demos")
    selected_csv = st.selectbox("Choose a dataset for ML demo", csv_files)
    df = pd.read_csv(os.path.join(csv_dir, selected_csv))
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns for ML demo.")
        return
    st.subheader("Select ML Task")
    task = st.radio("Task", ["Regression (Predict a Number)", "Classification (Predict a Category)", "Clustering (Find Groups)"])
    if task == "Regression (Predict a Number)":
        target = st.selectbox("Target column to predict", numeric_cols)
        features = st.multiselect("Features to use", [col for col in numeric_cols if col != target], default=[col for col in numeric_cols if col != target])
        if features:
            X = df[features].dropna()
            y = df[target].loc[X.index]
            if len(X) > 20:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model_name = st.selectbox("Algorithm", ["Linear Regression", "Ridge Regression", "Random Forest Regressor"])
                if model_name == "Linear Regression":
                    model = LinearRegression()
                elif model_name == "Ridge Regression":
                    model = Ridge()
                else:
                    model = RandomForestRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                st.success(f"Test RMSE: {mean_squared_error(y_test, preds, squared=False):.3f}")
                st.write("Predictions for first 5 test rows:", preds[:5])
            else:
                st.warning("Not enough data to train a regression model.")
    elif task == "Classification (Predict a Category)":
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not cat_cols:
            st.warning("No categorical columns found for classification.")
            return
        target = st.selectbox("Target column to classify", cat_cols)
        feature_candidates = [col for col in numeric_cols if col != target]
        features = st.multiselect("Features to use", feature_candidates, default=feature_candidates)
        if features:
            X = df[features].dropna()
            y = df[target].loc[X.index]
            if len(X) > 20 and y.nunique() > 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model_name = st.selectbox("Algorithm", ["Logistic Regression", "Random Forest Classifier"])
                if model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=500)
                else:
                    model = RandomForestClassifier()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                st.success(f"Test Accuracy: {accuracy_score(y_test, preds):.3f}")
                st.text(classification_report(y_test, preds))
                st.write("Predictions for first 5 test rows:", preds[:5])
            else:
                st.warning("Not enough data or only one class for classification.")
    else:
        # Clustering
        n_clusters = st.slider("Number of clusters", 2, min(10, len(df)), 3)
        features = st.multiselect("Features to use for clustering", numeric_cols, default=numeric_cols)
        if features:
            X = df[features].dropna()
            if len(X) > n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(X)
                st.write(f"Cluster centers:\n{kmeans.cluster_centers_}")
                st.write("Cluster label counts:", np.bincount(labels))
                df_demo = X.copy()
                df_demo['Cluster'] = labels
                st.dataframe(df_demo.head(10))
            else:
                st.warning("Not enough data for clustering.")

# --- Page Routing ---
if page == "🏠 Home":
    home()
elif page == "📊 Explore Datasets":
    explore_datasets()
elif page == "💡 Dataset Info & ML Use Cases":
    dataset_info()
else:
    ml_live_demo()
