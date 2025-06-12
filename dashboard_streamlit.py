import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Garment Industry Data Explorer", layout="wide")

csv_dir = 'csv'
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

st.title("Garment Industry Dataset Dashboard")

st.sidebar.header("Datasets")
selected_csv = st.sidebar.selectbox("Choose a dataset", csv_files)

# Load the selected dataset
df = pd.read_csv(os.path.join(csv_dir, selected_csv))

st.header(f"Dataset: {selected_csv}")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

with st.expander("Show Columns and Types"):
    st.write(pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str)}))

with st.expander("Show First 10 Rows"):
    st.dataframe(df.head(10))

with st.expander("Show Basic Statistics"):
    st.write(df.describe(include='all'))

# ML/Prediction suggestion page
st.sidebar.markdown("---")
if st.sidebar.button("Go to ML Application & Prediction"):
    st.session_state['show_ml'] = True

if st.session_state.get('show_ml', False):
    st.header("ML Application & Prediction (Demo)")
    st.info("This section will show a demo ML prediction for the selected dataset. For a real project, you would select features and train a model.")
    # Simple demo: For numeric columns, fit a linear regression to predict the last column from the others (if possible)
    from sklearn.linear_model import LinearRegression
    import numpy as np
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 1:
        X = df[numeric_cols[:-1]].dropna()
        y = df[numeric_cols[-1]].loc[X.index]
        if len(X) > 10:
            model = LinearRegression()
            model.fit(X, y)
            st.success(f"Trained Linear Regression to predict `{numeric_cols[-1]}` from other numeric columns.")
            st.write("Model coefficients:", dict(zip(numeric_cols[:-1], model.coef_)))
            st.write("Intercept:", model.intercept_)
            # Predict for the first 5 rows
            preds = model.predict(X.head(5))
            st.write("Predictions for first 5 rows:", preds)
        else:
            st.warning("Not enough numeric data to train a model.")
    else:
        st.warning("Not enough numeric columns for ML demo.")

# Info page for each dataset
st.sidebar.markdown("---")
if st.sidebar.button("Show Dataset Purpose & ML Ideas"):
    st.session_state['show_info'] = True

if st.session_state.get('show_info', False):
    st.header("Dataset Parameters, Use Cases, and ML Applications")
    st.write("### Columns:")
    st.write(list(df.columns))
    st.write("### What can it be used to predict?")
    st.write("- This depends on the business problem. For example, you can predict production delays, fabric requirements, operator performance, stock shortages, etc.")
    st.write("### Example ML Applications:")
    st.markdown("""
    - **Production Forecasting:** Predict future production or delays using historical data.
    - **Quality Control:** Detect defects or predict quality issues from process data.
    - **Inventory Optimization:** Predict fabric/material needs to minimize shortages and overstock.
    - **Operator Performance:** Classify or predict operator efficiency/quadrant.
    """)
    st.write("For a specific ML workflow, please select a dataset and let me know your business question!")
