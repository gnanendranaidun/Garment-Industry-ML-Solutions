import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from production_optimization import ProductionOptimizer
from loss_time_analysis import LossTimeAnalyzer
from quality_control import QualityController
from product_analysis import ProductAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import random

# Set page config
st.set_page_config(
    page_title="Garment Industry Analytics Dashboard",
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
    </style>
""", unsafe_allow_html=True)

# --- Page Functions ---
# Define all page functions first so they are available when `pages` dict is created

def home_page():
    st.title("Garment Industry ML Dashboard")
    st.markdown("""
    Welcome to the Garment Industry ML Dashboard. This application provides insights and recommendations
    across various aspects of garment manufacturing using machine learning.

    Use the navigation sidebar on the left to explore different modules:
    - **Production Optimization:** Analyze and optimize production line balancing and efficiency.
    - **Loss Time Analysis:** Identify and address critical loss time patterns.
    - **Quality Control:** Monitor and improve product quality through defect analysis.
    - **Product Analysis:** Understand product performance and operator competency.
    - **Worker Allocation:** Simulate and optimize worker assignments for tasks.
    """)

    st.subheader("Overall Data Status")
    # Display a summary of loaded data or data issues
    # This part can be enhanced to show which files loaded successfully and which had errors
    st.info("Data loading status will be displayed here once data is loaded.")


def production_optimization_page(analyzers):
    st.title("Production Optimization")
    st.markdown("""
    ### Understanding Production Optimization
    This module helps in analyzing and optimizing the garment production process, focusing on line balancing,
    operation efficiency, and bottleneck identification.
    **ML Algorithms:** While not explicitly using complex ML models like deep learning, the optimization process can be framed as an optimization problem (e.g., using Integer Linear Programming or heuristic algorithms). For predictive maintenance of machines, regression models could be employed. For worker performance prediction, historical data can be used with regression models to forecast future efficiency.
    **Further Predictions/Optimizations:**
    - **Predictive Maintenance:** Predict machine breakdowns based on operational data to schedule maintenance proactively.
    - **Dynamic Line Rebalancing:** Automatically adjust line configurations in real-time based on fluctuating demand or operator availability.
    - **Optimal Buffer Stock:** Predict optimal buffer stock levels between operations to minimize idle time and maximize flow.
    """)

    production_data = analyzers.get('production', {}).get('data', {})
    production_optimizer = analyzers.get('production', {}).get('analyzer')

    if production_data:
        st.subheader("Operation Statistics")
        st.markdown("""
        **Analysis:** This section provides descriptive statistics for operations, identifying maximum, minimum,
        and average times, which helps in understanding performance variability.
        **Features Used:** All numeric columns representing operation times (e.g., 'Operation tack time', 'Req.Tack time', 'Time (min)').
        **Calculation Logic:** Pandas `describe()` function on numeric columns.
        """)

        # Display statistics for each sheet
        for sheet_name, df in production_data.items():
            st.markdown(f"#### Sheet: {sheet_name}")
            if df is not None and not df.empty:
                # Dynamically find timing columns
                timing_cols = [col for col in df.columns if 'time' in col.lower() or 'tack' in col.lower() or 'min' in col.lower()]
                timing_cols = [col for col in timing_cols if pd.api.types.is_numeric_dtype(df[col])]

                if not timing_cols:
                    st.info(f"No numeric timing columns found in {sheet_name}.")
                    continue

                st.dataframe(df[timing_cols].describe())
            else:
                st.info(f"No data or empty data for sheet: {sheet_name}.")

        st.subheader("Bottleneck Identification")
        st.markdown("""
        **Analysis:** Identifies operations with the highest processing times, which are potential bottlenecks.
        **Features Used:** Operation times.
        **Calculation Logic:** Maximum value of each timing column, sorted in descending order.
        """)

        for sheet_name, df in production_data.items():
            st.markdown(f"#### Sheet: {sheet_name}")
            if df is not None and not df.empty:
                timing_cols = [col for col in df.columns if 'time' in col.lower() or 'tack' in col.lower() or 'min' in col.lower()]
                timing_cols = [col for col in timing_cols if pd.api.types.is_numeric_dtype(df[col])]

                if not timing_cols:
                    st.info(f"No numeric timing columns found in {sheet_name} for bottleneck analysis.")
                    continue

                # Ensure relevant columns are present before proceeding
                required_cols = [col for col in ['Operation', 'Operation tack time', 'Req.Tack time'] if col in df.columns]

                if not required_cols:
                    st.warning(f"Skipping bottleneck analysis for sheet {sheet_name}: Required columns like 'Operation', 'Operation tack time', 'Req.Tack time' not found.")
                    continue
                
                # Dynamically check and use existing columns for bottleneck identification
                cols_to_check = [col for col in ['Operation tack time', 'Req.Tack time'] if col in df.columns]
                if not cols_to_check:
                    st.info(f"No valid tack time columns found for bottleneck analysis in {sheet_name}.")
                    continue

                # Drop rows with NaN in the selected timing columns
                df_cleaned = df.dropna(subset=cols_to_check)
                if df_cleaned.empty:
                    st.info(f"No valid data after dropping NaNs for bottleneck analysis in {sheet_name}.")
                    continue

                # Convert relevant columns to numeric, coercing errors
                for col in cols_to_check:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                df_cleaned = df_cleaned.dropna(subset=cols_to_check) # Drop NaNs again after coercion

                if not df_cleaned.empty:
                    # Aggregate max time per operation or row
                    # This part might need adjustment based on how 'Operation' relates to 'tack times'
                    # Assuming 'Operation' is a unique identifier or a grouping key
                    if 'Operation' in df_cleaned.columns:
                        # For each operation, find the max tack time across relevant columns
                        df_cleaned['Max_Tack_Time'] = df_cleaned[cols_to_check].max(axis=1)
                        bottlenecks = df_cleaned.groupby('Operation')['Max_Tack_Time'].max().sort_values(ascending=False)
                    else:
                        # Fallback if 'Operation' column is missing: consider max across rows directly
                        bottlenecks = df_cleaned[cols_to_check].max().sort_values(ascending=False)
                        st.warning(f"'Operation' column not found in {sheet_name}. Bottlenecks identified based on max values in timing columns.")
                    
                    if not bottlenecks.empty:
                        st.dataframe(bottlenecks.head())
                    else:
                        st.info(f"No bottlenecks identified in {sheet_name}.")
                else:
                    st.info(f"No valid data for bottleneck analysis after cleaning in {sheet_name}.")
            else:
                st.info(f"No data or empty data for sheet: {sheet_name}.")

        st.subheader("Line Balancing Optimization")
        st.markdown("""
        **Analysis:** Evaluates the balance of work across operations, identifying inefficiencies and suggesting improvements.
        **Features Used:** Operation times, cycle times.
        **Calculation Logic:** Compares current cycle time to a target cycle time, and calculates line efficiency.
        """)
        if production_optimizer:
            optimization_results = production_optimizer.optimize_line_balancing()
            if optimization_results:
                for sheet_name, results in optimization_results.items():
                    st.markdown(f"#### Sheet: {sheet_name}")
                    st.write(f"Current Cycle Time: {results.get('current_cycle_time', 'N/A'):.2f}")
                    st.write(f"Target Cycle Time: {results.get('target_cycle_time', 'N/A'):.2f}")
                    st.write(f"Line Efficiency: {results.get('efficiency', 'N/A'):.2f}%")
                    if results.get('optimization_needed') is not None and not results['optimization_needed'].empty:
                        st.write("Operations needing optimization:")
                        st.write(results['optimization_needed'][results['optimization_needed']].index.tolist())
                    else:
                        st.info(f"No specific operations needing optimization identified for {sheet_name}.")
            else:
                st.info("No optimization results available.")
        else:
            st.info("Production optimizer not initialized. Optimization results not available.")

        st.subheader("Optimization Recommendations")
        if production_optimizer:
            recommendations = production_optimizer.generate_recommendations()
            if recommendations:
                for sheet_name, sheet_recs in recommendations.items(): # This assumes recommendations is a dict of lists
                    if sheet_recs:
                        st.markdown(f"**Recommendations for {sheet_name}:**")
                        for rec in sheet_recs:
                            st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
                    else:
                        st.info(f"No specific recommendations generated for {sheet_name} at this time.")
            else:
                st.info("No specific recommendations generated for production optimization at this time.")
        else:
            st.info("Production optimizer not initialized. Recommendations not available.")

        st.subheader("Detailed Analysis")
        if production_data:
            selected_sheet = st.selectbox("Select Sheet", list(production_data.keys()))
            if selected_sheet:
                df = production_data[selected_sheet]
                st.dataframe(df)
        else:
            st.info("No sheets available in production data for detailed analysis.")

    else:
        st.error("No production optimization data available. Please check the data files and ensure 'Capacity study , Line balancing sheet.xlsx' is correctly formatted.")


def loss_time_analysis_page(analyzers):
    st.title("Loss Time Analysis")
    st.markdown("""
    ### Understanding Loss Time Analysis
    This module focuses on identifying, quantifying, and analyzing the causes of production loss times
    to help improve efficiency and productivity.
    **ML Algorithms:**
    - **Classification Models (e.g., Random Forest, Gradient Boosting):** Can predict the *reason* for loss time based on various operational parameters (e.g., machine type, shift, product).
    - **Regression Models (e.g., Linear Regression, Ridge Regression):** Can predict the *duration* of loss time events.
    - **Clustering (e.g., K-Means):** Can group similar loss time events or patterns to identify common underlying issues.
    - **Time Series Forecasting (e.g., ARIMA, Prophet):** Can forecast future loss time occurrences or durations based on historical trends.
    **Further Predictions/Optimizations:**
    - **Root Cause Prediction:** Predict the most likely root causes of recurring loss times.
    - **Preventive Action Recommendation:** Suggest specific preventive actions to mitigate predicted loss times.
    - **Impact Assessment:** Quantify the financial or operational impact of different loss time categories.
    """)

    loss_time_data = analyzers.get('loss_time', {}).get('data', {})
    loss_time_analyzer = analyzers.get('loss_time', {}).get('analyzer')

    if loss_time_data:
        st.subheader("Loss Time Patterns by Line")
        st.markdown("""
        **Analysis:** Aggregates loss time data by production line, showing total loss time, frequency, and average duration.
        **Features Used:** 'Line ', 'Act' (actual loss time).
        **Calculation Logic:** Grouping by 'Line ' and calculating count, sum, mean, and standard deviation of 'Act'.
        """)

        for sheet_name, df in loss_time_data.items():
            st.markdown(f"#### Sheet: {sheet_name}")
            if df is not None and not df.empty:
                df_cleaned = df.dropna(subset=['Line ', 'Act'])
                df_cleaned['Act'] = pd.to_numeric(df_cleaned['Act'], errors='coerce')
                df_cleaned = df_cleaned.dropna(subset=['Act'])

                if not df_cleaned.empty:
                    line_stats = df_cleaned.groupby('Line ')['Act'].agg(['count', 'sum', 'mean', 'std']).round(2)
                    st.dataframe(line_stats)
                else:
                    st.info(f"No valid data after cleaning for loss time patterns by line in {sheet_name}.")
            else:
                st.info(f"No data or empty data for sheet: {sheet_name}.")

        st.subheader("Loss Time Patterns by Reason")
        st.markdown("""
        **Analysis:** Breaks down loss time by reason category, identifying the most common and impactful reasons for delays.
        **Features Used:** 'Unnamed: 8' (assumed to be loss reason), 'Act'.
        **Calculation Logic:** Grouping by 'Unnamed: 8' and calculating count, sum, mean, and standard deviation of 'Act'.
        """)

        for sheet_name, df in loss_time_data.items():
            st.markdown(f"#### Sheet: {sheet_name}")
            if df is not None and not df.empty:
                df_cleaned = df.dropna(subset=['Unnamed: 8', 'Act'])
                df_cleaned['Act'] = pd.to_numeric(df_cleaned['Act'], errors='coerce')
                df_cleaned = df_cleaned.dropna(subset=['Act'])

                if not df_cleaned.empty:
                    reason_stats = df_cleaned.groupby('Unnamed: 8')['Act'].agg(['count', 'sum', 'mean', 'std']).round(2)
                    st.dataframe(reason_stats)

                    # Plotting top reasons
                    top_reasons_plot = reason_stats.sort_values(by='sum', ascending=False).head(10)
                    fig = px.bar(top_reasons_plot, y='sum', x=top_reasons_plot.index,
                                 title=f'Top 10 Loss Reasons in {sheet_name}',
                                 labels={'sum': 'Total Loss Time (minutes)', 'index': 'Loss Reason'},
                                 color_discrete_sequence=px.colors.qualitative.Dark24)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No valid data after cleaning for loss time patterns by reason in {sheet_name}.")
            else:
                st.info(f"No data or empty data for sheet: {sheet_name}.")

        st.subheader("Critical Loss Issues")
        st.markdown("""
        **Analysis:** Identifies the top loss time reasons that contribute most significantly to overall production delays.
        **Features Used:** 'Unnamed: 8', 'Act'.
        **Calculation Logic:** Summing 'Act' by 'Unnamed: 8' and identifying top contributors.
        """)

        if loss_time_analyzer:
            critical_issues_results = loss_time_analyzer.identify_critical_issues()
            if critical_issues_results:
                for sheet_name, results in critical_issues_results.items():
                    st.markdown(f"#### Sheet: {sheet_name}")
                    if results.get('critical_issues') is not None and not results['critical_issues'].empty:
                        st.write("Top Critical Issues:")
                        for reason, time in results['critical_issues'].items():
                            impact = results['impact_percentage'].get(reason, 'N/A')
                            st.markdown(f'<div class="insight-box">- **{reason}**: {time:.2f} minutes ({impact}% of total loss time)</div>', unsafe_allow_html=True)
                    else:
                        st.info(f"No critical issues identified for {sheet_name} at this time.")
            else:
                st.info("No critical issues results available.")
        else:
            st.info("Loss time analyzer not initialized. Critical issues not available.")

        st.subheader("Loss Time Recommendations")
        if loss_time_analyzer:
            recommendations = loss_time_analyzer.generate_recommendations()
            if recommendations:
                for sheet_name, sheet_recs in recommendations.items():
                    if sheet_recs:
                        st.markdown(f"**Recommendations for {sheet_name}:**")
                        for rec in sheet_recs:
                            st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
                    else:
                        st.info(f"No specific recommendations generated for {sheet_name} at this time.")
            else:
                st.info("No specific recommendations generated for loss time at this time.")
        else:
            st.info("Loss time analyzer not initialized. Recommendations not available.")

        st.subheader("Detailed Analysis")
        if loss_time_data:
            selected_sheet = st.selectbox("Select Sheet", list(loss_time_data.keys()))
            if selected_sheet:
                df = loss_time_data[selected_sheet]
                st.dataframe(df)
        else:
            st.info("No sheets available in loss time data for detailed analysis.")
    else:
        st.error("No loss time analysis data available. Please check the data files and ensure 'CCL loss time_.xlsx' is correctly formatted.")


def quality_control_page(analyzers):
    st.title("Quality Control")
    st.markdown("""
    ### Understanding Quality Control
    This module analyzes inspection data and Factory Return Authorization (FRA) data to identify quality issues,
    track defect trends, and provide recommendations for improvement.
    **ML Algorithms:**
    - **Classification Models (e.g., Support Vector Machines, Logistic Regression):** Can predict the likelihood of a product being defective based on manufacturing parameters or inspection results.
    - **Anomaly Detection (e.g., Isolation Forest, One-Class SVM):** Can identify unusual patterns in inspection data that indicate potential quality deviations or fraudulent returns.
    - **Time Series Forecasting:** Can predict future defect rates or return volumes.
    **Further Predictions/Optimizations:**
    - **Defect Cause Prediction:** Predict the most probable causes of new defects based on historical patterns and product specifications.
    - **Early Warning System:** Develop an alert system that notifies about potential quality issues before they escalate.
    - **Supplier Quality Scorecard:** Predict supplier quality performance based on past delivery and defect data.
    """)

    quality_data = analyzers.get('quality', {}).get('data', {})
    quality_controller = analyzers.get('quality', {}).get('analyzer')

    if quality_data:
        st.subheader("Inspection Data Analysis")
        st.markdown("""
        **Analysis:** Provides an overview of inspection results, including pass rates and distribution of defects.
        **Features Used:** 'QTY PASSED', 'QTY REJECTED'.
        **Calculation Logic:** Calculates pass rate as (QTY PASSED / (QTY PASSED + QTY REJECTED)) * 100.
        """)
        
        if 'Master_Inspection_Sheet' in quality_data and not quality_data['Master_Inspection_Sheet'].empty:
            inspection_df = quality_data['Master_Inspection_Sheet'].copy()
            if 'QTY PASSED' in inspection_df.columns and 'QTY REJECTED' in inspection_df.columns:
                inspection_df['QTY PASSED'] = pd.to_numeric(inspection_df['QTY PASSED'], errors='coerce').fillna(0)
                inspection_df['QTY REJECTED'] = pd.to_numeric(inspection_df['QTY REJECTED'], errors='coerce').fillna(0)

                total_inspected = inspection_df['QTY PASSED'].sum() + inspection_df['QTY REJECTED'].sum()
                total_passed = inspection_df['QTY PASSED'].sum()
                pass_rate = (total_passed / total_inspected * 100) if total_inspected > 0 else 0
                
                st.write(f"Overall Pass Rate: {pass_rate:.2f}%")

                # Plotting defects distribution
                defect_reasons = inspection_df.columns[inspection_df.columns.str.contains('REASON', case=False) & ~inspection_df.columns.str.contains('REJECTED', case=False)].tolist()
                if defect_reasons:
                    defect_counts = inspection_df[defect_reasons].sum().sort_values(ascending=False)
                    fig = px.bar(defect_counts, x=defect_counts.index, y=defect_counts.values,
                                title='Top Defect Reasons',
                                labels={'x': 'Defect Reason', 'y': 'Count'},
                                color_discrete_sequence=px.colors.qualitative.Set2)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No defect reason columns found in 'Master_Inspection_Sheet'.")
            else:
                st.info("Required columns 'QTY PASSED' or 'QTY REJECTED' not found in 'Master_Inspection_Sheet'.")
        else:
            st.info("No 'Master_Inspection_Sheet' found or it is empty in quality data.")

        st.subheader("FRA Data Analysis")
        st.markdown("""
        **Analysis:** Examines Factory Return Authorization (FRA) data to understand reasons for returns, quantities involved, and closure times.
        **Features Used:** 'FRA Reason', 'FRA Qty', 'Total Days \n(Taken for Closure)'.
        **Calculation Logic:** Aggregating quantity by reason and calculating average closure time.
        """)
        if 'FDR & FRA tracker' in quality_data and not quality_data['FDR & FRA tracker'].empty:
            fra_df = quality_data['FDR & FRA tracker'].copy()
            if 'FRA Reason' in fra_df.columns and 'FRA Qty' in fra_df.columns:
                fra_df['FRA Qty'] = pd.to_numeric(fra_df['FRA Qty'], errors='coerce').fillna(0)
                fra_df['Total Days \n(Taken for Closure)'] = pd.to_numeric(fra_df['Total Days \n(Taken for Closure)'], errors='coerce')
                
                reason_qty = fra_df.groupby('FRA Reason')['FRA Qty'].sum().sort_values(ascending=False)
                st.write("Quantity by FRA Reason:")
                st.dataframe(reason_qty)

                avg_closure_time = fra_df['Total Days \n(Taken for Closure)'].mean()
                st.write(f"Average Days for Closure: {avg_closure_time:.2f}")

                fig = px.pie(names=reason_qty.index, values=reason_qty.values,
                             title='Distribution of FRA Reasons',
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Required columns 'FRA Reason' or 'FRA Qty' not found in 'FDR & FRA tracker' sheet.")
        else:
            st.info("No 'FDR & FRA tracker' sheet found or it is empty in quality data.")

        st.subheader("Quality Issue Identification")
        st.markdown("""
        **Analysis:** Pinpoints the major quality issues based on their frequency and impact.
        **Features Used:** 'FRA Reason', 'FRA Qty'.
        **Calculation Logic:** Identifies top reasons by total quantity rejected.
        """)
        if quality_controller:
            issue_results = quality_controller.identify_quality_issues()
            if issue_results and issue_results.get('top_issues') is not None:
                st.write("Top 5 Quality Issues:")
                for reason, qty in issue_results['top_issues'].items():
                    st.markdown(f'<div class="insight-box">- **{reason}**: {qty:.2f} units</div>', unsafe_allow_html=True)
            else:
                st.info("No major quality issues identified at this time.")
        else:
            st.info("Quality controller not initialized. Issue identification not available.")

        st.subheader("Quality Improvement Recommendations")
        if quality_controller:
            recommendations = quality_controller.generate_recommendations()
            if recommendations:
                # Check if recommendations is a list or a dictionary
                if isinstance(recommendations, dict):
                    for sheet_name, sheet_recs in recommendations.items():
                        if sheet_recs:
                            st.markdown(f"**Recommendations for {sheet_name}:**")
                            for rec in sheet_recs:
                                st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
                        else:
                            st.info(f"No specific recommendations generated for {sheet_name} at this time.")
                elif isinstance(recommendations, list):
                    # If it's a list of recommendations (as in quality_control.py)
                    for rec in recommendations:
                        st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
                else:
                    st.info("Recommendations are in an unexpected format.")
            else:
                st.info("No specific recommendations generated for quality control at this time.")
        else:
            st.info("Quality controller not initialized. Recommendations not available.")
        
        st.subheader("Detailed Analysis")
        if quality_data:
            selected_sheet = st.selectbox("Select Sheet", list(quality_data.keys()))
            if selected_sheet:
                df = quality_data[selected_sheet]
                st.dataframe(df)
        else:
            st.info("No sheets available in quality data for detailed analysis.")

    else:
        st.error("No quality control data available. Please check the data files and ensure 'Stores - Data sets for AI training program.xlsx' is correctly formatted.")


def product_analysis_page(analyzers):
    st.title("Product Analysis")
    st.markdown("""
    ### Understanding Product Analysis
    This module analyzes your 'Quadrant data - AI.xlsx' focusing on operator performance and
    categorization into quadrants to understand product-wise performance variations.
    **ML Algorithms:** KMeans Clustering is explicitly used in the `product_analysis.py` module to group operators into performance clusters (e.g., for the 'Competency Matrix' analysis). Other unsupervised learning methods like PCA could be used for dimensionality reduction.
    **Further Predictions:**
    - **Performance Trajectory Prediction:** Predict future performance levels of operators based on their historical data.
    - **Training Needs Identification:** Identify operators who might benefit from specific training interventions based on their quadrant classification.
    - **Optimal Team Formation:** Group operators with complementary skills for specific tasks or production lines to maximize output.
    """)

    product_data = analyzers.get('product', {}).get('data', {})
    product_analyzer = analyzers.get('product', {}).get('analyzer')

    if product_data:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Operator Performance Distribution")
            st.markdown("""
            **Analysis:** This chart shows the distribution of operator performance percentages, helping to identify performance clusters.
            **Features Used:** 'Performance %'.
            **Calculation Logic:** A histogram of the 'Performance %' column to show value distribution and frequency.
            """)
            if 'Competency Matrix' in product_data and not product_data['Competency Matrix'].empty:
                competency_df = product_data['Competency Matrix'].copy()
                if 'Performance %' in competency_df.columns:
                    competency_df['Performance %'] = pd.to_numeric(competency_df['Performance %'], errors='coerce')
                    fig = px.histogram(competency_df, x='Performance %', nbins=20,
                                    title='Distribution of Operator Performance',
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Required column 'Performance %' not found in 'Competency Matrix' sheet.")
            else:
                st.info("No 'Competency Matrix' sheet found or it is empty in product data.")

        with col2:
            st.subheader("Quadrant Breakdown")
            st.markdown("""
            **Analysis:** This pie chart visualizes the distribution of operators across different performance quadrants (e.g., High Performer, Average, Low Performer).
            **Features Used:** 'Quadrant'.
            **Calculation Logic:** Counting the occurrences of each unique value in the 'Quadrant' column.
            """)
            if 'Quadrant details' in product_data and not product_data['Quadrant details'].empty:
                quadrant_df = product_data['Quadrant details'].copy()
                if 'Quadrant' in quadrant_df.columns:
                    quadrant_counts = quadrant_df['Quadrant'].value_counts().reset_index()
                    quadrant_counts.columns = ['Quadrant', 'Count']
                    fig = px.pie(quadrant_counts, values='Count', names='Quadrant',
                                title='Operator Distribution by Quadrant')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Required column 'Quadrant' not found in 'Quadrant details' sheet.")
            else:
                st.info("No 'Quadrant details' sheet found or it is empty in product data.")

        st.subheader("Performance Improvement Recommendations")
        if product_analyzer:
            recommendations = product_analyzer.generate_recommendations()
            if recommendations:
                # Check if recommendations is a list or a dictionary
                if isinstance(recommendations, dict):
                    for sheet_name, sheet_recs in recommendations.items():
                        if sheet_recs:
                            st.markdown(f"**Recommendations for {sheet_name}:**")
                            for rec in sheet_recs:
                                st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
                        else:
                            st.info(f"No specific recommendations generated for {sheet_name} at this time.")
                elif isinstance(recommendations, list):
                    # If it's a list of recommendations (as in product_analysis.py)
                    for rec in recommendations:
                        st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
                else:
                    st.info("Recommendations are in an unexpected format.")
            else:
                st.info("No specific recommendations generated for product analysis at this time.")
        else:
            st.info("Product analyzer not initialized. Recommendations not available.")
        
        st.subheader("Detailed Analysis")
        if product_data:
            selected_sheet = st.selectbox("Select Sheet", list(product_data.keys()))
            if selected_sheet:
                df = product_data[selected_sheet]
                st.dataframe(df)
        else:
            st.info("No sheets available in product data for detailed analysis.")

    else:
        st.error("No product analysis data available. Please check the data files and ensure 'Quadrant data - AI.xlsx' is correctly formatted.")


def worker_allocation_page(analyzers):
    st.title("Worker Allocation")
    st.markdown("""
    ### Understanding Worker Allocation
    This section focuses on optimizing worker deployment based on various factors such as skill, performance, and task requirements.
    This aims to maximize productivity and minimize idle time.
    **ML Algorithms:** For real-world worker allocation, this can involve complex optimization algorithms (e.g., Linear Programming, Integer Programming) to find the best assignment of workers to tasks under various constraints. More advanced scenarios might use Reinforcement Learning to dynamically adapt allocations based on real-time feedback. Predictive models (e.g., regression) could estimate task completion times.
    **How Workers are Allocated (Simulated Logic):** In this simulation, workers are assigned to tasks based on a simple matching algorithm. Tasks are prioritized (High > Medium > Low), and then by required skill level. Workers are selected based on meeting or exceeding the required skill level, with higher performance workers being prioritized first. Once a worker is assigned, they are no longer available for other tasks.
    **Features Used:**
    - **Worker Data:** 'Worker ID', 'Skill Level (1-10)', 'Experience (Years)', 'Current Performance (%)', 'Task Preference'.
    - **Task Data:** 'Task ID', 'Required Skill Level (1-10)', 'Estimated Time (hours)', 'Priority'.
    **Further Predictions/Optimizations:**
    - **Dynamic Re-allocation:** Predict the need for re-allocating workers based on real-time production changes or unexpected delays.
    - **Fatigue Prediction:** Use sensor data or historical work patterns to predict worker fatigue and optimize breaks or shift rotations.
    - **Skill Gap Identification:** Analyze allocation results to identify skill gaps in the workforce and suggest targeted training programs.
    """)

    num_workers = st.slider("Number of Workers", 10, 500, 30)
    num_tasks = st.slider("Number of Tasks", 5, 100, 15)

    if st.button("Generate Sample Worker Data"):
        worker_data = {
            'Worker ID': [f'W{i:03d}' for i in range(num_workers)],
            'Skill Level (1-10)': [random.randint(1, 11) for _ in range(num_workers)],
            'Experience (Years)': [random.randint(1, 15) for _ in range(num_workers)],
            'Current Performance (%)': [random.randint(70, 100) for _ in range(num_workers)],
            'Task Preference': [random.choice(['Cutting', 'Sewing', 'Finishing', 'Quality Control', 'Packing']) for _ in range(num_workers)]
        }
        task_data = {
            'Task ID': [f'T{i:03d}' for i in range(num_tasks)],
            'Required Skill Level (1-10)': [random.randint(1, 11) for _ in range(num_tasks)],
            'Estimated Time (hours)': [random.randint(1, 8) for _ in range(num_tasks)],
            'Priority': [random.choice(['High', 'Medium', 'Low']) for _ in range(num_tasks)]
        }
        
        st.session_state['worker_df'] = pd.DataFrame(worker_data)
        st.session_state['task_df'] = pd.DataFrame(task_data)
        st.success("Sample data generated!")

    if 'worker_df' in st.session_state and 'task_df' in st.session_state:
        st.subheader("Worker Data")
        st.dataframe(st.session_state['worker_df'])

        st.subheader("Task Data")
        st.dataframe(st.session_state['task_df'])

        if st.button("Run Allocation Simulation"):
            st.subheader("Simulated Worker-Task Allocation")
            
            allocated_tasks = []
            available_workers = st.session_state['worker_df'].copy()
            available_tasks = st.session_state['task_df'].copy()

            priority_map = {'High': 3, 'Medium': 2, 'Low': 1}
            available_tasks['Priority_Rank'] = available_tasks['Priority'].map(priority_map)
            available_tasks = available_tasks.sort_values(by=['Priority_Rank', 'Required Skill Level (1-10)'], ascending=[False, False])

            for index, task in available_tasks.iterrows():
                suitable_workers = available_workers[
                    available_workers['Skill Level (1-10)'] >= task['Required Skill Level (1-10)']
                ].sort_values(by='Current Performance (%)', ascending=False)

                if not suitable_workers.empty:
                    worker_id = suitable_workers.iloc[0]['Worker ID']
                    allocated_tasks.append({
                        'Task ID': task['Task ID'],
                        'Assigned Worker ID': worker_id,
                        'Task Priority': task['Priority'],
                        'Worker Skill': suitable_workers.iloc[0]['Skill Level (1-10)'],
                        'Worker Performance': suitable_workers.iloc[0]['Current Performance (%)']
                    })
                    available_workers = available_workers[available_workers['Worker ID'] != worker_id]
                    if available_workers.empty:
                        break

            if allocated_tasks:
                st.dataframe(pd.DataFrame(allocated_tasks))
            else:
                st.info("No tasks could be allocated with the current worker and task data.")

    else:
        st.info("Generate sample worker and task data to simulate allocation.")


# --- Sidebar and Data Loading Logic ---

# Set up the sidebar navigation (moved to top-level for single rendering)
st.sidebar.title("Navigation")
pages = {
    "🏠 Home": home_page,
    "⚙️ Production Optimization": production_optimization_page,
    "📉 Loss Time Analysis": loss_time_analysis_page,
    "✅ Quality Control": quality_control_page,
    "👕 Product Analysis": product_analysis_page,
    "🧑‍🏭 Worker Allocation": worker_allocation_page
}

selected_page = st.sidebar.radio("Go to", list(pages.keys()))

@st.cache_resource(ttl=3600) # Cache the loading of analyzers to avoid re-initializing on every rerun
def load_data():
    analyzers = {
        'production': {'data': {},'analyzer': None},
        'loss_time': {'data': {},'analyzer': None},
        'quality': {'data': {},'analyzer': None},
        'product': {'data': {},'analyzer': None}
    }

    # Production Optimization
    try:
        production_optimizer = ProductionOptimizer('Capacity study , Line balancing sheet.xlsx')
        production_data = production_optimizer.load_data()
        if production_data:
            analyzers['production'] = {'data': production_data, 'analyzer': production_optimizer}
        else:
            st.warning("Production data loading returned unexpected type (bool). Setting to empty dictionary.")
            analyzers['production'] = {'data': {}, 'analyzer': None}
    except Exception as e:
        st.error(f"Error loading Production Optimization data: {e}")
        analyzers['production'] = {'data': {}, 'analyzer': None} # Ensure it's a dict on error

    # Loss Time Analysis
    try:
        loss_time_analyzer = LossTimeAnalyzer('CCL loss time_.xlsx')
        loss_time_data = loss_time_analyzer.load_data()
        if loss_time_data:
            analyzers['loss_time'] = {'data': loss_time_data, 'analyzer': loss_time_analyzer}
        else:
            st.warning("Loss time data loading returned unexpected type (bool). Setting to empty dictionary.")
            analyzers['loss_time'] = {'data': {}, 'analyzer': None}
    except Exception as e:
        st.error(f"Error loading Loss Time Analysis data: {e}")
        analyzers['loss_time'] = {'data': {}, 'analyzer': None} # Ensure it's a dict on error

    # Quality Control
    try:
        quality_controller = QualityController('Stores - Data sets for AI training program.xlsx')
        quality_data = quality_controller.load_data()
        if quality_data:
            analyzers['quality'] = {'data': quality_data, 'analyzer': quality_controller}
        else:
            st.warning("Quality control data loading returned unexpected type (bool). Setting to empty dictionary.")
            analyzers['quality'] = {'data': {}, 'analyzer': None}
    except Exception as e:
        st.error(f"Error loading Quality Control data: {e}")
        analyzers['quality'] = {'data': {}, 'analyzer': None} # Ensure it's a dict on error

    # Product Analysis
    try:
        product_analyzer = ProductAnalyzer('Quadrant data - AI.xlsx')
        product_data = product_analyzer.load_data()
        if product_data:
            analyzers['product'] = {'data': product_data, 'analyzer': product_analyzer}
        else:
            st.warning("Product data loading returned unexpected type (bool). Setting to empty dictionary.")
            analyzers['product'] = {'data': {}, 'analyzer': None}
    except Exception as e:
        st.error(f"Error loading Product Analysis data: {e}")
        analyzers['product'] = {'data': {}, 'analyzer': None} # Ensure it's a dict on error

    return analyzers

# --- Main App Logic ---
def main():
    analyzers = load_data() 

    # Display the selected page
    if selected_page == "🏠 Home":
        home_page()
    else:
        pages[selected_page](analyzers)

if __name__ == "__main__":
    main()