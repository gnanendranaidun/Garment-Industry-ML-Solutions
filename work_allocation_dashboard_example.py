"""
Work Allocation Dashboard Example
Demonstrates integration of synthetic work allocation datasets with Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Work Allocation Dashboard",
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
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_work_allocation_data():
    """Load work allocation datasets"""
    try:
        production_df = pd.read_csv('data/production_floor_work_allocation.csv')
        qc_df = pd.read_csv('data/quality_control_work_allocation.csv')
        
        # Process production data
        production_df['Department'] = production_df['Skill_Level_Department'].str.split(' - ').str[1]
        production_df['Skill_Level'] = production_df['Skill_Level_Department'].str.split(' - ').str[0]
        
        # Process QC data
        qc_df['QC_Level'] = qc_df['QC_Department'].str.split(' - ').str[0]
        qc_df['QC_Dept'] = qc_df['QC_Department'].str.split(' - ').str[1]
        
        return production_df, qc_df
    except FileNotFoundError:
        st.error("Work allocation datasets not found. Please run generate_work_allocation_datasets.py first.")
        return None, None

def create_production_dashboard(production_df):
    """Create production floor dashboard"""
    
    st.markdown('<div class="main-header">🏭 Production Floor Work Allocation</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(production_df)}</h3>
            <p>Total Workers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_target = production_df['Production_Target_Units'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_target:.0f}</h3>
            <p>Avg Daily Target</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_priority = len(production_df[production_df['Priority_Level'].isin(['Critical', 'Urgent'])])
        st.markdown(f"""
        <div class="metric-card">
            <h3>{high_priority}</h3>
            <p>High Priority Tasks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        departments = production_df['Department'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{departments}</h3>
            <p>Active Departments</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Department distribution
        dept_counts = production_df['Department'].value_counts()
        fig_dept = px.pie(
            values=dept_counts.values,
            names=dept_counts.index,
            title="Worker Distribution by Department",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_dept, use_container_width=True)
    
    with col2:
        # Production targets by skill level
        skill_targets = production_df.groupby('Skill_Level')['Production_Target_Units'].mean().sort_values()
        fig_skill = px.bar(
            x=skill_targets.values,
            y=skill_targets.index,
            orientation='h',
            title="Average Production Targets by Skill Level",
            color=skill_targets.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_skill, use_container_width=True)
    
    # Priority analysis
    st.subheader("📊 Priority Level Analysis")
    priority_counts = production_df['Priority_Level'].value_counts()
    fig_priority = px.bar(
        x=priority_counts.index,
        y=priority_counts.values,
        title="Task Distribution by Priority Level",
        color=priority_counts.values,
        color_continuous_scale='reds'
    )
    st.plotly_chart(fig_priority, use_container_width=True)

def create_qc_dashboard(qc_df):
    """Create quality control dashboard"""
    
    st.markdown('<div class="main-header">🔍 Quality Control Work Allocation</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(qc_df)}</h3>
            <p>Total Inspectors</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        qc_depts = qc_df['QC_Dept'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{qc_depts}</h3>
            <p>QC Departments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        urgent_inspections = len(qc_df[qc_df['Urgency_Level'].isin(['Critical', 'Urgent'])])
        st.markdown(f"""
        <div class="metric-card">
            <h3>{urgent_inspections}</h3>
            <p>Urgent Inspections</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        shifts = qc_df['Shift_Time'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{shifts}</h3>
            <p>Active Shifts</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # QC department workload
        dept_counts = qc_df['QC_Dept'].value_counts()
        fig_qc_dept = px.bar(
            x=dept_counts.index,
            y=dept_counts.values,
            title="Inspection Workload by QC Department",
            color=dept_counts.values,
            color_continuous_scale='blues'
        )
        fig_qc_dept.update_xaxis(tickangle=45)
        st.plotly_chart(fig_qc_dept, use_container_width=True)
    
    with col2:
        # QC level distribution
        level_counts = qc_df['QC_Level'].value_counts()
        fig_level = px.pie(
            values=level_counts.values,
            names=level_counts.index,
            title="Inspector Distribution by QC Level",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_level, use_container_width=True)
    
    # Urgency analysis
    st.subheader("⚡ Urgency Level Analysis")
    urgency_counts = qc_df['Urgency_Level'].value_counts()
    fig_urgency = px.bar(
        x=urgency_counts.index,
        y=urgency_counts.values,
        title="Inspection Distribution by Urgency Level",
        color=urgency_counts.values,
        color_continuous_scale='oranges'
    )
    st.plotly_chart(fig_urgency, use_container_width=True)

def create_combined_analysis(production_df, qc_df):
    """Create combined analysis dashboard"""
    
    st.markdown('<div class="main-header">📈 Combined Work Allocation Analysis</div>', unsafe_allow_html=True)
    
    # Workload comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Production vs QC Workload")
        
        # Create comparison chart
        workload_data = {
            'Category': ['Production Workers', 'QC Inspectors'],
            'Count': [len(production_df), len(qc_df)],
            'High Priority': [
                len(production_df[production_df['Priority_Level'].isin(['Critical', 'Urgent'])]),
                len(qc_df[qc_df['Urgency_Level'].isin(['Critical', 'Urgent'])])
            ]
        }
        
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            name='Total Assignments',
            x=workload_data['Category'],
            y=workload_data['Count'],
            marker_color='lightblue'
        ))
        fig_comparison.add_trace(go.Bar(
            name='High Priority',
            x=workload_data['Category'],
            y=workload_data['High Priority'],
            marker_color='red'
        ))
        
        fig_comparison.update_layout(
            title="Workload Distribution",
            barmode='group'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col2:
        st.subheader("Resource Allocation Summary")
        
        # Summary statistics
        total_workers = len(production_df) + len(qc_df)
        production_ratio = len(production_df) / total_workers * 100
        qc_ratio = len(qc_df) / total_workers * 100
        
        st.write(f"**Total Workforce:** {total_workers}")
        st.write(f"**Production Staff:** {len(production_df)} ({production_ratio:.1f}%)")
        st.write(f"**QC Staff:** {len(qc_df)} ({qc_ratio:.1f}%)")
        
        avg_prod_target = production_df['Production_Target_Units'].mean()
        st.write(f"**Average Production Target:** {avg_prod_target:.0f} units/day")
        
        # Efficiency insights
        expert_workers = len(production_df[production_df['Skill_Level'] == 'Expert'])
        senior_qc = len(qc_df[qc_df['QC_Level'] == 'Senior QC'])
        
        st.write(f"**Expert Production Workers:** {expert_workers}")
        st.write(f"**Senior QC Inspectors:** {senior_qc}")

def main():
    """Main dashboard function"""
    
    # Load data
    production_df, qc_df = load_work_allocation_data()
    
    if production_df is None or qc_df is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("📋 Work Allocation Dashboard")
    page = st.sidebar.selectbox(
        "Select Dashboard",
        ["Production Floor", "Quality Control", "Combined Analysis"]
    )
    
    # Display selected dashboard
    if page == "Production Floor":
        create_production_dashboard(production_df)
    elif page == "Quality Control":
        create_qc_dashboard(qc_df)
    else:
        create_combined_analysis(production_df, qc_df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Dataset Info:**")
    st.sidebar.markdown(f"Production Records: {len(production_df)}")
    st.sidebar.markdown(f"QC Records: {len(qc_df)}")
    st.sidebar.markdown("Generated: Synthetic Data")

if __name__ == "__main__":
    main()
