"""
Dataset Verification and Analysis Script
Validates the synthetic work allocation datasets and provides sample analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_verify_datasets():
    """Load and verify both work allocation datasets"""
    
    print("🔍 Loading and Verifying Work Allocation Datasets")
    print("=" * 60)
    
    # Load datasets
    try:
        production_df = pd.read_csv('data/production_floor_work_allocation.csv')
        qc_df = pd.read_csv('data/quality_control_work_allocation.csv')
        print("✅ Datasets loaded successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error loading datasets: {e}")
        return None, None
    
    # Verify structure
    print(f"\n📊 Dataset Structure Verification:")
    print(f"Production Floor Dataset: {production_df.shape}")
    print(f"Quality Control Dataset: {qc_df.shape}")
    
    # Check for required columns
    expected_prod_cols = ['Worker_ID', 'Skill_Level_Department', 'Assigned_Task_Order', 
                         'Production_Target_Units', 'Machine_Workstation_Assignment', 
                         'Shift_Duration_Hours', 'Priority_Level']
    
    expected_qc_cols = ['Inspector_ID', 'QC_Department', 'Inspection_Type_Order',
                       'Quality_Target', 'Workstation_Assignment', 'Shift_Time', 'Urgency_Level']
    
    prod_cols_match = all(col in production_df.columns for col in expected_prod_cols)
    qc_cols_match = all(col in expected_qc_cols for col in expected_qc_cols)
    
    print(f"Production columns match: {'✅' if prod_cols_match else '❌'}")
    print(f"QC columns match: {'✅' if qc_cols_match else '❌'}")
    
    return production_df, qc_df

def analyze_production_dataset(df):
    """Analyze production floor allocation dataset"""
    
    print(f"\n🏭 Production Floor Dataset Analysis")
    print("=" * 40)
    
    # Extract departments and skill levels
    df['Department'] = df['Skill_Level_Department'].str.split(' - ').str[1]
    df['Skill_Level'] = df['Skill_Level_Department'].str.split(' - ').str[0]
    
    # Department distribution
    print(f"\n📈 Department Distribution:")
    dept_counts = df['Department'].value_counts()
    for dept, count in dept_counts.items():
        print(f"  {dept}: {count} workers ({count/len(df)*100:.1f}%)")
    
    # Skill level distribution
    print(f"\n🎯 Skill Level Distribution:")
    skill_counts = df['Skill_Level'].value_counts()
    for skill, count in skill_counts.items():
        print(f"  {skill}: {count} workers ({count/len(df)*100:.1f}%)")
    
    # Priority distribution
    print(f"\n⚡ Priority Level Distribution:")
    priority_counts = df['Priority_Level'].value_counts()
    for priority, count in priority_counts.items():
        print(f"  {priority}: {count} tasks ({count/len(df)*100:.1f}%)")
    
    # Production targets analysis
    print(f"\n🎯 Production Targets Analysis:")
    print(f"  Average target: {df['Production_Target_Units'].mean():.1f} units")
    print(f"  Min target: {df['Production_Target_Units'].min()} units")
    print(f"  Max target: {df['Production_Target_Units'].max()} units")
    print(f"  Std deviation: {df['Production_Target_Units'].std():.1f} units")
    
    # Shift duration analysis
    print(f"\n⏰ Shift Duration Analysis:")
    shift_counts = df['Shift_Duration_Hours'].value_counts().sort_index()
    for duration, count in shift_counts.items():
        print(f"  {duration} hours: {count} workers ({count/len(df)*100:.1f}%)")

def analyze_qc_dataset(df):
    """Analyze quality control allocation dataset"""
    
    print(f"\n🔍 Quality Control Dataset Analysis")
    print("=" * 40)
    
    # Extract QC levels and departments
    df['QC_Level'] = df['QC_Department'].str.split(' - ').str[0]
    df['QC_Dept'] = df['QC_Department'].str.split(' - ').str[1]
    
    # QC Department distribution
    print(f"\n📈 QC Department Distribution:")
    dept_counts = df['QC_Dept'].value_counts()
    for dept, count in dept_counts.items():
        print(f"  {dept}: {count} inspectors ({count/len(df)*100:.1f}%)")
    
    # QC Level distribution
    print(f"\n🎯 QC Level Distribution:")
    level_counts = df['QC_Level'].value_counts()
    for level, count in level_counts.items():
        print(f"  {level}: {count} inspectors ({count/len(df)*100:.1f}%)")
    
    # Urgency distribution
    print(f"\n⚡ Urgency Level Distribution:")
    urgency_counts = df['Urgency_Level'].value_counts()
    for urgency, count in urgency_counts.items():
        print(f"  {urgency}: {count} tasks ({count/len(df)*100:.1f}%)")
    
    # Shift time analysis
    print(f"\n⏰ Shift Time Distribution:")
    shift_counts = df['Shift_Time'].value_counts()
    for shift, count in shift_counts.items():
        print(f"  {shift}: {count} inspectors ({count/len(df)*100:.1f}%)")
    
    # Quality target types
    print(f"\n🎯 Quality Target Types:")
    target_types = df['Quality_Target'].str.split(' ').str[0].value_counts()
    for target_type, count in target_types.items():
        print(f"  {target_type}: {count} targets ({count/len(df)*100:.1f}%)")

def generate_sample_insights(production_df, qc_df):
    """Generate sample insights from the datasets"""
    
    print(f"\n💡 Sample Insights & Analytics")
    print("=" * 40)
    
    # Production insights
    production_df['Department'] = production_df['Skill_Level_Department'].str.split(' - ').str[1]
    production_df['Skill_Level'] = production_df['Skill_Level_Department'].str.split(' - ').str[0]
    
    # Average production by department
    dept_avg = production_df.groupby('Department')['Production_Target_Units'].mean().sort_values(ascending=False)
    print(f"\n🏭 Average Production Targets by Department:")
    for dept, avg in dept_avg.items():
        print(f"  {dept}: {avg:.1f} units/day")
    
    # Production by skill level
    skill_avg = production_df.groupby('Skill_Level')['Production_Target_Units'].mean().sort_values(ascending=False)
    print(f"\n👥 Average Production Targets by Skill Level:")
    for skill, avg in skill_avg.items():
        print(f"  {skill}: {avg:.1f} units/day")
    
    # QC insights
    qc_df['QC_Level'] = qc_df['QC_Department'].str.split(' - ').str[0]
    qc_df['QC_Dept'] = qc_df['QC_Department'].str.split(' - ').str[1]
    
    # QC workload distribution
    qc_dept_counts = qc_df['QC_Dept'].value_counts()
    print(f"\n🔍 QC Workload by Department:")
    for dept, count in qc_dept_counts.items():
        print(f"  {dept}: {count} inspections/day")
    
    # Critical/Urgent tasks
    critical_prod = len(production_df[production_df['Priority_Level'].isin(['Critical', 'Urgent'])])
    critical_qc = len(qc_df[qc_df['Urgency_Level'].isin(['Critical', 'Urgent'])])
    
    print(f"\n⚠️  High Priority Tasks:")
    print(f"  Production (Critical/Urgent): {critical_prod} tasks ({critical_prod/len(production_df)*100:.1f}%)")
    print(f"  QC (Critical/Urgent): {critical_qc} inspections ({critical_qc/len(qc_df)*100:.1f}%)")

def main():
    """Main verification and analysis function"""
    
    # Load and verify datasets
    production_df, qc_df = load_and_verify_datasets()
    
    if production_df is None or qc_df is None:
        return
    
    # Analyze datasets
    analyze_production_dataset(production_df)
    analyze_qc_dataset(qc_df)
    
    # Generate insights
    generate_sample_insights(production_df, qc_df)
    
    print(f"\n✅ Dataset Verification Complete!")
    print(f"📋 Both datasets are ready for integration with Streamlit applications")
    print(f"🔗 See WORK_ALLOCATION_DATASETS_README.md for detailed documentation")

if __name__ == "__main__":
    main()
