"""
Synthetic Dataset Generator for Garment Industry Work Allocation
Creates realistic datasets for production floor and quality control work allocation
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_production_floor_allocation_dataset():
    """Generate Production Floor Work Allocation Dataset"""
    
    # Define realistic garment industry data
    departments = ['Cutting', 'Sewing', 'Finishing', 'Pressing', 'Packing', 'Embroidery', 'Pattern Making']
    skill_levels = ['Trainee', 'Junior', 'Senior', 'Expert', 'Supervisor']
    
    # Garment types and orders
    garment_types = ['T-Shirt', 'Jeans', 'Dress Shirt', 'Polo Shirt', 'Jacket', 'Trousers', 'Blouse', 'Skirt', 'Hoodie', 'Shorts']
    order_prefixes = ['ORD', 'PO', 'WO', 'JOB']
    
    # Machine types by department
    machines = {
        'Cutting': ['Cutting Table CT-', 'Fabric Spreader FS-', 'Band Knife BK-', 'Straight Knife SK-'],
        'Sewing': ['Single Needle SN-', 'Overlock OL-', 'Flatlock FL-', 'Button Hole BH-', 'Button Attach BA-'],
        'Finishing': ['Steam Press SP-', 'Hand Iron HI-', 'Spot Cleaner SC-', 'Thread Trimmer TT-'],
        'Pressing': ['Steam Press SP-', 'Vacuum Press VP-', 'Hand Press HP-', 'Tunnel Finisher TF-'],
        'Packing': ['Packing Table PT-', 'Folding Machine FM-', 'Poly Bag Sealer PBS-', 'Carton Sealer CS-'],
        'Embroidery': ['Embroidery Machine EM-', 'Sequin Machine SM-', 'Applique Machine AM-'],
        'Pattern Making': ['CAD Station CAD-', 'Digitizer DIG-', 'Plotter PLT-', 'Pattern Table PT-']
    }
    
    # Priority levels
    priority_levels = ['Low', 'Medium', 'High', 'Urgent', 'Critical']
    
    # Shift times
    shift_times = ['06:00-14:00', '14:00-22:00', '22:00-06:00', '08:00-17:00', '09:00-18:00']
    
    # Generate 500 rows of data
    data = []
    
    for i in range(500):
        # Generate worker ID
        worker_id = f"W{1000 + i:04d}"
        
        # Select department and corresponding skill level
        department = random.choice(departments)
        skill_level = random.choice(skill_levels)
        
        # Generate order/task
        order_prefix = random.choice(order_prefixes)
        order_number = random.randint(10000, 99999)
        garment = random.choice(garment_types)
        assigned_task = f"{order_prefix}-{order_number} ({garment})"
        
        # Production target based on garment type and skill level
        base_targets = {
            'T-Shirt': 120, 'Polo Shirt': 100, 'Dress Shirt': 80, 'Jeans': 60,
            'Trousers': 70, 'Jacket': 40, 'Hoodie': 50, 'Shorts': 90,
            'Blouse': 75, 'Skirt': 85, 'Dress': 65
        }
        
        base_target = base_targets.get(garment, 80)
        
        # Adjust target based on skill level
        skill_multipliers = {'Trainee': 0.6, 'Junior': 0.8, 'Senior': 1.0, 'Expert': 1.2, 'Supervisor': 1.1}
        target_multiplier = skill_multipliers[skill_level]
        
        # Add some randomness
        production_target = int(base_target * target_multiplier * random.uniform(0.85, 1.15))
        
        # Machine assignment
        machine_types = machines.get(department, ['General Machine GM-'])
        machine_type = random.choice(machine_types)
        machine_number = random.randint(1, 20)
        machine_assignment = f"{machine_type}{machine_number:02d}"
        
        # Shift duration (8 hours standard with some variation)
        shift_duration = random.choice(['8.0', '8.5', '7.5', '9.0', '8.0', '8.0'])  # Most are 8 hours
        
        # Priority level (weighted towards Medium and High)
        priority_weights = [0.1, 0.4, 0.3, 0.15, 0.05]  # Low, Medium, High, Urgent, Critical
        priority_level = np.random.choice(priority_levels, p=priority_weights)
        
        data.append({
            'Worker_ID': worker_id,
            'Skill_Level_Department': f"{skill_level} - {department}",
            'Assigned_Task_Order': assigned_task,
            'Production_Target_Units': production_target,
            'Machine_Workstation_Assignment': machine_assignment,
            'Shift_Duration_Hours': shift_duration,
            'Priority_Level': priority_level
        })
    
    return pd.DataFrame(data)

def generate_quality_control_allocation_dataset():
    """Generate Quality Control Work Allocation Dataset"""
    
    # QC specific departments
    qc_departments = ['Inline QC', 'Final QC', 'Fabric QC', 'Trim QC', 'Packing QC', 'Pre-Production QC', 'AQL Inspection']
    
    # Inspector skill levels
    inspector_levels = ['QC Trainee', 'QC Inspector', 'Senior QC', 'QC Supervisor', 'QC Manager']
    
    # Inspection types
    inspection_types = [
        'Seam Quality Check', 'Measurement Check', 'Color Matching', 'Fabric Defect Check',
        'Button/Zipper Function', 'Label Verification', 'Packaging Inspection', 'Final Audit',
        'Trim Quality Check', 'Embroidery Quality', 'Print Quality Check', 'Washing Test',
        'Durability Test', 'Fit Check', 'Appearance Check'
    ]
    
    # QC workstations
    qc_workstations = [
        'QC Table QT-', 'Inspection Booth IB-', 'Measurement Station MS-', 'Light Box LB-',
        'Fabric Testing FT-', 'Wash Test WT-', 'Audit Station AS-', 'Photo Studio PS-'
    ]
    
    # Quality targets (defect rates, inspection quotas)
    target_types = ['Defect Rate <', 'Inspection Quota', 'Accuracy Target', 'Throughput Target']
    
    # Urgency levels
    urgency_levels = ['Routine', 'Standard', 'Priority', 'Urgent', 'Critical']
    
    # Shift times for QC
    qc_shift_times = ['07:00-15:00', '15:00-23:00', '08:00-17:00', '09:00-18:00', '10:00-19:00']
    
    data = []
    
    for i in range(500):
        # Generate inspector ID
        inspector_id = f"QC{2000 + i:04d}"
        
        # Select QC department and level
        qc_department = random.choice(qc_departments)
        inspector_level = random.choice(inspector_levels)
        
        # Generate inspection task
        inspection_type = random.choice(inspection_types)
        order_number = random.randint(10000, 99999)
        batch_number = random.randint(100, 999)
        inspection_task = f"Batch-{batch_number} | {inspection_type} (Order: {order_number})"
        
        # Quality target based on inspection type
        target_type = random.choice(target_types)
        
        if target_type == 'Defect Rate <':
            target_value = random.choice(['2%', '3%', '1.5%', '2.5%', '4%'])
        elif target_type == 'Inspection Quota':
            quota = random.randint(50, 200)
            target_value = f"{quota} pieces"
        elif target_type == 'Accuracy Target':
            accuracy = random.choice(['95%', '98%', '99%', '97%', '96%'])
            target_value = accuracy
        else:  # Throughput Target
            throughput = random.randint(20, 80)
            target_value = f"{throughput} pieces/hour"
        
        quality_target = f"{target_type} {target_value}"
        
        # Workstation assignment
        workstation_type = random.choice(qc_workstations)
        workstation_number = random.randint(1, 15)
        workstation_assignment = f"{workstation_type}{workstation_number:02d}"
        
        # Shift time
        shift_time = random.choice(qc_shift_times)
        
        # Urgency level (weighted towards Standard and Priority)
        urgency_weights = [0.15, 0.35, 0.3, 0.15, 0.05]  # Routine, Standard, Priority, Urgent, Critical
        urgency_level = np.random.choice(urgency_levels, p=urgency_weights)
        
        data.append({
            'Inspector_ID': inspector_id,
            'QC_Department': f"{inspector_level} - {qc_department}",
            'Inspection_Type_Order': inspection_task,
            'Quality_Target': quality_target,
            'Workstation_Assignment': workstation_assignment,
            'Shift_Time': shift_time,
            'Urgency_Level': urgency_level
        })
    
    return pd.DataFrame(data)

def main():
    """Generate and save both datasets"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Generating Production Floor Work Allocation Dataset...")
    production_df = generate_production_floor_allocation_dataset()
    
    print("Generating Quality Control Work Allocation Dataset...")
    qc_df = generate_quality_control_allocation_dataset()
    
    # Save datasets
    production_file = 'data/production_floor_work_allocation.csv'
    qc_file = 'data/quality_control_work_allocation.csv'
    
    production_df.to_csv(production_file, index=False)
    qc_df.to_csv(qc_file, index=False)
    
    print(f"\n✅ Datasets created successfully!")
    print(f"📊 Production Floor Dataset: {production_file}")
    print(f"   - Shape: {production_df.shape}")
    print(f"   - Columns: {list(production_df.columns)}")
    
    print(f"\n📊 Quality Control Dataset: {qc_file}")
    print(f"   - Shape: {qc_df.shape}")
    print(f"   - Columns: {list(qc_df.columns)}")
    
    # Display sample data
    print(f"\n📋 Sample Production Floor Data:")
    print(production_df.head(3).to_string())
    
    print(f"\n📋 Sample Quality Control Data:")
    print(qc_df.head(3).to_string())
    
    return production_df, qc_df

if __name__ == "__main__":
    main()
