# Garment Industry Work Allocation Datasets

## Overview
This document describes two synthetic datasets created for garment industry work allocation at the beginning of each workday. Both datasets contain realistic, factual data representing actual garment manufacturing scenarios.

## Dataset 1: Production Floor Work Allocation
**File:** `data/production_floor_work_allocation.csv`
**Size:** 500 rows × 7 columns

### Purpose
Daily shift assignments for production workers across various departments in garment manufacturing facilities.

### Columns Description
1. **Worker_ID**: Unique identifier for each worker (W1000-W1499)
2. **Skill_Level_Department**: Combined skill level and department assignment
   - Skill Levels: Trainee, Junior, Senior, Expert, Supervisor
   - Departments: Cutting, Sewing, Finishing, Pressing, Packing, Embroidery, Pattern Making
3. **Assigned_Task_Order**: Specific task/order assignment with garment type
   - Order formats: ORD-XXXXX, PO-XXXXX, WO-XXXXX, JOB-XXXXX
   - Garment types: T-Shirt, Jeans, Dress Shirt, Polo Shirt, Jacket, Trousers, Blouse, Skirt, Hoodie, Shorts
4. **Production_Target_Units**: Daily production target adjusted for skill level and garment complexity
5. **Machine_Workstation_Assignment**: Specific machine/workstation assignment
   - Department-specific machines (e.g., Single Needle SN-01, Overlock OL-05, Steam Press SP-10)
6. **Shift_Duration_Hours**: Work shift duration (typically 7.5-9.0 hours)
7. **Priority_Level**: Task priority (Low, Medium, High, Urgent, Critical)

### Key Features
- Realistic production targets based on garment complexity and worker skill
- Department-appropriate machine assignments
- Weighted priority distribution (most tasks are Medium/High priority)
- Authentic garment industry terminology and processes

## Dataset 2: Quality Control Work Allocation
**File:** `data/quality_control_work_allocation.csv`
**Size:** 500 rows × 7 columns

### Purpose
Quality control and inspection scheduling for QC personnel across different inspection stages.

### Columns Description
1. **Inspector_ID**: Unique identifier for each QC inspector (QC2000-QC2499)
2. **QC_Department**: QC level and department specialization
   - QC Levels: QC Trainee, QC Inspector, Senior QC, QC Supervisor, QC Manager
   - Departments: Inline QC, Final QC, Fabric QC, Trim QC, Packing QC, Pre-Production QC, AQL Inspection
3. **Inspection_Type_Order**: Specific inspection task with batch and order information
   - Inspection types: Seam Quality Check, Measurement Check, Color Matching, Fabric Defect Check, etc.
   - Format: Batch-XXX | Inspection Type (Order: XXXXX)
4. **Quality_Target**: Quality metrics and targets
   - Defect Rate targets (< 1.5% to < 4%)
   - Inspection Quotas (pieces to inspect)
   - Accuracy Targets (95%-99%)
   - Throughput Targets (pieces/hour)
5. **Workstation_Assignment**: QC-specific workstation assignment
   - QC Tables, Inspection Booths, Light Boxes, Measurement Stations, etc.
6. **Shift_Time**: QC shift schedules (07:00-15:00, 08:00-17:00, 09:00-18:00, 10:00-19:00, 15:00-23:00)
7. **Urgency_Level**: Inspection urgency (Routine, Standard, Priority, Urgent, Critical)

### Key Features
- Comprehensive QC inspection types covering all garment quality aspects
- Realistic quality targets and metrics
- Appropriate workstation assignments for different inspection types
- Balanced urgency distribution reflecting real QC priorities

## Data Quality & Realism
Both datasets feature:
- **Authentic Industry Terminology**: Uses real garment manufacturing terms and processes
- **Realistic Operational Constraints**: Production targets and quality metrics based on industry standards
- **Proper Resource Allocation**: Machine and workstation assignments match department requirements
- **Balanced Distributions**: Priority/urgency levels reflect real-world manufacturing priorities
- **Comprehensive Coverage**: Represents full spectrum of garment manufacturing operations

## Integration with Existing Framework
These datasets are designed to integrate seamlessly with the existing Streamlit garment industry ML solutions framework:
- CSV format for easy loading with pandas
- Compatible with existing data visualization components
- Suitable for ML model training and analysis
- Follows established naming conventions and structure

## Usage Examples
```python
import pandas as pd

# Load production floor allocation data
production_df = pd.read_csv('data/production_floor_work_allocation.csv')

# Load quality control allocation data
qc_df = pd.read_csv('data/quality_control_work_allocation.csv')

# Basic analysis
print(f"Production workers: {len(production_df)}")
print(f"QC inspectors: {len(qc_df)}")
print(f"Departments: {production_df['Skill_Level_Department'].str.split(' - ').str[1].unique()}")
```

## Data Generation
The datasets were generated using a sophisticated Python script that:
- Ensures realistic relationships between variables
- Maintains industry-standard distributions
- Creates authentic garment manufacturing scenarios
- Provides comprehensive coverage of operations

Both datasets represent a typical day's work allocation in a medium to large-scale garment manufacturing facility and can be used for:
- Workforce planning and optimization
- Production scheduling analysis
- Quality control process improvement
- Machine learning model development
- Dashboard visualization and reporting
