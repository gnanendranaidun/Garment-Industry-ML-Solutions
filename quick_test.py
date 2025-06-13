#!/usr/bin/env python3
"""
Quick test for production model training
"""

import pandas as pd
import numpy as np
from pathlib import Path

def test_capacity_data_loading():
    """Test loading capacity data with improved logic"""
    print("🔍 Testing Capacity Data Loading")
    print("=" * 40)
    
    csv_dir = Path('csv')
    capacity_files = [f for f in csv_dir.glob('Capacity_study_*') if f.suffix == '.csv']
    
    for file in capacity_files[:3]:  # Test first 3 files
        print(f"\n📄 Testing: {file.name}")
        
        try:
            # First, try reading normally
            df = pd.read_csv(file)
            print(f"   Initial load: {len(df)} rows, {len(df.columns)} columns")
            
            # Check if first row contains "Unnamed" columns
            if df.columns[0].startswith('Unnamed'):
                print("   Detected 'Unnamed' columns - trying header=1")
                # Use row 1 as headers
                df = pd.read_csv(file, header=1)
                df = df.dropna(how='all')
                print(f"   After header fix: {len(df)} rows, {len(df.columns)} columns")
            
            # Show column names
            print(f"   Columns: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
            
            # Check for numeric columns
            numeric_cols = []
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if not df[col].isna().all():
                    numeric_cols.append(col)
            
            print(f"   Numeric columns: {len(numeric_cols)}")
            if numeric_cols:
                print(f"   Sample numeric columns: {numeric_cols[:3]}")
            
            # Check for production-related columns
            production_cols = []
            for col in df.columns:
                col_str = str(col).lower()
                if any(keyword in col_str for keyword in ['smv', 'eff', 'capacity', 'cycle', 'tgt', 'production']):
                    production_cols.append(col)
            
            print(f"   Production-related columns: {production_cols}")
            
            if len(numeric_cols) >= 2 and len(df.dropna()) >= 5:
                print("   ✅ Suitable for model training")
            else:
                print("   ❌ Not suitable for model training")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_simple_model_training():
    """Test simple model training logic"""
    print("\n🤖 Testing Simple Model Training")
    print("=" * 40)
    
    try:
        # Load the specific file that should work
        file_path = Path('csv/Capacity_study_,_Line_balancing_sheet__Capacity_study_after.csv')
        
        if not file_path.exists():
            print("❌ Target file not found")
            return False
        
        # Load with proper headers
        df = pd.read_csv(file_path, header=1)
        df = df.dropna(how='all')
        
        print(f"✅ Loaded data: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Find suitable features and target
        features = []
        targets = []
        
        for col in df.columns:
            col_str = str(col).lower()
            if 'smv' in col_str or 'cycle' in col_str:
                features.append(col)
            elif 'eff' in col_str or 'capacity' in col_str:
                targets.append(col)
        
        print(f"Potential features: {features}")
        print(f"Potential targets: {targets}")
        
        if features and targets:
            # Try simple training
            feature_col = features[0]
            target_col = targets[0]
            
            # Clean data
            clean_df = df[[feature_col, target_col]].dropna()
            print(f"Clean data: {len(clean_df)} rows")
            
            if len(clean_df) >= 5:
                X = clean_df[[feature_col]]
                y = clean_df[target_col]
                
                print(f"X shape: {X.shape}, y shape: {y.shape}")
                print(f"y variance: {y.var()}")
                
                if y.var() > 0:
                    print("✅ Data is suitable for model training!")
                    return True
                else:
                    print("❌ No variance in target variable")
            else:
                print("❌ Insufficient clean data")
        else:
            print("❌ Could not identify features and targets")
            
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Quick Production Model Test")
    print("=" * 50)
    
    test_capacity_data_loading()
    success = test_simple_model_training()
    
    if success:
        print("\n🎉 SUCCESS: Data is ready for model training!")
    else:
        print("\n❌ FAILED: Data needs more work")
