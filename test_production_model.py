#!/usr/bin/env python3
"""
Diagnostic script to test production efficiency model training
This script specifically tests the production model training with your data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_capacity_data():
    """Analyze capacity data files to understand structure"""
    print("🔍 Analyzing Capacity Data Files")
    print("=" * 50)
    
    csv_dir = Path('csv')
    if not csv_dir.exists():
        print("❌ CSV directory not found")
        return None
    
    # Find capacity files
    capacity_files = [f for f in csv_dir.glob('Capacity_study_*') if f.suffix == '.csv']
    
    if not capacity_files:
        print("❌ No capacity study files found")
        return None
    
    print(f"✅ Found {len(capacity_files)} capacity files:")
    
    capacity_data = {}
    for file in capacity_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                print(f"\n📄 {file.name}")
                print(f"   Rows: {len(df)}")
                print(f"   Columns: {len(df.columns)}")
                
                # Show column names
                print(f"   Columns: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
                
                # Check for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                print(f"   Numeric columns: {len(numeric_cols)}")
                
                # Show sample data
                print(f"   Sample data:")
                print(df.head(3).to_string(max_cols=5))
                
                capacity_data[file.stem] = df
        except Exception as e:
            print(f"❌ Error loading {file.name}: {e}")
    
    return capacity_data

def test_model_training():
    """Test the production efficiency model training"""
    print("\n🤖 Testing Production Efficiency Model Training")
    print("=" * 50)
    
    try:
        # Import the dashboard module
        import garment_industry_dashboard as gid
        
        # Load data using the dashboard's function
        print("📊 Loading data...")
        data = gid.load_garment_data()
        
        if not data:
            print("❌ No data loaded")
            return False
        
        print(f"✅ Data loaded: {list(data.keys())}")
        
        # Check capacity data specifically
        if 'capacity' not in data:
            print("❌ No capacity data found in loaded data")
            return False
        
        capacity_data = data['capacity']
        print(f"✅ Capacity data found: {len(capacity_data)} datasets")
        
        # Initialize ML predictor
        print("\n🔧 Initializing ML predictor...")
        ml_predictor = gid.GarmentMLPredictor()
        
        # Test production efficiency model training
        print("\n🎯 Training production efficiency model...")
        success = ml_predictor.train_production_efficiency_model(capacity_data)
        
        if success:
            print("✅ Production efficiency model trained successfully!")
            
            # Show model details
            if 'production_efficiency' in ml_predictor.model_metrics:
                metrics = ml_predictor.model_metrics['production_efficiency']
                print(f"\n📊 Model Performance:")
                print(f"   R² Score: {metrics['r2_score']:.3f}")
                print(f"   MSE: {metrics['mse']:.3f}")
                print(f"   Features: {metrics['features']}")
                print(f"   Target: {metrics['target']}")
                print(f"   Dataset: {metrics['dataset']}")
                print(f"   Samples: {metrics['n_samples']}")
                
                # Test prediction
                print(f"\n🔮 Testing prediction...")
                feature_values = [1.0] * len(metrics['features'])  # Dummy values
                prediction = ml_predictor.predict_production_efficiency(feature_values)
                
                if prediction is not None:
                    print(f"✅ Prediction successful: {prediction:.3f}")
                else:
                    print("❌ Prediction failed")
            
            return True
        else:
            print("❌ Production efficiency model training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return False

def suggest_fixes():
    """Suggest fixes based on analysis"""
    print("\n💡 Troubleshooting Suggestions")
    print("=" * 50)
    
    suggestions = [
        "1. **Check Data Quality**: Ensure CSV files have proper numeric data",
        "2. **Verify Column Names**: Look for columns with SMV, Efficiency, Capacity, etc.",
        "3. **Remove Empty Rows**: Clean data files of header rows and empty entries",
        "4. **Check File Format**: Ensure CSV files are properly formatted",
        "5. **Minimum Data**: Need at least 5-10 rows of clean numeric data"
    ]
    
    for suggestion in suggestions:
        print(f"   {suggestion}")

def main():
    """Main diagnostic function"""
    print("🔧 Production Efficiency Model Diagnostic Tool")
    print("=" * 60)
    
    # Step 1: Analyze capacity data
    capacity_data = analyze_capacity_data()
    
    if not capacity_data:
        print("\n❌ No capacity data found - cannot proceed with model training")
        suggest_fixes()
        return False
    
    # Step 2: Test model training
    success = test_model_training()
    
    if success:
        print("\n🎉 SUCCESS: Production efficiency model is working correctly!")
        print("\n✅ Your dashboard should now work without the training error.")
        print("\n🚀 Next steps:")
        print("   1. Run: streamlit run garment_industry_dashboard.py")
        print("   2. Navigate to ML Predictions section")
        print("   3. The production model should train automatically")
    else:
        print("\n❌ FAILED: Production efficiency model training is not working")
        suggest_fixes()
        
        print("\n🔧 Recommended fixes:")
        print("   1. Check the capacity CSV files for proper data structure")
        print("   2. Ensure numeric columns contain valid numbers")
        print("   3. Remove any header rows or metadata from CSV files")
        print("   4. Verify at least one file has 10+ rows of clean data")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
