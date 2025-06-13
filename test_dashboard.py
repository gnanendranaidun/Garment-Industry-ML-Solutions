#!/usr/bin/env python3
"""
Test script for the Garment Industry Dashboard
This script tests the main components without running the full Streamlit app
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_data_directory():
    """Test if data directory exists and has files"""
    print("\n🔍 Testing data directory...")
    
    csv_dir = Path('csv')
    if not csv_dir.exists():
        print("❌ CSV directory not found")
        print("💡 Create a 'csv' directory and add your data files")
        return False
    
    csv_files = list(csv_dir.glob('*.csv'))
    if not csv_files:
        print("⚠️ No CSV files found in csv directory")
        print("💡 Add CSV files to the 'csv' directory")
        return False
    
    print(f"✅ Found {len(csv_files)} CSV files:")
    for file in csv_files[:5]:  # Show first 5 files
        print(f"   📄 {file.name}")
    if len(csv_files) > 5:
        print(f"   ... and {len(csv_files) - 5} more files")
    
    return True

def test_dashboard_import():
    """Test importing the dashboard module"""
    print("\n🔍 Testing dashboard import...")
    
    try:
        import garment_industry_dashboard
        print("✅ Dashboard module imported successfully")
        return True
    except Exception as e:
        print(f"❌ Dashboard import failed: {e}")
        print("\n🔍 Error details:")
        traceback.print_exc()
        return False

def test_dashboard_functions():
    """Test key dashboard functions"""
    print("\n🔍 Testing dashboard functions...")
    
    try:
        import garment_industry_dashboard as gid
        
        # Test data loading function
        print("   Testing data loading...")
        # Note: This will show warnings if no data is found, which is expected
        data = gid.load_garment_data()
        print(f"   ✅ Data loading function works (loaded {len(data)} categories)")
        
        # Test ML predictor class
        print("   Testing ML predictor...")
        ml_predictor = gid.GarmentMLPredictor()
        print("   ✅ ML predictor class instantiated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Function testing failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Garment Industry Dashboard Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Dependencies", test_imports),
        ("Data Directory", test_data_directory),
        ("Dashboard Import", test_dashboard_import),
        ("Dashboard Functions", test_dashboard_functions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The dashboard should work without errors.")
        print("\n🚀 To run the dashboard:")
        print("   streamlit run garment_industry_dashboard.py")
    else:
        print("⚠️ Some tests failed. Please address the issues above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
