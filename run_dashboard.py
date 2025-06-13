#!/usr/bin/env python3
"""
Garment Industry Analytics Dashboard Launcher
============================================

This script provides an easy way to launch the Garment Industry Analytics Dashboard
with proper error handling and user guidance.

Usage:
    python run_dashboard.py

Requirements:
    - Python 3.7+
    - All dependencies listed in requirements.txt
    - CSV data files in the 'csv/' directory
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Error: Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'scikit-learn',
        'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
        else:
            print(f"✅ {package} is installed")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("\nTo install missing packages, run:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_data_directory():
    """Check if CSV data directory exists and contains files"""
    csv_dir = 'csv'
    
    if not os.path.exists(csv_dir):
        print(f"❌ Error: '{csv_dir}' directory not found")
        print("\nPlease create the 'csv' directory and add your data files")
        return False
    
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"⚠️  Warning: No CSV files found in '{csv_dir}' directory")
        print("The dashboard will run but may have limited functionality")
        return True
    
    print(f"✅ Found {len(csv_files)} CSV files in '{csv_dir}' directory")
    return True

def check_dashboard_file():
    """Check if the main dashboard file exists"""
    dashboard_file = 'garment_industry_comprehensive_dashboard.py'
    
    if not os.path.exists(dashboard_file):
        print(f"❌ Error: '{dashboard_file}' not found")
        print("Please ensure the dashboard file is in the current directory")
        return False
    
    print(f"✅ Dashboard file found: {dashboard_file}")
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    dashboard_file = 'garment_industry_comprehensive_dashboard.py'
    
    print("\n🚀 Launching Garment Industry Analytics Dashboard...")
    print("📱 The dashboard will open in your default web browser")
    print("🔗 Default URL: http://localhost:8501")
    print("\n💡 Tips:")
    print("   - Use Ctrl+C to stop the dashboard")
    print("   - Refresh the browser page if you encounter any issues")
    print("   - Check the terminal for any error messages")
    print("\n" + "="*60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            dashboard_file,
            '--server.headless', 'false',
            '--server.port', '8501',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {str(e)}")
        print("\nTry running manually with:")
        print(f"streamlit run {dashboard_file}")

def main():
    """Main function to run all checks and launch dashboard"""
    print("👔 Garment Industry Analytics Dashboard Launcher")
    print("=" * 60)
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Data Directory", check_data_directory),
        ("Dashboard File", check_dashboard_file)
    ]
    
    print("\n🔍 Running system checks...")
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    if not all_passed:
        print("\n❌ Some checks failed. Please resolve the issues above before running the dashboard.")
        return False
    
    print("\n✅ All checks passed!")
    
    # Ask user if they want to proceed
    try:
        response = input("\n🚀 Launch the dashboard? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            launch_dashboard()
        else:
            print("👋 Dashboard launch cancelled")
    except KeyboardInterrupt:
        print("\n👋 Dashboard launch cancelled")
    
    return True

if __name__ == "__main__":
    main()
