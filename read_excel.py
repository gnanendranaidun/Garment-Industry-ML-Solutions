import pandas as pd
import numpy as np

def read_excel_file(file_path):
    """Read and display information about all sheets in an Excel file."""
    print(f"\n{'='*80}")
    print(f"File: {file_path}")
    print(f"{'='*80}")
    
    try:
        # Get all sheet names
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        print(f"\nAvailable sheets: {sheet_names}")
        
        # Read each sheet
        for sheet_name in sheet_names:
            print(f"\n{'-'*80}")
            print(f"Sheet: {sheet_name}")
            print(f"{'-'*80}")
            
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            print("\nColumns:")
            print(df.columns.tolist())
            
            print("\nFirst few rows:")
            print(df.head())
            
            print("\nData types:")
            print(df.dtypes)
            
            print("\nBasic statistics:")
            print(df.describe(include='all'))
            
            print(f"\nTotal rows: {len(df)}")
            print(f"Total columns: {len(df.columns)}")
            
    except Exception as e:
        print(f"Error reading file: {str(e)}")

def main():
    # List of Excel files to read
    excel_files = [
        'Capacity study , Line balancing sheet.xlsx',
        'CCL loss time_.xlsx',
        'Stores - Data sets for AI training program.xlsx',
        'Quadrant data - AI.xlsx'
    ]
    
    # Read each file
    for file in excel_files:
        read_excel_file(file)

if __name__ == "__main__":
    main() 