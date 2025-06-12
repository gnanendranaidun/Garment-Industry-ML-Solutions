import os
import pandas as pd

# List of Excel files to convert
excel_files = [
    'Capacity study , Line balancing sheet.xlsx',
    'CCL loss time_.xlsx',
    'Stores - Data sets for AI training program.xlsx',
    'Quadrant data - AI.xlsx'
]

csv_dir = 'csv'
os.makedirs(csv_dir, exist_ok=True)

for file in excel_files:
    xl = pd.ExcelFile(file)
    for sheet in xl.sheet_names:
        df = pd.read_excel(file, sheet_name=sheet)
        # Clean up file and sheet names for CSV
        base = os.path.splitext(os.path.basename(file))[0].replace(' ', '_')
        sheet_clean = sheet.replace(' ', '_')
        csv_name = f"{base}__{sheet_clean}.csv"
        csv_path = os.path.join(csv_dir, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
