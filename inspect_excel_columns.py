import pandas as pd
import os

excel_files = [
    'Capacity study , Line balancing sheet.xlsx',
    'CCL loss time_.xlsx'
]

for file in excel_files:
    if os.path.exists(file):
        print(f'\n--- Columns in {file} ---')
        try:
            # Read only the header row to get column names for all sheets
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                print(f'Sheet: {sheet_name}')
                # Read only first row to get column names
                df_header = pd.read_excel(file, sheet_name=sheet_name, nrows=1)
                print(df_header.columns.tolist())
        except Exception as e:
            print(f'Error reading {file}: {e}')
    else:
        print(f'Warning: {file} not found.') 