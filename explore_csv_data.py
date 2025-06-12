import os
import pandas as pd

csv_dir = 'csv'
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

for csv_file in csv_files:
    path = os.path.join(csv_dir, csv_file)
    print(f"\n{'='*80}")
    print(f"File: {csv_file}")
    print(f"{'='*80}")
    try:
        df = pd.read_csv(path)
        print("Columns:")
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
        print(f"Error reading {csv_file}: {e}")
