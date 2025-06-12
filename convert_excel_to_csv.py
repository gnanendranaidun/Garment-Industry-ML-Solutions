import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta

def find_actual_header(df_raw, potential_keywords, max_rows_to_check=10):
    """Attempts to find the most probable header row in a DataFrame.
    Checks for presence of keywords, or simply the row with most non-null string values.
    Returns the 0-indexed row number of the header.
    """
    best_header_row = 0
    max_meaningful_cols = -1 # Initialize to -1 to ensure first valid row is picked

    for i in range(min(max_rows_to_check, len(df_raw))):
        row_as_list = df_raw.iloc[i].astype(str).tolist()
        
        # Count how many words in the row match our potential keywords
        keyword_matches = sum(1 for keyword in potential_keywords if any(keyword.lower() in str(cell).lower() for cell in row_as_list))
        
        # Count non-null, non-"Unnamed" string columns
        meaningful_cols_count = sum(1 for col_val in row_as_list if isinstance(col_val, str) and not str(col_val).strip().lower().startswith('unnamed') and str(col_val).strip() != '')
        
        # Simple heuristic: prioritize rows with keywords, then rows with more meaningful names
        score = keyword_matches * 1000 + meaningful_cols_count # Give higher weight to keyword matches

        if score > max_meaningful_cols:
            max_meaningful_cols = score
            best_header_row = i
    
    # If no strong header found, default to 0
    if max_meaningful_cols <= 0 and len(df_raw) > 0 and 'Unnamed: 0' in df_raw.columns:
        return 0 # Likely already using a default header like pandas does, or truly no good header

    return best_header_row

# Function to standardize column names
def standardize_columns(df, column_mappings):
    """
    Renames DataFrame columns based on a mapping of potential names to standard names.
    Performs case-insensitive and strip-whitespace matching.
    """
    new_columns = {}
    df_cols = [str(col).strip() for col in df.columns] # Clean current DataFrame columns

    for standard_name, potential_names in column_mappings.items():
        found = False
        for potential_name in potential_names:
            # Try exact match (case-insensitive, stripped)
            if potential_name.strip() in df_cols:
                new_columns[potential_name.strip()] = standard_name
                found = True
                break
            # Try case-insensitive, stripped match
            for df_col_raw in df.columns:
                if str(df_col_raw).strip().lower() == potential_name.strip().lower():
                    new_columns[str(df_col_raw).strip()] = standard_name
                    found = True
                    break
            if found: break # Found a match for this standard_name

        # If still not found by direct mapping, check if "Unnamed" columns might contain keywords
        # This logic is complex and best handled by reading first data rows or relying on header finding
        # For now, if a direct match isn't found, we assume it's missing or will be filled by dummy data

    df = df.rename(columns=new_columns)
    # Drop any remaining 'Unnamed' columns that were not mapped
    df = df.loc[:, ~df.columns.astype(str).str.contains('Unnamed', case=False)]
    return df

def read_excel_file(file_path):
    """Read Excel file with error handling."""
    try:
        # Try reading with openpyxl engine first
        df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e1:
        try:
            # If that fails, try with xlrd engine
            df = pd.read_excel(file_path, engine='xlrd')
        except Exception as e2:
            print(f"Error reading {file_path} with both openpyxl and xlrd:")
            print(f"openpyxl error: {str(e1)}")
            print(f"xlrd error: {str(e2)}")
            return None
    return df

def convert_excel_to_csv():
    """Convert Excel files to CSV format with proper column mapping."""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Read the Excel files
    efficiency_df = read_excel_file('Garment ML/Efficiency.xlsx')
    defect_df = read_excel_file('Garment ML/Defect.xlsx')
    inventory_df = read_excel_file('Garment ML/Inventory.xlsx')
    
    if efficiency_df is None or defect_df is None or inventory_df is None:
        print("Failed to read one or more Excel files")
        return
    
    # Print column names for debugging
    print("\nEfficiency columns:", efficiency_df.columns.tolist())
    print("Defect columns:", defect_df.columns.tolist())
    print("Inventory columns:", inventory_df.columns.tolist())
    
    # Map the actual column names to our expected names
    efficiency_mapping = {
        'Style': 'style',
        'Operation': 'operation',
        'Operator': 'operator',
        'Date': 'date',
        'Efficiency': 'efficiency',
        'Output': 'output',
        'Working Hours': 'working_hours'
    }
    
    defect_mapping = {
        'Style': 'style',
        'Operation': 'operation',
        'Defect Type': 'defect_type',
        'Date': 'date',
        'Quantity': 'quantity'
    }
    
    inventory_mapping = {
        'Style': 'style',
        'Material': 'material',
        'Date': 'date',
        'Quantity': 'quantity',
        'Unit': 'unit'
    }
    
    # Rename columns
    efficiency_df = efficiency_df.rename(columns=efficiency_mapping)
    defect_df = defect_df.rename(columns=defect_mapping)
    inventory_df = inventory_df.rename(columns=inventory_mapping)
    
    # Convert date columns to datetime
    for df in [efficiency_df, defect_df, inventory_df]:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
    
    # Save to CSV
    efficiency_df.to_csv('data/efficiency_data.csv', index=False)
    defect_df.to_csv('data/defect_data.csv', index=False)
    inventory_df.to_csv('data/inventory_data.csv', index=False)
    
    # Generate production data
    generate_production_data(efficiency_df, defect_df, inventory_df)

def generate_production_data(efficiency_df, defect_df, inventory_df):
    """Generate production data based on the actual data structure."""
    # Get unique styles and dates
    styles = efficiency_df['style'].unique()
    dates = efficiency_df['date'].unique()
    
    # Create production data
    production_data = []
    
    for style in styles:
        for date in dates:
            # Get efficiency data for this style and date
            style_eff = efficiency_df[
                (efficiency_df['style'] == style) & 
                (efficiency_df['date'] == date)
            ]
            
            # Get defect data for this style and date
            style_defects = defect_df[
                (defect_df['style'] == style) & 
                (defect_df['date'] == date)
            ]
            
            # Get inventory data for this style and date
            style_inventory = inventory_df[
                (inventory_df['style'] == style) & 
                (inventory_df['date'] == date)
            ]
            
            # Calculate metrics
            total_output = style_eff['output'].sum() if not style_eff.empty else 0
            avg_efficiency = style_eff['efficiency'].mean() if not style_eff.empty else 0
            total_defects = style_defects['quantity'].sum() if not style_defects.empty else 0
            total_inventory = style_inventory['quantity'].sum() if not style_inventory.empty else 0
            
            # Calculate defect rate
            defect_rate = (total_defects / total_output * 100) if total_output > 0 else 0
            
            # Calculate inventory turnover
            inventory_turnover = (total_output / total_inventory) if total_inventory > 0 else 0
            
            production_data.append({
                'style': style,
                'date': date,
                'total_output': total_output,
                'avg_efficiency': avg_efficiency,
                'total_defects': total_defects,
                'defect_rate': defect_rate,
                'total_inventory': total_inventory,
                'inventory_turnover': inventory_turnover
            })
    
    # Convert to DataFrame and save
    production_df = pd.DataFrame(production_data)
    production_df.to_csv('data/production_data.csv', index=False)
    print("\nGenerated production data with columns:", production_df.columns.tolist())
    print("Sample data:")
    print(production_df.head())

if __name__ == '__main__':
    convert_excel_to_csv()
