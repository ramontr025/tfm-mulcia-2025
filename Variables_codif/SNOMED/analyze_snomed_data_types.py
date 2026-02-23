import pandas as pd
import json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

PROJECT_ROOT = Path(find_dotenv()).parent.resolve()
print(f"PROJECT_ROOT set to: {PROJECT_ROOT}")

DATA_DIR = PROJECT_ROOT / 'data'

MAPPINGS = [
    PROJECT_ROOT / 'Variables_codif/SNOMED/Prueba_ECL/feature_mapping_disorder_finding.json',
    PROJECT_ROOT / 'Variables_codif/SNOMED/Prueba_ECL/feature_mapping_procedures_therapies.json',
    PROJECT_ROOT / 'Variables_codif/SNOMED/Prueba_ECL/feature_mapping_resto.json'
]

DATA_FILE = DATA_DIR / 'synthea_30k.csv'
SNOMED_CODES_LIST = DATA_DIR / 'codificaciones_identificadas.xlsx'
OUTPUT_FILE = PROJECT_ROOT / 'Variables_codif/SNOMED/Prueba_ECL/snomed_data_types_report.xlsx'

def get_inferred_type(unique_vals):
    if len(unique_vals) == 0:
        return 'Empty (All Null)'
        
    str_vals = {str(v).lower().strip() for v in unique_vals}
    
    # Verdadero
    if str_vals.issubset({'true', '1', '1.0', 'positive', 'yes'}):
        return 'Pure Boolean (True only)'
        
    # Verdadero/falso
    if str_vals.issubset({'true', 'false', '1', '0', '1.0', '0.0', 'yes', 'no', 'positive', 'negative'}):
        return 'Boolean (True/False)'
        
    # Si no es booleano, ¿qué es?
    if len(str_vals) <= 15:
        return 'Categorical (Nominal)'
        
    # Si muchos, puede ser numérico o nominal de alta cardinalidad
    try:
        [float(v) for v in unique_vals]
        return 'Continuous Numeric'
    except ValueError:
        return 'High Cardinality Categorical / Text'

def main():
    print(f"Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    
    print(f"Loading SNOMED dictionary {SNOMED_CODES_LIST}...")
    df_dict = pd.read_excel(SNOMED_CODES_LIST)
    
    # Map code -> name for readability
    code_to_name = dict(zip(df_dict['code'], df_dict['name']))
    
    print("Collecting active codes from JSON mappings...")
    active_codes = set()
    for m in MAPPINGS:
        with open(m, 'r') as f:
            mapping = json.load(f)
            for group, codes in mapping.items():
                for code in codes.keys():
                    active_codes.add(f"C-{code}")
                    
    target_cols = [c for c in active_codes if c in df.columns]
    print(f"Found {len(target_cols)} distinct targeted SNOMED columns in dataset.")
    
    report_data = []
    
    for i, col in enumerate(target_cols):
        # Droping NAs to just see what the actual populated values look like
        unique_vals = df[col].dropna().unique().tolist()
        
        inferred_type = get_inferred_type(unique_vals)
        num_non_nulls = df[col].notna().sum()
        
        # Convert list of values to a clean string for excel
        vals_str = ", ".join([str(v) for v in unique_vals])
        
        # Get the human-readable name if available
        name = code_to_name.get(col, 'Unknown')
        
        report_data.append({
            'SNOMED_Code': col,
            'Provided_Name': name,
            'Inferred_Data_Type': inferred_type,
            'Unique_Values': vals_str,
            'Non_Null_Count': num_non_nulls
        })
        
    # Convert to df and sort: put the weird ones (not purely 'True') at the top for visibility
    report_df = pd.DataFrame(report_data)
    
    # Sort such that nominal/categorical are first, then boolean, etc.
    type_priority = {
        'Categorical (Nominal)': 1,
        'High Cardinality Categorical / Text': 2,
        'Continuous Numeric': 3,
        'Boolean (True/False)': 4,
        'Pure Boolean (True only)': 5,
        'Empty (All Null)': 6
    }
    
    report_df['sort_priority'] = report_df['Inferred_Data_Type'].map(type_priority)
    report_df.sort_values(by=['sort_priority', 'Non_Null_Count'], ascending=[True, False], inplace=True)
    report_df.drop(columns=['sort_priority'], inplace=True)
    
    print(f"\n--- Summary of Data Types ---")
    print(report_df['Inferred_Data_Type'].value_counts())
    
    try:
        report_df.to_excel(OUTPUT_FILE, index=False)
        print(f"\nDetailed report saved to {OUTPUT_FILE}")
    except PermissionError:
        print(f"\nWARNING: Could not save report to {OUTPUT_FILE} because it is open. Continuing with dataset generation...")

    print("\nApplying data typing to the raw dataset columns...")
    for _, row in report_df.iterrows():
        col = row['SNOMED_Code']
        col_type = row['Inferred_Data_Type']
        
        if col_type in ['Pure Boolean (True only)', 'Boolean (True/False)']:
            df[col] = df[col].apply(lambda x: 1 if pd.notna(x) and str(x).lower().strip() in ['true', '1', '1.0', 'positive', 'yes'] else 0)
        elif col_type == 'Continuous Numeric':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Apply JSON Groupings
    print("\nApplying JSON Groupings to generate final features...")
    df_final = pd.DataFrame()
    total_new_features = 0
    
    for mapping_file in MAPPINGS:
        try:
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
        except Exception as e:
            print(f"WARNING: Could not read {mapping_file}: {e}")
            continue
            
        for group_name, codes_dict in mapping.items():
            new_col_name = f"{group_name.replace(' ', '_').replace('(', '').replace(')', '')}"
            cols_to_combine = [f"C-{c}" for c in codes_dict.keys() if f"C-{c}" in df.columns]
            
            if not cols_to_combine:
                continue
                
            if len(cols_to_combine) == 1:
                # Single variable (like rest/standalones), copy it preserving its type/categories
                df_final[new_col_name] = df[cols_to_combine[0]]
            else:
                # Multiple variables (disorders/procedures mapped to 1/0), take the logical max/OR
                df_final[new_col_name] = df[cols_to_combine].max(axis=1)
                
            total_new_features += 1

    OUTPUT_CSV = DATA_DIR / 'synthea_30k_snomed_typed.csv'
    print(f"\nConsolidation complete. Created {total_new_features} grouped features.")
    print(f"Saving final dataset (only JSON variables) to {OUTPUT_CSV}...")
    df_final.to_csv(OUTPUT_CSV, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
