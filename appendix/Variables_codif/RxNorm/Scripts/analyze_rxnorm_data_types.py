import pandas as pd
import json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Determine project root safely
env_path = find_dotenv()
if env_path:
    PROJECT_ROOT = Path(env_path).parent.resolve()
else:
    # If no .env, fallback to relative path assuming script is in Variables_codif/RxNorm
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

print(f"PROJECT_ROOT set to: {PROJECT_ROOT}")

DATA_DIR = PROJECT_ROOT / 'data'
RXNORM_MAPPING = PROJECT_ROOT / 'Variables_codif/RxNorm/rxnorm_atc_mapping.json'
RXNORM_CODES_LIST = PROJECT_ROOT / 'Variables_codif/RxNorm/rxnorm_to_atc5_mapping.xlsx'

DATA_FILE = DATA_DIR / 'synthea_30k.csv'
OUTPUT_FILE = PROJECT_ROOT / 'Variables_codif/RxNorm/rxnorm_data_types_report.xlsx'

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
    if not DATA_FILE.exists():
        print(f"Error: Dataset {DATA_FILE} not found!")
        return
        
    df = pd.read_csv(DATA_FILE, low_memory=False)
    
    print(f"Loading RxNorm dictionary {RXNORM_CODES_LIST}...")
    try:
        df_dict = pd.read_excel(RXNORM_CODES_LIST)
        code_to_name = dict(zip(df_dict['code'], df_dict['name']))
    except Exception as e:
        print(f"Could not load dictionary: {e}")
        code_to_name = {}
    
    print("Collecting active codes from JSON mapping...")
    active_codes = set()
    with open(RXNORM_MAPPING, 'r') as f:
        mapping = json.load(f)
        # Iterate over Group Name
        for group_name, atc3_dict in mapping.items():
            # Iterate over ATC3
            for atc3, atc5_dict in atc3_dict.items():
                # Iterate over ATC5
                for atc5, codes in atc5_dict.items():
                    for code in codes:
                        active_codes.add(code)
                    
    target_cols = [c for c in active_codes if c in df.columns]
    print(f"Found {len(target_cols)} distinct targeted RxNorm columns in dataset out of {len(active_codes)} in mapping.")
    
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
            'RxNorm_Code': col,
            'Provided_Name': name,
            'Inferred_Data_Type': inferred_type,
            'Unique_Values': vals_str,
            'Non_Null_Count': num_non_nulls
        })
        
    # Convert to df and sort: put the weird ones (not purely 'True') at the top for visibility
    if len(report_data) == 0:
        print("No columns to report!")
        return
        
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
    
    report_df['sort_priority'] = report_df['Inferred_Data_Type'].map(type_priority).fillna(99)
    report_df.sort_values(by=['sort_priority', 'Non_Null_Count'], ascending=[True, False], inplace=True)
    report_df.drop(columns=['sort_priority'], inplace=True)
    
    print(f"\n--- Summary of Data Types ---")
    print(report_df['Inferred_Data_Type'].value_counts())
    
    try:
        report_df.to_excel(OUTPUT_FILE, index=False)
        print(f"\nDetailed report saved to {OUTPUT_FILE}")
    except PermissionError:
        print(f"\nWARNING: Could not save report to {OUTPUT_FILE} because it is open.")

    print("\nApplying data typing to the raw dataset columns...")
    for _, row in report_df.iterrows():
        col = row['RxNorm_Code']
        col_type = row['Inferred_Data_Type']
        
        # Encodificamos las booleanas (presencia = 1, resto/nulo = 0)
        # INCLUYE LAS VACÍAS para que queden como 0 en lugar de NaN
        if col_type in ['Pure Boolean (True only)', 'Boolean (True/False)', 'Empty (All Null)']:
            df[col] = df[col].apply(lambda x: 1 if pd.notna(x) and str(x).lower().strip() in ['true', '1', '1.0', 'positive', 'yes'] else 0)
        # Para las numéricas continuas y categóricas nominales dejamos los datos tal cual (incluyendo NAs)
        elif col_type == 'Continuous Numeric':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Si es Categorical o High Cardinality, se queda como string/tipo original.

    # Aplicar las agrupaciones del JSON (Group Name -> ATC3 -> ATC5 -> [RxNorm codes])
    print("\nApplying JSON Groupings to generate final features...")
    df_final = pd.DataFrame()
    total_new_features = 0
    
    with open(RXNORM_MAPPING, 'r') as f:
        mapping = json.load(f)
        
    for group_name, atc3_dict in mapping.items():
        for atc3, atc5_dict in atc3_dict.items():
            # Inicializamos lista de todos los códigos para este ATC3
            codes_for_atc3 = []
            for atc5, codes in atc5_dict.items():
                for code in codes:
                    codes_for_atc3.append(code)
                    
            # Filtramos solo los códigos que realmente existen en el dataset crudo
            valid_cols = [c for c in codes_for_atc3 if c in df.columns]
            
            if not valid_cols:
                continue
                
            # Si es el grupo especial ("Standalone Variables"), usamos el nombre de la variable cruda
            new_col_name = f"ATC3_{atc3}" if not atc3.startswith('C-') else atc3
            
            if len(valid_cols) == 1:
                # Si hay solo un elemento (como los standalone numéricos/categóricos o ATC3s con solo 1 droga binaria)
                # se copia tal cual para preservar su tipo original (int, float, object).
                df_final[new_col_name] = df[valid_cols[0]]
            else:
                # Filtramos a numérico ignorando strings para el MAX
                df_numeric_cols = df[valid_cols].apply(pd.to_numeric, errors='coerce')
                # Hacemos max() para agrupar (presencia lógica)
                df_final[new_col_name] = df_numeric_cols.max(axis=1)
                
                # Formateamos NAs a 0 y como son variables que eran binarias (0, 1), casteamos a int para evitar 0.0
                df_final[new_col_name] = df_final[new_col_name].fillna(0).astype(int)
            
            total_new_features += 1

    OUTPUT_CSV = DATA_DIR / 'synthea_30k_rxnorm_typed.csv'
    print(f"\nConsolidation complete. Created {total_new_features} grouped features.")
    print(f"Saving final dataset (only JSON variables) to {OUTPUT_CSV}...")
    df_final.to_csv(OUTPUT_CSV, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
