import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Determine project root safely
env_path = find_dotenv()
if env_path:
    PROJECT_ROOT = Path(env_path).parent.resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

print(f"PROJECT_ROOT set to: {PROJECT_ROOT}")

DATA_DIR = PROJECT_ROOT / 'data'
DATA_FILE = DATA_DIR / 'synthea_30k.csv'
CODES_LIST = DATA_DIR / 'codificaciones_identificadas.xlsx'
OUTPUT_FILE = DATA_DIR / 'synthea_30k_loinc_only.csv'

def main():
    print(f"Loading codificaciones from: {CODES_LIST}")
    if not CODES_LIST.exists():
        print(f"Error: {CODES_LIST} not found.")
        return
        
    df_codes = pd.read_excel(CODES_LIST)
    
    # Identify LOINC codes 
    # Based on the system we saw before, LOINC may be explicitly labeled or we match the format.
    # We will use the 'SISTEMA_CODIFICACION' column if it exists and accurately reflects LOINC.
    loinc_codes = df_codes[df_codes['SISTEMA_CODIFICACION'] == 'LOINC']['code'].tolist()
    
    # Prefix with 'C-' as it appears in the dataset
    target_cols = [f"C-{code}" if not str(code).startswith('C-') else str(code) for code in loinc_codes]
    
    # Adding essential identifiers if needed (e.g. PATIENT id)
    # The user asked for "exclusively LOINC", but usually an ID is needed for joining.
    # We will include 'PATIENT' if it exists.
    
    print(f"Loading dataset from: {DATA_FILE}")
    if not DATA_FILE.exists():
        print(f"Error: Dataset {DATA_FILE} not found!")
        return
        
    df = pd.read_csv(DATA_FILE, low_memory=False)
    
    # Filter only the columns that actually exist in the dataframe
    existing_target_cols = [col for col in target_cols if col in df.columns]
    
    print(f"Found {len(existing_target_cols)} LOINC columns out of {len(target_cols)} expected in the dataset.")
    
    cols_to_extract = existing_target_cols
    
    # Include PATIENT column if it exists to allow merging later
    if 'PATIENT' in df.columns:
        print("Including 'PATIENT' column for joining purposes.")
        cols_to_extract = ['PATIENT'] + existing_target_cols
        
    df_loinc = df[cols_to_extract]
    
    print(f"Saving extracted LOINC data to: {OUTPUT_FILE}")
    df_loinc.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Done! Saved {df_loinc.shape[0]} rows and {df_loinc.shape[1]} columns.")

if __name__ == "__main__":
    main()
