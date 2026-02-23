import pandas as pd
import json

AUX_FILE = 'Variables_codif/SNOMED/Prueba_ECL/auxiliar.xlsx'
MAPPING_FILE = 'Variables_codif/SNOMED/Prueba_ECL/feature_mapping_resto.json'

target_codes = [
    'C-703151001', 'C-399211009', 'C-429007001', 'C-446654005', 
    'C-261352009', 'C-840544004', 'C-66348005', 'C-103579009', 
    'C-125680007', 'C-186034007', 'C-263495000', 'C-398070004', 
    'C-424144002', 'C-161621004'
]

def main():
    print("Loading auxiliary Excel...")
    df_aux = pd.read_excel(AUX_FILE)
    
    # Filter only target codes
    df_resto = df_aux[df_aux['Original_Col_Name'].isin(target_codes)].copy()
    print(f"Found {len(df_resto)} out of {len(target_codes)} specific variables.")
    
    group_mapping = {}
    
    # Map each code as a standalone group (1-to-1)
    for _, row in df_resto.iterrows():
        col_name = row['Original_Col_Name']
        code = str(row['Code']).replace('C-', '')
        name = str(row['Provided_Name']).strip()
        
        # Each gets its own unique group name
        group_mapping[name] = {code: name}
        
    with open(MAPPING_FILE, 'w') as f:
        json.dump(group_mapping, f, indent=4)
        
    print(f"Created {MAPPING_FILE} with {len(group_mapping)} standalone groups.")

if __name__ == "__main__":
    main()
