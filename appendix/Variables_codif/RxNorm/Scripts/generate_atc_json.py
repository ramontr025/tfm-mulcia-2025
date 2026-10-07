import pandas as pd
import json
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# We can use the Excel file we just created which already has ATC_Level5 extracted
INPUT_FILE = os.path.join(os.path.dirname(__file__), 'rxnorm_to_atc5_mapping.xlsx')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'rxnorm_atc_mapping.json')

def generate_json_mapping():
    print(f"Reading data from: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
         print(f"Error: File not found at {INPUT_FILE}")
         return

    df = pd.read_excel(INPUT_FILE)
    
    # The structure requested:
    # {
    #   "Group Name": {
    #       "ATC3": {
    #           "ATC5": ["RxNorm1", "RxNorm2"]
    #       }
    #   }
    # }
    
    TYPES_FILE = os.path.join(os.path.dirname(__file__), 'rxnorm_data_types_report.xlsx')
    if os.path.exists(TYPES_FILE):
        df_types = pd.read_excel(TYPES_FILE)
        code_types = dict(zip(df_types['RxNorm_Code'], df_types['Inferred_Data_Type']))
    else:
        code_types = {}

    mapping = {}
    
    # Iterate through the rows and build the nested dictionary
    for _, row in df.iterrows():
        atc3 = str(row['ATC_Level3']).strip()
        group_name = str(row['Therapeutic_Group_Name']).strip()
        atc5 = str(row['ATC_Level5']).strip() if pd.notna(row['ATC_Level5']) else "UNKNOWN"
        rxnorm = str(row['code']).strip()
        
        c_type = code_types.get(rxnorm, 'Empty (All Null)')
        
        # Las vacías ("Empty (All Null)") se tratan igual que las binarias por defecto.
        # Las Categóricas y Numéricas van en una entrada única separada.
        if c_type in ['Categorical (Nominal)', 'High Cardinality Categorical / Text', 'Continuous Numeric']:
            unique_group = f"Standalone Variables"
            unique_atc3 = rxnorm
            unique_atc5 = rxnorm
            
            if unique_group not in mapping:
                mapping[unique_group] = {}
            if unique_atc3 not in mapping[unique_group]:
                mapping[unique_group][unique_atc3] = {}
            if unique_atc5 not in mapping[unique_group][unique_atc3]:
                mapping[unique_group][unique_atc3][unique_atc5] = []
            if rxnorm not in mapping[unique_group][unique_atc3][unique_atc5]:
                mapping[unique_group][unique_atc3][unique_atc5].append(rxnorm)
        else:
            # Excluir 'nan' y 'Z' solo para las booleanas que intentan formar grupos
            if atc3.lower() == 'nan' or atc3.upper() == 'Z':
                continue
                
            # Initialize Group Name key if not exists
            if group_name not in mapping:
                mapping[group_name] = {}
                
            # Initialize ATC3 key inside Group Name if not exists
            if atc3 not in mapping[group_name]:
                mapping[group_name][atc3] = {}
                
            # Initialize ATC5 key inside ATC3 if not exists
            if atc5 not in mapping[group_name][atc3]:
                mapping[group_name][atc3][atc5] = []
                
            # Add RxNorm to the list
            if rxnorm not in mapping[group_name][atc3][atc5]:
                 mapping[group_name][atc3][atc5].append(rxnorm)

    print(f"Saving JSON to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(mapping, f, indent=4)
        
    print("Generation complete.")
    
    # Print a small sample
    print("\nSample of generated JSON:")
    sample_key = list(mapping.keys())[0]
    print(json.dumps({sample_key: mapping[sample_key]}, indent=4))

if __name__ == "__main__":
    generate_json_mapping()
