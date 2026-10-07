import pandas as pd
import re
import os

# Define paths
# Assuming script is run from project root (tfm-mulcia-2025)
# Adjust if necessary based on where you run it from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'medication_mapping_v2.csv')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'rxnorm_to_atc5_mapping.xlsx')

def extract_atc5(justification):
    """
    Extracts the ATC code from the Justification_Ref string.
    Expected format: "WHO ATC Index [CODE]"
    """
    if pd.isna(justification):
        return None
    
    # Text is like "WHO ATC Index B01AD02"
    # We look for the code at the end
    match = re.search(r'WHO ATC Index\s+([A-Z0-9]+)', str(justification))
    if match:
        return match.group(1)
    return None

def main():
    print(f"Reading from: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        print(f"Error: File not found at {DATA_FILE}")
        # Try relative path if absolute fails (e.g. running from root)
        DATA_FILE_REL = 'data/medication_mapping_v2.csv'
        if os.path.exists(DATA_FILE_REL):
             print(f"Found at relative path: {DATA_FILE_REL}")
             df = pd.read_csv(DATA_FILE_REL)
        else:
            return
    else:
        df = pd.read_csv(DATA_FILE)

    print("Extracting ATC Level 5 codes...")
    df['ATC_Level5'] = df['Justification_Ref'].apply(extract_atc5)

    # Select and reorder columns as requested
    # User focused on mapping clarity
    # code, name, ATC_Level5, ATC_Level3, Therapeutic_Group_Name
    cols = ['code', 'name', 'ATC_Level5', 'ATC_Level3', 'Therapeutic_Group_Name', 'Justification_Ref']
    
    # Verify we have all columns
    existing_cols = [c for c in cols if c in df.columns]
    final_df = df[existing_cols]

    print(f"Saving to: {OUTPUT_FILE}")
    final_df.to_excel(OUTPUT_FILE, index=False)
    print("Done!")
    print(final_df.head())

if __name__ == "__main__":
    main()
