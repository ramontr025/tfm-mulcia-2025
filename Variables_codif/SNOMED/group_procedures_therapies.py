import pandas as pd
import requests
import json
import time
from collections import defaultdict

# --- Configuration ---
INPUT_FILE = 'data/synthea_30k.csv'
AUX_FILE = 'Variables_codif/SNOMED/Prueba_ECL/auxiliar.xlsx'
MAPPING_FILE = 'Variables_codif/SNOMED/Prueba_ECL/feature_mapping_procedures_therapies.json'
REPORT_FILE = 'Variables_codif/SNOMED/Prueba_ECL/variables_seleccionadas_procedures_therapies.xlsx'

BASE_URL = 'http://localhost:8080/MAIN/concepts'
FREQ_THRESHOLD_PERCENT = 3.0
PROCEDURE_MIN_PERCENT = 0.1  # Rule from Norma_mantener.md

# Discard roots that are too abstract
IGNORE_ROOTS = {'138875005', '404684003', '64572001', '71388002', '363787002', '272379006', '243796009', '71388002'}

# --- Manual Overrides from user feedback ---
MANUAL_OVERRIDES = {
    '243174005': 'Asistencia_Respiratoria',
    '40617009': 'Asistencia_Respiratoria',
    '26763009': 'Asistencia_Respiratoria',
    '60027007': 'rx_otros',
    '19490002': 'rx_otros',
    '74016001': 'rx_otros',
    '268425006': 'rx_otros',
    '447365002': 'Procedure_on_heart',
    '449214001': '__STANDALONE__',
    '305351004': '__STANDALONE__',
    '305340004': '__STANDALONE__',
    '88039007': '__STANDALONE__',
}

def is_positive(val):
    if pd.isna(val): return False
    s_val = str(val).lower().strip()
    if s_val in ['true', '1', '1.0', 'positive', 'abnormal', 'ordered', 'yes']: return True
    if s_val in ['false', '0', '0.0', 'negative', 'normal', 'no', 'none']: return False
    return True

def get_ancestors_api(code):
    try:
        res = requests.get(BASE_URL, params={'ecl': f">{code}", 'limit': 200}, timeout=10).json()
        return [(a['conceptId'], a['pt']['term']) for a in res.get('items', [])]
    except:
        return []

def main():
    print("Loading datasets...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    num_patients = len(df)
    
    df_aux = pd.read_excel(AUX_FILE)
    
    # FILTER 1: Get Procedures and Therapies according to Norma_mantener.md
    # Therapies: keep all
    df_therapies = df_aux[df_aux['Semantic_Tag'] == 'regime/therapy'].copy()
    
    # Procedures: keep if % >= 0.1
    df_aux['pct'] = pd.to_numeric(df_aux['%_No_Nulos'], errors='coerce')
    df_procedures = df_aux[(df_aux['Semantic_Tag'] == 'procedure') & (df_aux['pct'] >= PROCEDURE_MIN_PERCENT)].copy()
    
    # Combine
    df_target = pd.concat([df_therapies, df_procedures])
    print(f"Total Therapies kept: {len(df_therapies)}")
    print(f"Total Procedures kept (>= {PROCEDURE_MIN_PERCENT}%): {len(df_procedures)}")
    
    # Exclude day packs just in case
    df_target = df_target[~df_target['Provided_Name'].str.lower().str.contains('day pack', na=False)]
    
    feature_cols = df_target['Original_Col_Name'].dropna().tolist()
    # Filter to ensure they actually exist in the synthea CSV
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    df_target['code_str'] = df_target['Code'].astype(str).str.replace('C-', '')
    code_to_name = pd.Series(df_target.Provided_Name.values, index=df_target.code_str).to_dict()
    
    print(f"Final valid columns matching Target Rules: {len(feature_cols)}")

    # Precompute boolean indices
    print("Precomputing boolean indices...")
    var_presence = {}
    for col in feature_cols:
        var_presence[col] = df[col].apply(is_positive)
        
    freq_threshold_count = int((FREQ_THRESHOLD_PERCENT / 100.0) * num_patients)

    # Extract specific >= 3% features and Overrides
    standalone_codes = []
    codes_to_group = []
    
    group_mapping = defaultdict(dict)
    report_data = []

    for col in feature_cols:
        code = str(col).replace('C-', '')
        name = code_to_name.get(code, f"Unknown ({code})")
        valid_name = name.strip()
        
        if code in MANUAL_OVERRIDES:
            target = MANUAL_OVERRIDES[code]
            if target == '__STANDALONE__':
                group_mapping[valid_name][code] = name
                report_data.append({'Original_Code': col, 'Name': name, 'Status': 'Manual Override (Standalone)', 'Assigned_Group': valid_name, 'Semantic_Tag': df_target[df_target['Original_Col_Name']==col]['Semantic_Tag'].values[0]})
            else:
                group_name = f"Group_{target}" if not target.startswith("Group_") else target
                group_mapping[group_name][code] = name
                report_data.append({'Original_Code': col, 'Name': name, 'Status': 'Manual Override', 'Assigned_Group': group_name, 'Semantic_Tag': df_target[df_target['Original_Col_Name']==col]['Semantic_Tag'].values[0]})
            continue
            
        count = var_presence[col].sum()
        if count >= freq_threshold_count:
            group_mapping[valid_name][code] = name
            report_data.append({'Original_Code': col, 'Name': name, 'Status': f'Specific (>= {FREQ_THRESHOLD_PERCENT}%)', 'Assigned_Group': valid_name, 'Semantic_Tag': df_target[df_target['Original_Col_Name']==col]['Semantic_Tag'].values[0]})
        else:
            codes_to_group.append((col, code, name))
            
    print(f"Specific/Manual Features identified: {len(report_data)}")
    print(f"Features requiring dynamic grouping (< {FREQ_THRESHOLD_PERCENT}%): {len(codes_to_group)}")

    # Fetch trees
    print("Fetching ancestor trees from Snowstorm...")
    ancestor_map = defaultdict(list)
    ancestor_names = {}
    
    start_time = time.time()
    for i, (col, code, name) in enumerate(codes_to_group):
        if i % 25 == 0: print(f"  {i}/{len(codes_to_group)} codes fetched...")
        ancestors = get_ancestors_api(code)
        for anc_id, anc_name in ancestors:
            if anc_id not in IGNORE_ROOTS:
                ancestor_map[anc_id].append(col)
                ancestor_names[anc_id] = anc_name
                
    print(f"Fetched trees in {time.time() - start_time:.2f} seconds.")

    # Evaluate Groups
    valid_candidates = []
    for anc_id, cols_under_anc in ancestor_map.items():
        if len(cols_under_anc) < 2: continue
        
        combined_presence = var_presence[cols_under_anc[0]].copy()
        for c in cols_under_anc[1:]:
            combined_presence = combined_presence | var_presence[c]
            
        if combined_presence.sum() >= freq_threshold_count:
            valid_candidates.append({
                'anc_id': anc_id,
                'name': ancestor_names[anc_id],
                'cols': set(cols_under_anc),
                'specificity_score': len(cols_under_anc)
            })

    # Sort candidates by Specificity
    valid_candidates.sort(key=lambda x: x['specificity_score'])
    print(f"Found {len(valid_candidates)} valid SNOMED groups >= {FREQ_THRESHOLD_PERCENT}% threshold.")

    # Greedy Assignment
    uncovered_cols = set([c[0] for c in codes_to_group])
    dynamic_groups_created = 0
    
    for candidate in valid_candidates:
        if not uncovered_cols: break
        intersection = candidate['cols'].intersection(uncovered_cols)
        
        if len(intersection) >= 1:
            subset_presence = None
            for c in intersection:
                if subset_presence is None: subset_presence = var_presence[c].copy()
                else: subset_presence = subset_presence | var_presence[c]
                
            if subset_presence is not None and subset_presence.sum() >= freq_threshold_count:
                group_name = f"Group_{candidate['name'].replace(' ', '_').replace('(', '').replace(')', '')}"
                for col in intersection:
                    code = str(col).replace('C-', '')
                    name = code_to_name.get(code, f"Unknown ({code})")
                    group_mapping[group_name][code] = name
                    sem_val = df_target[df_target['Original_Col_Name']==col]['Semantic_Tag'].values[0]
                    report_data.append({'Original_Code': col, 'Name': name, 'Status': 'Dynamic Grouped', 'Assigned_Group': group_name, 'Semantic_Tag': sem_val})
                    uncovered_cols.remove(col)
                dynamic_groups_created += 1

    # Miscelanea
    if uncovered_cols:
        print(f"Rare codes relegated to Miscelanea (too rare to group): {len(uncovered_cols)}")
        for col in uncovered_cols:
            code = str(col).replace('C-', '')
            name = code_to_name.get(code, f"Unknown ({code})")
            sem_val = df_target[df_target['Original_Col_Name']==col]['Semantic_Tag'].values[0]
            group_mapping['Group_Rare_Procedures_Therapies'][code] = name
            report_data.append({'Original_Code': col, 'Name': name, 'Status': 'Rare / Ungroupable', 'Assigned_Group': 'Group_Rare_Procedures_Therapies', 'Semantic_Tag': sem_val})

    with open(MAPPING_FILE, 'w') as f:
        json.dump(group_mapping, f, indent=4)
        
    pd.DataFrame(report_data).to_excel(REPORT_FILE, index=False)
    
    print("\n--- Summary ---")
    print(f"Original Analyzed Features: {len(feature_cols)}")
    print(f"Final Mapped Features (dict output keys): {len(group_mapping)}")
    print(f"Dynamic Groups Created: {dynamic_groups_created}")
    print(f"Files written to:\n  JSON: {MAPPING_FILE}\n  EXCEL: {REPORT_FILE}")

if __name__ == "__main__":
    main()
