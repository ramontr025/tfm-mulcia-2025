import json
import pandas as pd
import numpy as np

def load_json_mapping(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return mapping

def extract_flat_mapping(mapping):
    """
    Convierte el diccionario anidado del JSON a un diccionario plano agrupado por nivel 2 (ej. B01):
    { 'ATC2_CODE' : ['C-123', 'C-456', ...] }
    """
    flat_mapping = {}
    for category, subcategories in mapping.items():
        for atc2_code, codes_dict in subcategories.items():
            if atc2_code not in flat_mapping:
                flat_mapping[atc2_code] = []
            for target_var, original_vars in codes_dict.items():
                flat_mapping[atc2_code].extend(original_vars)
    return flat_mapping

def main():
    json_path = 'Variables_codif/RxNorm/JSON/rxnorm_atc_mapping.json'
    csv_path = 'data/synthea_30k.csv'
    output_path = 'Variables_codif/RxNorm/rxnorm_mapped_data.csv'

    print("Cargando JSON...")
    mapping = load_json_mapping(json_path)
    flat_mapping = extract_flat_mapping(mapping)
    
    # Recolectar todas las variables RxNorm originales necesarias
    all_needed_vars = set()
    for vars_list in flat_mapping.values():
        all_needed_vars.update(vars_list)
        
    all_needed_vars = list(all_needed_vars)

    print(f"Leyendo CSV ({csv_path})...")
    # Leer ptnum y las variables necesarias (ignoramos las que no estén en el CSV)
    cols_to_read = ['ptnum'] + all_needed_vars
    
    try:
        # Leer primero 1 fila para ver qué columnas existen realmente
        sample_df = pd.read_csv(csv_path, nrows=1)
        existing_cols = [col for col in cols_to_read if col in sample_df.columns]
        missing_cols = [col for col in cols_to_read if col not in sample_df.columns]
        
        if len(missing_cols) > 0:
            sample_missing = missing_cols[:min(5, len(missing_cols))]
            print(f"Aviso: Hay {len(missing_cols)} variables en el JSON que NO están en el CSV (ej: {sample_missing}). Se ignorarán.")
            
        full_df = pd.read_csv(csv_path)
        df = full_df[existing_cols].copy()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error al leer el CSV o al filtrar columnas: {e}")
        return

    # Comprobación de que las variables (excepto ptnum) sean True, False, o NA
    invalid_cols = []
    for col in existing_cols:
        if col == 'ptnum':
            continue
            
        # Obtenemos los valores únicos descartando NAs
        unique_vals = df[col].dropna().unique()
        # Verificamos si hay valores que no sean True o False (o 1.0, 0.0 debido al parseo, o strings 'True'/'False')
        # Dependiendo de cómo pandas lo haya leído puede ser bool, numeric o object.
        for val in unique_vals:
            if not (val in [True, False, 1, 0, 1.0, 0.0, 'True', 'False', 'true', 'false']):
                invalid_cols.append(col)
                break
                
    if invalid_cols:
        print("\n" + "="*80)
        print("¡ATENCIÓN! Se han encontrado variables que no son puramente True/False/NA.")
        print("Revisa estas variables: ")
        print(invalid_cols)
        print("="*80 + "\n")

    print("Procesando las agrupaciones...")
    result_df = pd.DataFrame({'ptnum': df['ptnum']})
    
    for target_var, original_vars in flat_mapping.items():
        # Filtrar solo las variables que realmente existen en el dataframe
        valid_vars = [v for v in original_vars if v in df.columns]
        
        if not valid_vars:
            continue
            
        # Extraemos el subset de columnas para esta variable objetivo
        subset = df[valid_vars].copy()
        
        # Mapeamos a boolean real si no lo es (por seguridad si vienen como strings o números)
        subset = subset.replace({1.0: True, 0.0: False, 1: True, 0: False, 'True': True, 'False': False, 'true': True, 'false': False})
        
        # Lógica de agregación fila por fila:
        # Si al menos un True -> 1
        # Si todo NA -> NA
        # Si todo False -> 0
        # Mezcla False/NA -> NA
        
        # any() devuelve True si hay al menos un True en la fila. (Omitiendo NAs automáticamente)
        has_true = subset.any(axis=1)
        
        # isna().all() devuelve True si todas las columnas de la fila son NA
        all_na = subset.isna().all(axis=1)
        
        # El resto de casos (no todo NA, no hay True) significa que solo hay False o False+NA
        # Queremos distinguir si es TODO falso o mezcla.
        
        # sum() cuenta cuántos no son NA
        not_na_count = subset.notna().sum(axis=1)
        # Queremos todo falso: no hay True, y el conteo de no-NAs debe ser igual al total de columnas
        # Pero si solo tenemos algunas columanas y puede haber mezcla False/NA, la regla es:
        # If any is NA AND no True is present -> NA (because mixing False+NA => NA)
        # So it's only 0 when ALL values are False AND NO value is NA.
        all_false = (~has_true) & (not_na_count == len(valid_vars))
        
        # Asignamos
        # base: NA por defecto
        # numpy select es eficiente para esto
        conditions = [
            has_true,
            all_false
        ]
        choices = [
            1,
            0
        ]
        
        result_df[target_var] = np.select(conditions, choices, default=np.nan)

    # Imprimir un resumen
    print("\nResumen de variables procesadas (muestra de las primeras 5):")
    for col in list(result_df.columns)[1:6]:  # ignorar ptnum
        vc = result_df[col].value_counts(dropna=False)
        print(f"{col}:")
        print(vc)
        print("-" * 20)

    print(f"\nGuardando el resultado en {output_path}...")
    result_df.to_csv(output_path, index=False)
    print("¡Proceso completado con éxito!")

if __name__ == '__main__':
    main()
