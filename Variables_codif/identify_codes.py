import pandas as pd
import re
import os

# --- Configuración ---
DATA_PATH = 'data/synthea-lc-dataset-codes.csv'
OUTPUT_EXCEL = 'data/codificaciones_identificadas.xlsx'

def identify_coding_system(code, name):
    """
    Identifica el sistema de codificación basado en patrones del código y la descripción.
    """
    code_str = str(code).strip()
    name_str = str(name).lower()
    
    # 1. LOINC
    # Patrón: nnnnn-n (dígitos separados por guión, ultimo dígito es verificador)
    # Ejemplo: 72514-3
    if re.match(r'^\d+-\d+$', code_str):
        return "LOINC", "Logical Observation Identifiers Names and Codes"

    # 2. CVX (Vaccines)
    # Patrón: Numérico, corto (generalmente < 4 dígitos, a veces más), descripción contiene "vaccine"
    # Ejemplo: 140 (Influenza)
    # Nota: Synthea usa códigos CVX para vacunas.
    if code_str.isdigit() and ("vaccine" in name_str or "vax" in name_str or "immunization" in name_str):
         return "CVX", "Vaccine Administered"
        
    # 3. RxNorm (Medications)
    # Patrón: Numérico (SCTID de concepto o RxCUI). Synthea a veces mezcla.
    # Pero si la descripción parece un medicamento (tiene dosis, forma farmacéutica).
    # Palabras clave fuertes: Tablet, Injection, MG, ML, Solution, Suspension.
    med_keywords = ['tablet', 'injection', 'mg', 'ml', 'solution', 'suspension', 'capsule', 'inhaler', 'spray', 'cream', 'ointment', 'pen', 'syringe']
    if code_str.isdigit() and any(k in name_str for k in med_keywords) and "vaccine" not in name_str:
        return "RxNorm", "RxNorm (Medication)"

    # 4. SNOMED CT
    # Patrón: Numérico (SCTID), 6-18 dígitos.
    # Synthea usa mayoritariamente SNOMED-CT para conditions, findings, procedures.
    # Semantic tags típicos: (disorder), (finding), (procedure), (regime/therapy), (situation).
    if code_str.isdigit():
        # Verificación por semantic tags comunes en SNOMED
        if re.search(r'\([a-z\s\/]+\)$', name_str):
            return "SNOMED-CT", "Systematized Nomenclature of Medicine -- Clinical Terms"
        
        # Si es numérico y no cayó en los anteriores, asumimos SNOMED-CT por defecto en este dataset de Synthea
        # excepto si es muy corto (<5 dígitos podría ser ICD-9 o 10 CM pero Synthea suele usar SNOMED)
        if len(code_str) > 5:
            return "SNOMED-CT", "Systematized Nomenclature of Medicine -- Clinical Terms"
    
    # 5. Otros / Desconocido
    return "Desconocido", "No identificado"

def process_codes():
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra el archivo {DATA_PATH}")
        return

    print(f"Leyendo {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Limpieza: Eliminar prefijo 'C-' si existe (común en este dataset para evitar que Excel interprete como número)
    # Pero el usuario pidió "Tal que las dos primeras columnas se queden tal cual", 
    # así que NO modificaremos las columnas originales en el output final,
    # solo usaremos una versión limpia para la identificación.
    
    coding_systems = []
    standard_names = []
    
    for index, row in df.iterrows():
        raw_code = str(row['code'])
        name = str(row['name'])
        
        # Limpieza para análisis
        clean_code = raw_code.replace('C-', '')
        
        system_code, system_name = identify_coding_system(clean_code, name)
        coding_systems.append(system_code)
        standard_names.append(system_name)
    
    # Añadir columas
    df['SISTEMA_CODIFICACION'] = coding_systems
    df['NOMBRE_ESTANDAR'] = standard_names
    
    # Guardar Excel
    print(f"Generando Excel en {OUTPUT_EXCEL}...")
    df.to_excel(OUTPUT_EXCEL, index=False)
    print("Proceso completado.")

if __name__ == "__main__":
    process_codes()
