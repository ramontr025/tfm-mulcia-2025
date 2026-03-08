import pandas as pd
import requests
import json
import time
import os
from collections import Counter

# --- Configuración ---
INPUT_FILE = 'data/codificaciones_identificadas.xlsx'
OUTPUT_FILE = 'data/analisis_ancestros_api.xlsx'
REPORT_FILE = 'ancestor_analysis_report.md'
CACHE_FILE = 'data/snomed_cache.json'

# API Config
# API Config
# Set to True if using local Docker instance (see guia_despliegue_local.md)
USE_LOCAL = True 

if USE_LOCAL:
    BASE_URL = 'http://localhost:8080' # Local instance listens at root
    EDITION = 'MAIN' 
else:
    BASE_URL = 'https://browser.ihtsdotools.org/snowstorm/snomed-ct' # Public API has proxy path
    EDITION = 'MAIN'

USER_AGENT = 'Python/3.9'

# --- Funciones API ---

def get_latest_version():
    if USE_LOCAL:
        return None
    url = f"{BASE_URL}/browser/{EDITION}/versions"
    headers = {'User-Agent': USER_AGENT}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            versions = response.json()
            versions.sort(key=lambda x: x['effectiveDate'], reverse=True)
            return versions[0]['version']
    except:
        pass
    return None

def get_ancestors(concept_id, version, cache):
    """
    Obtiene los ancestros de un concepto. Usa caché local.
    Retorna una lista de dicts {conceptId, term}.
    """
    concept_id = str(concept_id)
    if concept_id in cache:
        return cache[concept_id]
    
    if version:
        url = f"{BASE_URL}/browser/{EDITION}/{version}/concepts/{concept_id}/parents?form=inferred"
    else:
        url = f"{BASE_URL}/browser/{EDITION}/concepts/{concept_id}/parents?form=inferred"
    headers = {'User-Agent': USER_AGENT}
    
    print(f"DEBUG: Requesting {url}")
    MAX_RETRIES = 1
    base_delay = 2.0  # Start with 2 seconds
    
    for attempt in range(MAX_RETRIES):
        try:
            # Dynamic sleep
            current_delay = base_delay * (2 ** attempt) if attempt > 0 else base_delay
            if attempt > 0:
                print(f"  Retry {attempt}/{MAX_RETRIES} waiting {current_delay:.1f}s...")
            time.sleep(current_delay)
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                # Successfully got parents, now try ancestors
                # Note: valid SNOMED concepts should have parents.
                
                # Fetch ancestors endpoint
                if version:
                    url_ancestors = f"{BASE_URL}/browser/{EDITION}/{version}/concepts/{concept_id}/ancestors?form=inferred"
                else:
                    url_ancestors = f"{BASE_URL}/browser/{EDITION}/concepts/{concept_id}/ancestors?form=inferred"
                resp_anc = requests.get(url_ancestors, headers=headers, timeout=15)
                
                if resp_anc.status_code == 200:
                    data = resp_anc.json()
                    result = [{'conceptId': item['conceptId'], 'term': item['fsn']['term']} for item in data]
                    cache[concept_id] = result
                    return result
                elif resp_anc.status_code == 404:
                    print(f"  Ancestors not found for {concept_id}")
                    cache[concept_id] = [] 
                    return []
                    
            elif response.status_code == 404:
                print(f"  Concept {concept_id} not found")
                cache[concept_id] = []
                return []
            elif response.status_code == 429:
                print(f"  Rate limited (429). Backing off significantly.")
                time.sleep(10) # Extra penalty
                continue
            else:
                print(f"  Error {response.status_code}")

        except Exception as e:
            print(f"  Exception fetching {concept_id}: {e}")
    
    print(f"Failed to fetch {concept_id} after {MAX_RETRIES} attempts.")
    return []

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

# --- Función Principal ---

def main():
    print("Iniciando análisis de ancestros...")
    
    # 1. Cargar Datos
    if not os.path.exists(INPUT_FILE):
        print(f"No existe {INPUT_FILE}")
        return
    df = pd.read_excel(INPUT_FILE)
    
    # Filtrar solo SNOMED
    snomed_codes = df[df['SISTEMA_CODIFICACION'] == 'SNOMED-CT']
    print(f"Total códigos SNOMED a analizar: {len(snomed_codes)}")
    
    # 2. Configurar API
    version = get_latest_version()
    if not version and not USE_LOCAL:
        print("No se pudo obtener versión API. Usando '2023-06-30' por defecto.")
        version = '2023-06-30' # Fallback
    print(f"Usando versión API: {version}")
    
    # 3. Procesar Códigos
    cache = load_cache()
    ancestor_counter = Counter()
    ancestor_names = {}
    
    # Mapping: Concepto del dataset -> Lista de Ancestros IDs
    code_to_ancestors = {}

    total = len(snomed_codes)
    for idx, row in snomed_codes.iterrows():
        code = str(row['code']).replace('C-', '')
        
        ancestors = get_ancestors(code, version, cache)
        
        # Guardar en caché cada cierto tiempo
        if idx % 10 == 0:
            save_cache(cache)
            print(f"Procesado {idx}/{total}...")

        ancestor_ids = []
        for anc in ancestors:
            aid = anc['conceptId']
            term = anc['term']
            ancestor_ids.append(aid)
            ancestor_names[aid] = term
            ancestor_counter[aid] += 1
            
        code_to_ancestors[code] = ancestor_ids

    save_cache(cache)
    
    # 4. Generar Reporte de Ancestros
    # Convertir contador a DataFrame
    anc_data = []
    for aid, count in ancestor_counter.most_common():
        anc_data.append({
            'AncestorID': aid,
            'Termino': ancestor_names.get(aid, 'Unknown'),
            'Cantidad_Descendientes': count,
            'Porcentaje_Cobertura': (count / total) * 100
        })
    
    df_ancestors = pd.DataFrame(anc_data)
    
    # Filtrar conceptos demasiado genéricos (Root, Clinical finding, etc.) si se desea
    # Root: 138875005
    df_ancestors = df_ancestors[df_ancestors['AncestorID'] != '138875005'] 
    
    print(f"Guardando resultados en {OUTPUT_FILE}")
    df_ancestors.to_excel(OUTPUT_FILE, index=False)
    
    # 5. Generar Markdown
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("# Análisis de Ancestros SNOMED CT (Basado en API Real)\n\n")
        f.write(f"Total de códigos analizados: {total}\n\n")
        f.write("## Top 20 Ancestros Más Comunes\n")
        f.write("Estos son los conceptos padres que agrupan mayor número de códigos en tu dataset.\n\n")
        f.write("| Rank | Concepto Ancestro | Cantidad Códigos | % Cobertura |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        for i, row in df_ancestors.head(20).iterrows():
            f.write(f"| {i+1} | {row['Termino']} | {row['Cantidad_Descendientes']} | {row['Porcentaje_Cobertura']:.1f}% |\n")

    print("Finalizado.")

if __name__ == "__main__":
    main()
