import pandas as pd
import json

# Rutas a los ficheros
ruta_json = r"c:\Users\rtrin\OneDrive\Escritorio\FORMACION\MASTER\TFM\tfm-mulcia-2025\tfm-mulcia-2025\Variables_codif\LOINC\res_laboratorio.json"
ruta_dataset = r"c:\Users\rtrin\OneDrive\Escritorio\FORMACION\MASTER\TFM\tfm-mulcia-2025\tfm-mulcia-2025\data\synthea_30k.csv"
ruta_output = r"c:\Users\rtrin\OneDrive\Escritorio\FORMACION\MASTER\TFM\tfm-mulcia-2025\tfm-mulcia-2025\data\synthea_30k_loinc_solo_seleccionadas.csv"

def extraer_variables():
    # 1. Cargar el JSON (Mapeo de Nombre -> Codigo)
    with open(ruta_json, 'r', encoding='utf-8') as f:
        mapeo = json.load(f)
    
    # Invertir el diccionario para que sea {Codigo: Nombre}
    codigos_a_nombres = {codigo: nombre for nombre, codigo in mapeo.items()}
    codigos_deseados = list(codigos_a_nombres.keys())
    
    # 2. Cargar el dataset original
    print("Cargando el dataset...")
    df = pd.read_csv(ruta_dataset, low_memory=False)
    
    # Asegurarnos de mantener 'ptnum' y 'label' si existen
    columnas_base = [col for col in ['ptnum', 'label'] if col in df.columns]
    
    # Filtrar solo los códigos que realmente existen en el dataset
    codigos_validos = [c for c in codigos_deseados if c in df.columns]
    print(f"Buscando {len(codigos_deseados)} codigos, encontrados {len(codigos_validos)} en el dataset.")
    
    # 3. Seleccionar las columnas y renombrarlas
    columnas_a_extraer = columnas_base + codigos_validos
    df_filtrado = df[columnas_a_extraer]
    
    # Renombrar usando el diccionario invertido
    renombrado = {codigo: codigos_a_nombres[codigo] for codigo in codigos_validos}
    df_filtrado = df_filtrado.rename(columns=renombrado)
    
    # 4. Guardar el nuevo dataset
    df_filtrado.to_csv(ruta_output, index=False)
    print(f"Dataset guardado en {ruta_output}")
    print(f"Dimensiones finales: {df_filtrado.shape}")

if __name__ == "__main__":
    extraer_variables()
